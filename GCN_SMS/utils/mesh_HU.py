"""
Compute per-vertex and per-tetra HU values from a CT volume and save them.

The inverse-mapping model uses the preassembled broadcast operator C[k] (if
present in the preprocessing NPZ) and solves the per-tet least-squares
pseudo-inverse h_k = pinv(C[k]) u_k. If C[k] is absent we fall back to the
mean of nodal HU samples (C[k]=1_4x1). HU sampling is performed with
trilinear interpolation on the CT grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np
import torch


def world_to_voxel(affine_inv: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    """Map world coordinates to voxel coordinates via the inverse affine."""
    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    hom = torch.cat([xyz, ones], dim=1)
    ijk = hom @ affine_inv.T
    return ijk[:, :3]


def trilinear_sample(volume: torch.Tensor, ijk: torch.Tensor) -> torch.Tensor:
    """
    Trilinear interpolation for an array of (i, j, k) voxel coordinates.

    Args:
        volume: Tensor with shape (D, H, W) on the target device.
        ijk: Tensor with shape (N, 3) of floating point voxel coordinates.

    Returns:
        Tensor with shape (N,) of interpolated values.
    """
    d, h, w = volume.shape

    i = ijk[:, 0]
    j = ijk[:, 1]
    k = ijk[:, 2]

    i0 = torch.floor(i).long().clamp(0, d - 1)
    j0 = torch.floor(j).long().clamp(0, h - 1)
    k0 = torch.floor(k).long().clamp(0, w - 1)
    i1 = (i0 + 1).clamp(0, d - 1)
    j1 = (j0 + 1).clamp(0, h - 1)
    k1 = (k0 + 1).clamp(0, w - 1)

    di = i - i0.to(i.dtype)
    dj = j - j0.to(j.dtype)
    dk = k - k0.to(k.dtype)

    c000 = volume[i0, j0, k0]
    c100 = volume[i1, j0, k0]
    c010 = volume[i0, j1, k0]
    c110 = volume[i1, j1, k0]
    c001 = volume[i0, j0, k1]
    c101 = volume[i1, j0, k1]
    c011 = volume[i0, j1, k1]
    c111 = volume[i1, j1, k1]

    c00 = c000 * (1.0 - di) + c100 * di
    c01 = c001 * (1.0 - di) + c101 * di
    c10 = c010 * (1.0 - di) + c110 * di
    c11 = c011 * (1.0 - di) + c111 * di
    c0 = c00 * (1.0 - dj) + c10 * dj
    c1 = c01 * (1.0 - dj) + c11 * dj
    return c0 * (1.0 - dk) + c1 * dk


def _sample_vertices(
    volume: torch.Tensor, ijk: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """
    Chunked trilinear sampling to avoid oversized temporary allocations.

    Args:
        volume: CT tensor on target device (D, H, W).
        ijk: Voxel coordinates for all vertices (N, 3).
        chunk_size: Maximum vertices to interpolate per chunk.

    Returns:
        Tensor of shape (N,) with sampled HU values.
    """
    out = torch.empty((ijk.shape[0],), device=volume.device, dtype=volume.dtype)
    for start in range(0, ijk.shape[0], chunk_size):
        end = min(start + chunk_size, ijk.shape[0])
        out[start:end] = trilinear_sample(volume, ijk[start:end])
    return out


def compute_mesh_hu(
    pre_npz_in: Path,
    ct_path: Path,
    pre_npz_out: Path | None = None,
    device: str = "auto",
    chunk_size: int = 200_000,
) -> Path:
    """
    Compute HU at mesh vertices and tetrahedra, and save alongside preprocessing NPZ data.

    Args:
        pre_npz_in: Existing preprocessing NPZ containing mesh_points and tetrahedra.
        ct_path: CT volume (NIfTI) path.
        pre_npz_out: Output NPZ path (default: <stem>_with_hu.npz).
        device: "auto", "cuda", or "cpu" for computation.
        chunk_size: Vertex batch size for interpolation to limit memory.
    """
    pre_npz_in = Path(pre_npz_in)
    ct_path = Path(ct_path)
    if pre_npz_out is None:
        pre_npz_out = pre_npz_in.with_name(pre_npz_in.stem + "_with_hu.npz")

    if not pre_npz_in.exists():
        raise FileNotFoundError(f"preprocessed npz not found: {pre_npz_in}")
    if not ct_path.exists():
        raise FileNotFoundError(f"CT volume not found: {ct_path}")

    data = np.load(pre_npz_in, allow_pickle=True)
    required = ("mesh_points", "tetrahedra")
    for key in required:
        if key not in data.files:
            raise ValueError(f"Preprocessed data missing '{key}'")

    points = data["mesh_points"].astype(np.float32)
    tets = data["tetrahedra"].astype(np.int64)
    coeff_matrix = (
        data["coefficient_matrix"].astype(np.float32)
        if "coefficient_matrix" in data.files
        else None
    )

    img = nib.load(str(ct_path))
    ct_hu = img.get_fdata().astype(np.float32)

    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

    volume_t = torch.from_numpy(ct_hu).to(torch_device)
    affine_inv_t = torch.from_numpy(np.linalg.inv(img.affine).astype(np.float32)).to(torch_device)
    points_t = torch.from_numpy(points).to(torch_device)

    ijk = world_to_voxel(affine_inv_t, points_t)
    vertex_hu = _sample_vertices(volume_t, ijk, chunk_size=chunk_size)

    tets_t = torch.from_numpy(tets).to(torch_device)
    u_k = vertex_hu[tets_t]  # (M, 4)

    hu_param_t: torch.Tensor | None = None
    # Use preassembled C[k] (M x 4 x P), solve h_k = pinv(C[k]) u_k per tet.
    C_t = torch.from_numpy(coeff_matrix).to(torch_device)  # (M, 4, P)
    C_pinv = torch.linalg.pinv(C_t)  # (M, P, 4) batched per tet
    recon = torch.bmm(C_pinv, C_t)  # (M, P, P)
    eye_p = torch.eye(C_t.shape[-1], device=torch_device, dtype=C_t.dtype)
    per_tet_fro = torch.linalg.norm(recon - eye_p.expand_as(recon), dim=(1, 2))
    pinv_mean = per_tet_fro.mean().item()
    pinv_max = per_tet_fro.max().item()
    pinv_p95 = torch.quantile(per_tet_fro, 0.95).item()
    print(
        "[mesh_HU] ||pinv(C)@C - I||_F per tet: "
        f"mean={pinv_mean:.3e}, p95={pinv_p95:.3e}, max={pinv_max:.3e} "
        f"(P={C_t.shape[-1]})"
    )
    hu_param_t = torch.bmm(C_pinv, u_k.unsqueeze(-1)).squeeze(-1)  # (M, P)
    tet_hu = hu_param_t





    hu_vertex_np = vertex_hu.detach().cpu().numpy()
    hu_tet_np = tet_hu.detach().cpu().numpy()

    out_dict: Dict[str, np.ndarray] = {key: data[key] for key in data.files}
    out_dict["hu_vertex"] = hu_vertex_np
    out_dict["hu_tetra"] = hu_tet_np
    if hu_param_t is not None:
        out_dict["hu_param_pinv"] = hu_tet_np
    np.savez(pre_npz_out, **out_dict)

    print(
        "[mesh_HU] Vertex HU stats: min={:.2f}, max={:.2f}".format(
            hu_vertex_np.min() if hu_vertex_np.size else float("nan"),
            hu_vertex_np.max() if hu_vertex_np.size else float("nan"),
        )
    )
    print(
        "[mesh_HU] Tetra HU stats: min={:.2f}, max={:.2f}".format(
            hu_tet_np.min() if hu_tet_np.size else float("nan"),
            hu_tet_np.max() if hu_tet_np.size else float("nan"),
        )
    )
    print(f"[mesh_HU] Wrote HU fields to {pre_npz_out}")
    return Path(pre_npz_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-vertex and per-tetra HU values from a CT volume."
    )
    parser.add_argument(
        "--preprocessed",
        default="EXPs/Emory4DCT/data_processed_deformation/Case10Pack/Case10Pack_T00_to_T10_lung_regions_11.npz",
        help="Existing preprocessing npz (must contain mesh_points and tetrahedra).",
    )
    parser.add_argument(
        "--ct", 
        default="data/Emory-4DCT/Case10Pack/NIFTI/case10_T00.nii.gz", 
        help="Fixed CT (NIFTI) path.")
    parser.add_argument(
        "--output",
        default="EXPs/Emory4DCT/data_processed_deformation/Case10Pack/Case10Pack_T10_to_T00_with_hu.npz",
        help="Output npz path (default: <preprocessed stem>_with_hu.npz).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["auto", "cuda", "cpu"],
        help="Device for interpolation/aggregation.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Vertex batch size for trilinear interpolation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_mesh_hu(
        pre_npz_in=Path(args.preprocessed),
        ct_path=Path(args.ct),
        pre_npz_out=Path(args.output) if args.output else None,
        device=args.device,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
