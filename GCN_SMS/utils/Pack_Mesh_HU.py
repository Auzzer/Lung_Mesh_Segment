"""
Pack per-tetra HU across a forward time sequence starting from a single T00 mesh.

Forward workflow:
  * Load the base preprocessing NPZ (mesh/tets at T00).
  * Discover forward CorrField displacements and fixed CTs in the provided folders.
    Steps are built by sorting all time stamps and using consecutive pairs
    (e.g., T00->T10, T10->T20, ...).
  * Sequentially warp the mesh with each displacement (defined in the fixed CT space),
    recompute per-tet HU via h_k = pinv(C_k) u_k (fallback to nodal mean beacuase
    the matrix is insigular sometimes), and accumulate.
  * Save per-step HU and their mean in one NPZ, keeping the original T00 mesh/tets.

sample usage:
python GCN_SMS/utils/Pack_Mesh_HU.py \
  --base-preprocessed data_processed_deformation/Case10Pack/Case10Pack_T00_to_T10_lung_regions_11.npz \
  --corrfield-dir data/Emory-4DCT/Case10Pack/CorrField \
  --ct-dir data/Emory-4DCT/Case10Pack/NIFTI \
  --output data_processed_deformation/hu_packs/Case10Pack/Case10Pack_T00_forward_hu.npz \
  --mask-output mesh_edge_mask/Case10Pack_mesh_mask.nii.gz \
  --mask-edge-step 0.5 

  
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator

TIME_RE = re.compile(r"T(\d{2})", re.IGNORECASE)


def hu_to_alpha(hu_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map HU -> E(H) (kPa) -> alpha (Pa) using paper mapping:
      E(H) = 1 kPa  for H<=-950
           = 1 + 19/750*(H+950) for -950<H<-200
           = 20 kPa for H>=-200
      alpha = 0.5 * E_Pa, so alpha in [500, 10000] Pa.

    Args:
        hu_mean: (M, P) per-tet HU (uses column 0)
    Returns:
        alpha_init: (M,) in Pa
        log_alpha_init: (M,) log(alpha_init)
    """
    H = hu_mean[:, 0].astype(np.float64)
    H_clamped = np.clip(H, -950.0, -200.0)
    E_kPa = 1.0 + 19.0 * (H_clamped + 950.0) / 750.0
    E_Pa = E_kPa * 1000.0
    alpha = 0.5 * E_Pa
    alpha = np.clip(alpha, 500.0, 1.0e4)
    log_alpha = np.log(alpha)
    return alpha.astype(np.float32), log_alpha.astype(np.float32)


def _parse_time(token: str) -> str:
    m = TIME_RE.search(token)
    if not m:
        raise ValueError(f"Cannot parse time token from {token}")
    return f"T{int(m.group(1)):02d}"


def _time_value(t: str) -> int:
    return int(t[1:])


def _discover_forward_sequence(corr_dir: Path, base_time: str) -> Tuple[str, List[Tuple[str, str, Path]]]:
    """
    Discover subject and forward steps (src_time, tgt_time, displacement_path)
    using consecutive sorted timestamps present in CorrField filenames.
    """
    corr_files = sorted(corr_dir.glob("*.nii.gz"))
    if not corr_files:
        raise FileNotFoundError(f"No CorrField files found in {corr_dir}")

    # Infer subject prefix from first file: <subject>_Txx_Tyy
    stem = corr_files[0].stem
    subj_prefix = stem.split("_T")[0]

    edges: Dict[Tuple[str, str], Path] = {}
    times: set[str] = set()
    for f in corr_files:
        parts = f.stem.split("_")
        if len(parts) < 3:
            continue
        src = _parse_time(parts[-2])
        tgt = _parse_time(parts[-1])
        edges[(src, tgt)] = f
        times.update([src, tgt])

    if base_time not in times:
        times.add(base_time)
    ordered = sorted(times, key=_time_value)
    if base_time not in ordered:
        raise ValueError(f"Base time {base_time} not found/parsed in CorrField files.")

    start_idx = ordered.index(base_time)
    steps: List[Tuple[str, str, Path]] = []
    for i in range(start_idx, len(ordered) - 1):
        src, tgt = ordered[i], ordered[i + 1]
        key = (src, tgt)
        if key not in edges:
            raise FileNotFoundError(f"Missing CorrField for step {src}->{tgt}")
        steps.append((src, tgt, edges[key]))
    if not steps:
        raise RuntimeError("No forward steps discovered from CorrField directory.")
    return subj_prefix, steps


def _build_tet_adjacency(tets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast face-based adjacency for tetrahedra.
    Returns (neighbor_indices, neighbor_offsets) CSR-style arrays.
    """
    tets = tets.astype(np.int64, copy=False)
    m = tets.shape[0]
    faces = np.concatenate(
        [
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ],
        axis=0,
    )
    faces = np.sort(faces, axis=1)
    owners = np.repeat(np.arange(m, dtype=np.int64), 4)

    perm = np.lexsort(faces.T)
    faces_sorted = faces[perm]
    owners_sorted = owners[perm]

    neighbors: List[set[int]] = [set() for _ in range(m)]
    i = 0
    nfaces = faces_sorted.shape[0]
    while i < nfaces:
        j = i + 1
        while j < nfaces and np.array_equal(faces_sorted[j], faces_sorted[i]):
            j += 1
        if j - i > 1:
            tet_ids = owners_sorted[i:j]
            for a in tet_ids:
                for b in tet_ids:
                    if a != b:
                        neighbors[a].add(int(b))
        i = j

    offsets = np.zeros(m + 1, dtype=np.int64)
    for k in range(m):
        offsets[k + 1] = offsets[k] + len(neighbors[k])
    indices = np.empty(offsets[-1], dtype=np.int64)
    for k in range(m):
        start, end = offsets[k], offsets[k + 1]
        if end > start:
            indices[start:end] = np.array(sorted(neighbors[k]), dtype=np.int64)
    return indices.astype(np.int32), offsets.astype(np.int32)


def _world_to_voxel(affine_inv: np.ndarray, xyz_world: np.ndarray) -> np.ndarray:
    """Map world (mm) coordinates to voxel coordinates using affine inverse."""
    hom = np.concatenate([xyz_world, np.ones((xyz_world.shape[0], 1))], axis=1)
    ijk = hom @ affine_inv.T
    return ijk[:, :3]


def _interp_displacement(disp_img: nib.Nifti1Image, points_m: np.ndarray) -> np.ndarray:
    """
    Interpolate a 4D CorrField displacement at mesh vertices.
    Displacement data assumed to be in mm in the fixed CT space.
    """
    disp = disp_img.get_fdata()
    affine_inv = np.linalg.inv(disp_img.affine)
    ijk = _world_to_voxel(affine_inv, points_m * 1e3)  # metres -> mm -> voxel

    grid = tuple(np.arange(n) for n in disp.shape[:3])
    out = np.zeros_like(points_m, dtype=np.float64)
    for c in range(3):
        interp = RegularGridInterpolator(
            grid, disp[..., c], bounds_error=False, fill_value=0.0, method="linear"
        )
        out[:, c] = interp(ijk)
    return out * 1e-3  # mm -> m


def _build_mesh_mask(
    ct_img: nib.Nifti1Image,
    points_m: np.ndarray,
    tets: np.ndarray,
    edge_step: float = 0.5,
) -> np.ndarray:
    """
    Rasterize mesh barycenters (value=1) and edges (value=2) into a mask
    aligned with the reference CT.
    """
    volume_shape = ct_img.shape[:3]
    mask = np.zeros(volume_shape, dtype=np.uint8)
    affine_inv = np.linalg.inv(ct_img.affine)

    points_vox = _world_to_voxel(affine_inv, points_m * 1e3)  # (N, 3)

    # Barycenters
    bary_vox = _world_to_voxel(affine_inv, points_m[tets].mean(axis=1) * 1e3)
    bary_idx = np.round(bary_vox).astype(np.int64)
    in_bounds = (
        (bary_idx >= 0)
        & (bary_idx < np.asarray(volume_shape, dtype=np.int64))
    ).all(axis=1)
    bary_idx = bary_idx[in_bounds]
    if bary_idx.size:
        mask[bary_idx[:, 0], bary_idx[:, 1], bary_idx[:, 2]] = 1

    # Unique mesh edges
    tet_edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
    edges = np.sort(tets[:, tet_edges].reshape(-1, 2), axis=1)
    edges = np.unique(edges, axis=0)
    max_idx = np.asarray(volume_shape, dtype=np.int64)

    for v0, v1 in edges:
        p0 = points_vox[v0]
        p1 = points_vox[v1]
        diff = p1 - p0
        steps = int(max(1, np.ceil(np.linalg.norm(diff) / edge_step)))
        line = p0 + diff * (np.linspace(0.0, 1.0, steps + 1)[:, None])
        idx = np.round(line).astype(np.int64)
        in_bounds = ((idx >= 0) & (idx < max_idx)).all(axis=1)
        idx = idx[in_bounds]
        if idx.size:
            mask[idx[:, 0], idx[:, 1], idx[:, 2]] = 2

    return mask


def _trilinear_sample_torch(volume: torch.Tensor, ijk: torch.Tensor) -> torch.Tensor:
    """
    Torch trilinear sampler for HU values using grid_sample to mirror
    lung_project_git/project/core/interpolation.py behavior (align_corners,
    border padding).
    """
    ijk = ijk[..., :3]  # drop homogeneous column if present
    shape = ijk.new_tensor(volume.shape, dtype=ijk.dtype)  # (3,)
    grid = (ijk / (shape - 1.0)) * 2.0 - 1.0  # normalize to [-1, 1]
    grid = grid.flip(-1).view(1, 1, 1, -1, 3)  # x, y, z order for grid_sample
    sampled = F.grid_sample(
        volume.view(1, 1, *volume.shape),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.view(-1)


def _compute_hu_for_mesh(
    points_m: np.ndarray,
    tets: np.ndarray,
    coeff_matrix: np.ndarray | None,
    ct_img: nib.Nifti1Image,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    """Sample HU at mesh vertices and compute per-tet HU via pinv(C_k)."""
    ct = ct_img.get_fdata().astype(np.float32)
    affine_inv = np.linalg.inv(ct_img.affine).astype(np.float32)
    points_mm = points_m.astype(np.float32) * 1e3

    volume_t = torch.from_numpy(ct).to(device)
    affine_inv_t = torch.from_numpy(affine_inv).to(device)
    pts_t = torch.from_numpy(points_mm).to(device)

    ones = torch.ones((pts_t.shape[0], 1), device=device, dtype=pts_t.dtype)
    hom = torch.cat([pts_t, ones], dim=1)
    ijk = hom @ affine_inv_t.T

    vertex_hu = torch.empty((pts_t.shape[0],), device=device, dtype=volume_t.dtype)
    for start in range(0, pts_t.shape[0], chunk_size):
        end = min(start + chunk_size, pts_t.shape[0])
        vertex_hu[start:end] = _trilinear_sample_torch(volume_t, ijk[start:end])

    tets_t = torch.from_numpy(tets.astype(np.int64)).to(device)
    u_k = vertex_hu[tets_t]  # (M, 4)
    coeff_matrix = None # keep skipping pinv part
    if coeff_matrix is not None:
        C_t = torch.from_numpy(coeff_matrix.astype(np.float32)).to(device)  # (M, 4, P)
        C_pinv = torch.linalg.pinv(C_t)  # (M, P, 4)
        tet_hu = torch.bmm(C_pinv, u_k.unsqueeze(-1)).squeeze(-1)  # (M, P)
        """
        remove this part becaause hu_tetra_mean stats: min=-63666.63, max=73381.91
        some tetras' vertice value skip, causing the pinv result unproperly large or small
        """
    else:
        ones4 = torch.ones((4, 1), device=device, dtype=vertex_hu.dtype)
        tet_hu = (u_k @ ones4).squeeze(-1) * 0.25  # (M,)
        tet_hu = tet_hu.unsqueeze(-1)

    return tet_hu.detach().cpu().numpy()


def pack_mesh_hu_sequence(
    base_npz: Path,
    corrfield_dir: Path,
    ct_dir: Path,
    output: Path,
    base_time: str | None = None,
    device: str = "auto",
    chunk_size: int = 200_000,
    mask_output: Path | None = None,
    mask_edge_step: float = 0.5,
) -> Path:
    """Forward-only HU accumulation over all timestamps discovered in corrfield_dir."""
    base_npz = Path(base_npz)
    if not base_npz.exists():
        raise FileNotFoundError(base_npz)
    data = np.load(base_npz, allow_pickle=True)
    for key in ("mesh_points", "tetrahedra"):
        if key not in data:
            raise ValueError(f"Base NPZ missing '{key}'")

    # Infer base time from filename if not provided
    if base_time is None:
        m = TIME_RE.search(base_npz.stem)
        base_time = f"T{int(m.group(1)):02d}" if m else "T00"

    subj, steps = _discover_forward_sequence(Path(corrfield_dir), base_time)

    points_m = data["mesh_points"].astype(np.float64)  # m
    tets = data["tetrahedra"].astype(np.int64)
    coeff_matrix = data["coefficient_matrix"].astype(np.float32) if "coefficient_matrix" in data else None
    tet_neighbor_indices = data["tet_neighbor_indices"].astype(np.int32) if "tet_neighbor_indices" in data else None
    tet_neighbor_offsets = data["tet_neighbor_offsets"].astype(np.int32) if "tet_neighbor_offsets" in data else None
    if tet_neighbor_indices is None or tet_neighbor_offsets is None:
        tet_neighbor_indices, tet_neighbor_offsets = _build_tet_adjacency(tets)
        print(f"[Pack_Mesh_HU] Built adjacency: {tet_neighbor_indices.shape[0]} edges")

    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    hu_steps = []
    bary_steps = []
    current_pts = points_m.copy()
    last_ct_img: nib.Nifti1Image | None = None

    for src, tgt, disp_path in steps:
        ct_candidates = [
            Path(ct_dir) / f"{subj}_{src}.nii.gz",
            Path(ct_dir) / f"{subj}_{src.lower()}.nii.gz",
            Path(ct_dir) / f"{subj}_{src.upper()}.nii.gz",
        ]
        ct_path = next((p for p in ct_candidates if p.exists()), None)
        if ct_path is None:
            raise FileNotFoundError(f"CT not found for {src}; tried: {ct_candidates}")
        disp_img = nib.load(str(disp_path))
        ct_img = nib.load(str(ct_path))

        # Warp vertices forward
        disp_vec = _interp_displacement(disp_img, current_pts)
        current_pts = current_pts + disp_vec

        # Barycenter (updated geometry) for reference/QA
        bary = current_pts[tets].mean(axis=1)  # (M, 3)

        # Compute HU on warped mesh
        tet_hu = _compute_hu_for_mesh(
            current_pts, tets, coeff_matrix, ct_img, torch_device, chunk_size
        )
        hu_steps.append(tet_hu)
        bary_steps.append(bary)
        print(f"[Pack_Mesh_HU] step {src}->{tgt}: HU shape {tet_hu.shape}")
        last_ct_img = ct_img

    hu_steps_np = np.stack(hu_steps, axis=0)  # (S, M, P)
    hu_mean = hu_steps_np.mean(axis=0)
    bary_steps_np = np.stack(bary_steps, axis=0)  # (S, M, 3)
    bary_mean = bary_steps_np.mean(axis=0)

    alpha_init, log_alpha_init = hu_to_alpha(hu_mean)

    out_dict = {key: data[key] for key in data.files}
    out_dict["hu_tetra_steps"] = hu_steps_np
    out_dict["hu_tetra_mean"] = hu_mean
    out_dict["barycenter_steps"] = bary_steps_np
    out_dict["barycenter_mean"] = bary_mean
    out_dict["tet_neighbor_indices"] = tet_neighbor_indices
    out_dict["tet_neighbor_offsets"] = tet_neighbor_offsets
    out_dict["mesh_points"] = points_m  # original T00 mesh
    out_dict["alpha_init_from_hu"] = alpha_init
    out_dict["log_alpha_init_from_hu"] = log_alpha_init
    np.savez(output, **out_dict)
    print(f"[Pack_Mesh_HU] Saved HU sequence to {output}")
    print(
        "[Pack_Mesh_HU] hu_tetra_mean stats: min={:.2f}, max={:.2f}".format(
            hu_mean.min(), hu_mean.max()
        )
    )
    if mask_output is not None:
        if last_ct_img is None:
            raise RuntimeError("Mask requested but no CT image was processed.")
        mask = _build_mesh_mask(last_ct_img, current_pts, tets, edge_step=mask_edge_step)
        mask_out_path = Path(mask_output)
        mask_out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(mask, last_ct_img.affine, last_ct_img.header), str(mask_out_path))
        print(f"[Pack_Mesh_HU] Saved mesh mask (bary=1, edges=2) to {mask_out_path}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward HU accumulation starting from a single T00 mesh (auto-discover steps)."
    )
    parser.add_argument("--base-preprocessed", required=True, help="Base NPZ (T00) with mesh_points/tetrahedra/(coefficient_matrix).")
    parser.add_argument("--corrfield-dir", required=True, help="Directory containing CorrField displacement NIfTIs (fixed->moving).")
    parser.add_argument("--ct-dir", required=True, help="Directory containing fixed CT NIfTIs named <subject>_Txx.nii.gz.")
    parser.add_argument("--output", required=True, help="Output NPZ path.")
    parser.add_argument("--base-time", help="Base time token (e.g., T00). Inferred from base-preprocessed filename if omitted.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for HU sampling.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Vertex batch size for trilinear interpolation.",
    )
    parser.add_argument(
        "--mask-output",
        help="Optional NIfTI output path for a mesh mask on the final CT (barycenters=1, edges=2).",
    )
    parser.add_argument(
        "--mask-edge-step",
        type=float,
        default=0.5,
        help="Step size in voxels when rasterizing edges into the mask.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pack_mesh_hu_sequence(
        base_npz=Path(args.base_preprocessed),
        corrfield_dir=Path(args.corrfield_dir),
        ct_dir=Path(args.ct_dir),
        output=Path(args.output),
        base_time=args.base_time,
        device=args.device,
        chunk_size=args.chunk_size,
        mask_output=Path(args.mask_output) if args.mask_output else None,
        mask_edge_step=args.mask_edge_step,
    )


if __name__ == "__main__":
    main()
