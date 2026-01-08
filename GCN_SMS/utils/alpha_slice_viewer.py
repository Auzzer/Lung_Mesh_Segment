"""
Map per-tet alpha/log_alpha from mesh NPZ onto CT voxel grid and save mask.

Usage:
    python -m GCN_SMS.utils.alpha_slice_viewer \
        --mesh-npz checkpoints/mesh_results/Case1Pack_epoch000.npz \
        --ct data/Emory-4DCT/Case1Pack/NIFTI/case1_T00.nii.gz \
        --mask-out checkpoints/mesh_results/Case1Pack_mask.nii.gz \
        --use-log-alpha \
        --return-kpa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.spatial import cKDTree

from .map_alpha_to_ct import NU, load_alpha, tet_to_node

# Add lung_project_git to path for imports
_lung_project_path = Path(__file__).resolve().parent.parent.parent / "lung_project_git"
if str(_lung_project_path) not in sys.path:
    sys.path.insert(0, str(_lung_project_path))

from project.core.transforms import world_to_voxel_coords 


def mesh_to_ct_coords(
    mesh_npz: str | Path,
    ct_path: str | Path,
    use_log_alpha: bool = True,
    base_time: str = "T00",
    return_kpa: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Map mesh nodes into CT voxel space and compute per-node stiffness.

    Returns
    -------
    ct_vol:
        CT volume as ``float32`` array with shape ``(I, J, K)``.
    vox_coords:
        ``(N, 3)`` node locations in voxel indices (i, j, k).
    E_node:
        Per-node Young's modulus values aligned with ``vox_coords``.
    affine:
        4x4 affine of the CT image.
    """
    mesh_npz = Path(mesh_npz)
    ct_path = Path(ct_path)

    pts_m, tets, alpha_tet = load_alpha(mesh_npz, use_log_alpha=use_log_alpha)

    if base_time.lower() not in ct_path.stem.lower():
        raise ValueError(f"CT path {ct_path} does not appear to match base time {base_time}")

    ct_img = nib.load(str(ct_path))
    ct_aff = ct_img.affine
    ct_vol = ct_img.get_fdata().astype(np.float32)

    # Per-node alpha and Young's modulus
    alpha_node = tet_to_node(alpha_tet, tets, pts_m.shape[0])
    E_node = 2.0 * (1.0 + NU) * alpha_node  # Young's modulus
    if return_kpa:
        E_node = E_node / 1000.0

    # World (m) -> voxel indices using core.transforms
    pts_mm = pts_m * 1e3
    pts_mm_t = torch.from_numpy(pts_mm)
    ct_aff_t = torch.from_numpy(ct_aff.astype(np.float32))
    vox_t = world_to_voxel_coords(pts_mm_t, ct_aff_t)
    vox = vox_t.numpy().astype(np.float32)  # (N, 3)

    return ct_vol, vox, E_node.astype(np.float32), ct_aff


def interpolate_to_ct(
    mesh_npz: str | Path,
    ct_path: str | Path,
    use_log_alpha: bool = True,
    base_time: str = "T00",
    return_kpa: bool = False,
    mask_path: str | Path | None = None,
    mask_out_path: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate per-tet alpha / log_alpha from a mesh NPZ onto the CT voxel grid.
    Save a CT-aligned mask (from ``mask_path`` or ``E_vol > 0``).
    """
    ct_vol, vox, E_node, ct_aff = mesh_to_ct_coords(
        mesh_npz=mesh_npz,
        ct_path=ct_path,
        use_log_alpha=use_log_alpha,
        base_time=base_time,
        return_kpa=return_kpa,
    )
    ct_shape = ct_vol.shape[:3]

    # Bounding box + padding in voxel space
    pad = 2.0
    vmin = np.clip(np.floor(vox.min(axis=0) - pad), 0, np.array(ct_shape) - 1)
    vmax = np.clip(np.ceil(vox.max(axis=0) + pad), 0, np.array(ct_shape) - 1)

    gi = np.arange(int(vmin[0]), int(vmax[0]) + 1)
    gj = np.arange(int(vmin[1]), int(vmax[1]) + 1)
    gk = np.arange(int(vmin[2]), int(vmax[2]) + 1)

    grid_i, grid_j, grid_k = np.meshgrid(gi, gj, gk, indexing="ij")
    grid_points = (
        np.stack([grid_i, grid_j, grid_k], axis=-1)
        .reshape(-1, 3)
        .astype(np.float32)
    )

    # Nearest neighbour interpolation via KDTree 
    tree = cKDTree(vox)
    _, idx = tree.query(grid_points, k=1, workers=-1)
    E_vals = E_node[idx].astype(np.float32)

    E_vol = np.zeros(ct_shape, dtype=np.float32)
    E_vol[np.ix_(gi, gj, gk)] = E_vals.reshape(len(gi), len(gj), len(gk))

    mask = None
    # restrict to lung mask
    if mask_path is not None:
        mask_img = nib.load(str(mask_path))
        mask_data = mask_img.get_fdata()
        if mask_data.shape[:3] != ct_shape:
            raise ValueError(
                f"Mask shape {mask_data.shape[:3]} does not match CT shape {ct_shape}"
            )
        mask = mask_data > 0.5
        E_vol[~mask] = 0.0

    if mask_out_path is not None:
        if mask is None:
            mask = E_vol > 0.0
        mask_out_path = Path(mask_out_path)
        mask_out_path.parent.mkdir(parents=True, exist_ok=True)
        ct_img = nib.load(str(ct_path))
        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), ct_img.affine, ct_img.header),
            str(mask_out_path),
        )

    return ct_vol, E_vol, ct_aff


def main() -> None:
    """Command-line interface for interpolating mesh values to CT grid."""
    parser = argparse.ArgumentParser(
        description="Interpolate per-tet alpha/log_alpha from mesh NPZ onto CT voxel grid."
    )
    parser.add_argument(
        "--mesh-npz",
        required=True,
        help="NPZ file with mesh_points, tetrahedra, and alpha/log_alpha.",
    )
    parser.add_argument(
        "--ct",
        required=True,
        help="CT NIfTI path (target grid/affine).",
    )
    parser.add_argument(
        "--mask-out",
        required=True,
        help="Output path for CT-aligned mask NIfTI.",
    )
    parser.add_argument(
        "--mask-path",
        default=None,
        help="Optional input lung mask to restrict interpolation.",
    )
    parser.add_argument(
        "--use-log-alpha",
        action="store_true",
        help="Interpret NPZ field as log_alpha and exponentiate.",
    )
    parser.add_argument(
        "--return-kpa",
        action="store_true",
        help="Convert Young's modulus to kPa (divide by 1000).",
    )
    parser.add_argument(
        "--base-time",
        default="T00",
        help="Base time token for CT validation (default: T00).",
    )
    args = parser.parse_args()

    ct_vol, E_vol, ct_aff = interpolate_to_ct(
        mesh_npz=args.mesh_npz,
        ct_path=args.ct,
        use_log_alpha=args.use_log_alpha,
        base_time=args.base_time,
        return_kpa=args.return_kpa,
        mask_path=args.mask_path,
        mask_out_path=args.mask_out,
    )
    
    print(f"[alpha_slice_viewer] Saved mask to {args.mask_out}")
    print(f"[alpha_slice_viewer] E volume range: [{E_vol.min():.2f}, {E_vol.max():.2f}]")


if __name__ == "__main__":
    main()

