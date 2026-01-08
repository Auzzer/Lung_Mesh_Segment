#!/usr/bin/env python
"""
Map per-tet alpha/log_alpha from a saved mesh result NPZ onto a CT-aligned NIfTI volume.
Uses CPU cKDTree nearest-neighbor over the CT grid (bounding box of the mesh) to avoid GPU OOM.

Usage:
  python GCN_SMS/utils/map_alpha_to_ct.py \
    --mesh-npz checkpoints/mesh_results/Case1Pack_epoch000_omegaU0.000_omegaA0.000.npz \
    --ct data/Emory-4DCT/Case1Pack/NIFTI/case1_T00.nii.gz \
    --out checkpoints/mesh_results/Case1Pack_epoch000_E_on_ct.nii.gz \
    --use-log-alpha
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

# Add lung_project_git to path for imports
_lung_project_path = Path(__file__).resolve().parent.parent.parent / "lung_project_git"
if str(_lung_project_path) not in sys.path:
    sys.path.insert(0, str(_lung_project_path))

from project.core.transforms import world_to_voxel_coords  

NU = 0.4  # Poisson ratio for E conversion: E = 2*(1+nu)*alpha


def load_alpha(mesh_npz: Path, use_log_alpha: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(mesh_npz, allow_pickle=True)
    pts_m = np.asarray(data["mesh_points"]).astype(np.float32)  # metres
    tets = np.asarray(data["tetrahedra"]).astype(np.int64)
    if use_log_alpha:
        log_alpha = np.asarray(data["log_alpha"]).astype(np.float32)
        alpha = np.exp(log_alpha)
    else:
        alpha = np.asarray(data["alpha"]).astype(np.float32)
    return pts_m, tets, alpha.squeeze()


def tet_to_node(alpha_tet: np.ndarray, tets: np.ndarray, n_nodes: int) -> np.ndarray:
    """Average tet values to nodes."""
    accum = np.zeros(n_nodes, dtype=np.float64)
    counts = np.zeros(n_nodes, dtype=np.int64)
    for tet_val, tet in zip(alpha_tet, tets):
        accum[tet] += tet_val
        counts[tet] += 1
    counts[counts == 0] = 1
    return (accum / counts).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Map per-tet alpha/log_alpha to a CT-aligned NIfTI volume.")
    ap.add_argument("--mesh-npz", required=True, help="NPZ with mesh_points, tetrahedra, alpha/log_alpha.")
    ap.add_argument("--ct", required=True, help="CT NIfTI path (target grid/affine; expected T00).")
    ap.add_argument("--out", required=True, help="Output NIfTI path for interpolated field.")
    ap.add_argument("--use-log-alpha", action="store_true", help="Interpret NPZ field as log_alpha and exponentiate.")
    ap.add_argument("--base-time", default="T00", help="Base time token for CT (default: T00).")
    args = ap.parse_args()

    mesh_npz = Path(args.mesh_npz)
    ct_path = Path(args.ct)
    out_path = Path(args.out)

    pts_m, tets, alpha_tet = load_alpha(mesh_npz, args.use_log_alpha)
    if args.base_time.lower() not in ct_path.stem.lower():
        raise ValueError(f"CT path {ct_path} does not appear to match base time {args.base_time}")
    ct_img = nib.load(str(ct_path))
    ct_aff = ct_img.affine
    ct_shape = ct_img.shape[:3]

    alpha_node = tet_to_node(alpha_tet, tets, pts_m.shape[0])
    E_node = 2.0 * (1.0 + NU) * alpha_node  # Young's modulus

    # World (m) -> voxel using core.transforms
    pts_mm = pts_m * 1e3
    pts_mm_t = torch.from_numpy(pts_mm)
    ct_aff_t = torch.from_numpy(ct_aff.astype(np.float32))
    vox_t = world_to_voxel_coords(pts_mm_t, ct_aff_t)
    vox = vox_t.numpy().astype(np.float32)  # (N,3)

    # Bounding box + small padding
    pad = 2.0
    vmin = np.clip(np.floor(vox.min(axis=0) - pad), 0, np.array(ct_shape) - 1)
    vmax = np.clip(np.ceil(vox.max(axis=0) + pad), 0, np.array(ct_shape) - 1)
    gi = np.arange(int(vmin[0]), int(vmax[0]) + 1)
    gj = np.arange(int(vmin[1]), int(vmax[1]) + 1)
    gk = np.arange(int(vmin[2]), int(vmax[2]) + 1)
    grid_i, grid_j, grid_k = np.meshgrid(gi, gj, gk, indexing="ij")
    grid_points = np.stack([grid_i, grid_j, grid_k], axis=-1).reshape(-1, 3).astype(np.float32)

    # Nearest-neighbor via KDTree (CPU)
    tree = cKDTree(vox)
    _, idx = tree.query(grid_points, k=1, workers=-1)
    E_vals = E_node[idx].astype(np.float32)
    E_vol = np.zeros(ct_shape, dtype=np.float32)
    E_vol[np.ix_(gi, gj, gk)] = E_vals.reshape(len(gi), len(gj), len(gk))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(E_vol, ct_aff, ct_img.header), str(out_path))
    print(f"[map] Saved CT-aligned E volume to {out_path}")


if __name__ == "__main__":
    main()
