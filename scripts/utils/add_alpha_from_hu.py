"""
Augment a preprocessing npz with HU-based alpha_k prior for SMS registration.

Usage:
    python scripts/utils/add_alpha_from_hu.py --preprocessed PATH.npz --ct PATH.nii.gz
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np

ALPHA_MIN = 5e2
ALPHA_MAX = 1e4
DEFAULT_ALPHA_EMPHYSEMA = 3e3
DEFAULT_ALPHA_NORMAL = 7e3
DEFAULT_HU_LOW = -950.0
DEFAULT_HU_HIGH = -800.0

NON_PARENCHYMA_PRIOR: Dict[int, float] = {
    6: 9e3,   # airways
    7: 1e4,   # vessels
}
LOBE_LABELS = (1, 2, 3, 4, 5)


@dataclass
class AlphaPriorConfig:
    hu_low: float = DEFAULT_HU_LOW
    hu_high: float = DEFAULT_HU_HIGH
    alpha_emphysema: float = DEFAULT_ALPHA_EMPHYSEMA
    alpha_normal: float = DEFAULT_ALPHA_NORMAL


def world_to_voxel(affine_inv: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    vec = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
    ijk = affine_inv @ vec
    return ijk[:3]


def trilinear_sample(volume: np.ndarray, ijk: np.ndarray) -> float:
    i, j, k = ijk
    i0 = int(np.floor(i))
    j0 = int(np.floor(j))
    k0 = int(np.floor(k))
    di = i - i0
    dj = j - j0
    dk = k - k0

    def sample(ii: int, jj: int, kk: int) -> float:
        ii = int(np.clip(ii, 0, volume.shape[0] - 1))
        jj = int(np.clip(jj, 0, volume.shape[1] - 1))
        kk = int(np.clip(kk, 0, volume.shape[2] - 1))
        return float(volume[ii, jj, kk])

    c000 = sample(i0, j0, k0)
    c100 = sample(i0 + 1, j0, k0)
    c010 = sample(i0, j0 + 1, k0)
    c110 = sample(i0 + 1, j0 + 1, k0)
    c001 = sample(i0, j0, k0 + 1)
    c101 = sample(i0 + 1, j0, k0 + 1)
    c011 = sample(i0, j0 + 1, k0 + 1)
    c111 = sample(i0 + 1, j0 + 1, k0 + 1)

    c00 = c000 * (1.0 - di) + c100 * di
    c01 = c001 * (1.0 - di) + c101 * di
    c10 = c010 * (1.0 - di) + c110 * di
    c11 = c011 * (1.0 - di) + c111 * di
    c0 = c00 * (1.0 - dj) + c10 * dj
    c1 = c01 * (1.0 - dj) + c11 * dj
    return float(c0 * (1.0 - dk) + c1 * dk)


def emphysema_alpha_from_hu(hu: float, params: AlphaPriorConfig) -> float:
    if params.hu_high <= params.hu_low:
        raise ValueError("hu_high must be greater than hu_low")
    if hu <= params.hu_low:
        w = 1.0
    elif hu >= params.hu_high:
        w = 0.0
    else:
        w = (params.hu_high - hu) / (params.hu_high - params.hu_low)
    log_alpha = w * math.log(params.alpha_emphysema) + (1.0 - w) * math.log(
        params.alpha_normal
    )
    alpha = math.exp(log_alpha)
    return float(np.clip(alpha, ALPHA_MIN * 1.05, ALPHA_MAX * 0.95))


def add_alpha_from_hu(
    pre_npz_in: Path,
    ct_path: Path,
    pre_npz_out: Path | None,
    params: AlphaPriorConfig,
) -> Path:
    pre_npz_in = Path(pre_npz_in)
    if pre_npz_out is None:
        pre_npz_out = pre_npz_in.with_name(pre_npz_in.stem + "_with_alpha.npz")

    if not pre_npz_in.exists():
        raise FileNotFoundError(f"preprocessed npz not found: {pre_npz_in}")
    if not ct_path.exists():
        raise FileNotFoundError(f"CT volume not found: {ct_path}")

    data = np.load(pre_npz_in, allow_pickle=True)
    required = ("mesh_points", "tetrahedra", "labels")
    for key in required:
        if key not in data.files:
            raise ValueError(f"Preprocessed data missing '{key}'")

    points = data["mesh_points"].astype(np.float64)
    tets = data["tetrahedra"].astype(np.int32)
    labels = data["labels"].astype(np.int32)

    img = nib.load(str(ct_path))
    ct_hu = img.get_fdata().astype(np.float32)
    affine_inv = np.linalg.inv(img.affine)

    alpha_k = np.zeros(tets.shape[0], dtype=np.float64)

    for idx in range(tets.shape[0]):
        lbl = int(labels[idx])
        if lbl not in LOBE_LABELS:
            alpha_k[idx] = NON_PARENCHYMA_PRIOR.get(lbl, params.alpha_normal)
            continue

        verts = points[tets[idx]]
        bary = verts.mean(axis=0)
        hu_val = trilinear_sample(ct_hu, world_to_voxel(affine_inv, bary))
        alpha_k[idx] = emphysema_alpha_from_hu(hu_val, params)

    out_dict = {key: data[key] for key in data.files}
    out_dict["alpha_k"] = alpha_k.astype(np.float64)
    np.savez(pre_npz_out, **out_dict)

    print(
        "[Alpha] HU prior stats: min={:.3e}, max={:.3e}".format(
            alpha_k.min() if alpha_k.size else float("nan"),
            alpha_k.max() if alpha_k.size else float("nan"),
        )
    )
    print(f"[Alpha] Wrote HU-based alpha_k to {pre_npz_out}")
    return Path(pre_npz_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add HU-based alpha_k prior to an SMS preprocessing npz."
    )
    parser.add_argument(
        "--preprocessed",
        required=True,
        help="Existing preprocessing npz (without alpha_k).",
    )
    parser.add_argument("--ct", required=True, help="Fixed CT (NIFTI) path.")
    parser.add_argument(
        "--alpha-out",
        help="Output npz path (default: <stem>_with_alpha.npz).",
    )
    parser.add_argument(
        "--hu-low",
        type=float,
        default=DEFAULT_HU_LOW,
        help="HU threshold where emphysema prior is full strength.",
    )
    parser.add_argument(
        "--hu-high",
        type=float,
        default=DEFAULT_HU_HIGH,
        help="HU threshold where normal-lung prior is full strength.",
    )
    parser.add_argument(
        "--alpha-emphysema",
        type=float,
        default=DEFAULT_ALPHA_EMPHYSEMA,
        help="Alpha mapped to HU <= hu_low.",
    )
    parser.add_argument(
        "--alpha-normal",
        type=float,
        default=DEFAULT_ALPHA_NORMAL,
        help="Alpha mapped to HU >= hu_high.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = AlphaPriorConfig(
        hu_low=args.hu_low,
        hu_high=args.hu_high,
        alpha_emphysema=args.alpha_emphysema,
        alpha_normal=args.alpha_normal,
    )
    add_alpha_from_hu(
        Path(args.preprocessed),
        Path(args.ct),
        Path(args.alpha_out) if args.alpha_out else None,
        params,
    )


if __name__ == "__main__":
    main()
