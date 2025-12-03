"""
Map stiffness (or K-derived scalars) back onto an image grid for visualization.

Workflows:
1) Mesh field mapping: read a mesh (e.g., alpha_final.xdmf from sms_reg_alpha_lbfgs.py)
   and interpolate a scalar field stored on points or cells onto the CT grid.
2) Nodal K reassembly: given an SMS preprocessing npz (deformation_processor_v2.py),
   rebuild 3x3 nodal K blocks using the same row outer-product recipe as
   `scripts/sms_reg_u_lbfgs.py`, reduce each block to a scalar metric, and interpolate.

Coordinate conversions use `project.core.transforms.world_to_voxel_coords` from
`lung_project_git`.

Dependencies: numpy, scipy, meshio, SimpleITK, lung_project_git (local path).

Optional: load a lung mask (either explicit path, or inferred from a
TotalSegment directory that mirrors the CT filenames) and zero values outside
the mask after interpolation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Literal

import meshio
import numpy as np
import SimpleITK as sitk
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def _add_lung_project_git_to_path(repo_root: Path) -> None:
    candidate = repo_root / "lung_project_git"
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def build_affine_from_image(image: sitk.Image) -> np.ndarray:
    """Compose a voxel-to-world affine from a SimpleITK image header."""
    direction = np.array(image.GetDirection(), dtype=float).reshape(3, 3)
    spacing = np.array(image.GetSpacing(), dtype=float)
    origin = np.array(image.GetOrigin(), dtype=float)

    affine = np.eye(4, dtype=float)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return affine


def infer_mask_path(
    image_path: Path,
    mask_root: Path | None = None,
    mask_name: str = "lung_combined_mask.nii.gz",
) -> Path:
    """
    Infer a TotalSegment-style mask path from a CT path.

    Example:
        image_path = .../Case5Pack/NIFTI/case5_T40.nii.gz
        mask_root  = .../Case5Pack/TotalSegment (default)
        returns    = .../Case5Pack/TotalSegment/case5_T40/lung_combined_mask.nii.gz
    """
    if mask_root is None:
        # use the CaseXPack directory as the anchor
        mask_root = image_path.parent.parent / "TotalSegment"

    name = image_path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]  # strip .nii.gz
    elif name.endswith(".nii"):
        name = name[:-4]
    candidate = mask_root / name / mask_name
    return candidate


def load_mesh_field(
    mesh_path: Path,
    field: str,
    location: Literal["auto", "point", "cell"] = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load coordinates and a scalar field from a mesh file.

    Returns:
        coords: (N, 3) world coordinates
        values: (N,) scalar values
    """
    mesh = meshio.read(mesh_path)
    coords = np.asarray(mesh.points, dtype=float)

    if location in {"auto", "point"} and field in mesh.point_data:
        values = np.asarray(mesh.point_data[field], dtype=float)
        values = values.reshape(values.shape[0], -1)
        if values.shape[1] != 1:
            raise ValueError(f"Expected scalar field; got shape {values.shape}")
        return coords, values[:, 0]

    if location in {"auto", "cell"} and field in mesh.cell_data:
        centers, vals = [], []
        for cell_block, data in zip(mesh.cells, mesh.cell_data[field]):
            cell_points = coords[np.asarray(cell_block.data, dtype=int)]
            centers.append(cell_points.mean(axis=1))
            vals.append(np.asarray(data, dtype=float))

        if not centers:
            raise ValueError(f"No cells found for field '{field}' in {mesh_path}")

        center_arr = np.concatenate(centers, axis=0)
        val_arr = np.concatenate(vals, axis=0).reshape(-1, 1)
        if val_arr.shape[1] != 1:
            raise ValueError(f"Expected scalar field; got shape {val_arr.shape}")
        return center_arr, val_arr[:, 0]

    raise KeyError(
        f"Unable to locate field '{field}' as point or cell data in {mesh_path}"
    )


def build_interpolator(
    coords: np.ndarray,
    values: np.ndarray,
    method: Literal["linear", "nearest", "hybrid"],
    fill_value: float,
) -> Callable[[np.ndarray], np.ndarray]:
    values = np.asarray(values, dtype=float)
    coords = np.asarray(coords, dtype=float)

    if method == "linear":
        return LinearNDInterpolator(coords, values, fill_value=fill_value)

    if method == "nearest":
        nearest = NearestNDInterpolator(coords, values)
        return lambda pts: nearest(pts)

    if method == "hybrid":
        linear = LinearNDInterpolator(coords, values, fill_value=np.nan)
        nearest = NearestNDInterpolator(coords, values)

        def interpolate(pts: np.ndarray) -> np.ndarray:
            vals = np.asarray(linear(pts))
            mask = np.isnan(vals)
            if mask.any():
                vals = vals.copy()
                vals[mask] = nearest(pts[mask])
            nan_mask = np.isnan(vals)
            if nan_mask.any():
                vals[nan_mask] = fill_value
            return vals

        return interpolate


def _ensure_keys(npz: np.lib.npyio.NpzFile, keys: list[str], source: Path) -> None:
    missing = [k for k in keys if k not in npz.files]
    if missing:
        raise ValueError(f"{source} is missing required keys: {', '.join(missing)}")


def _accumulate_node_block(
    node_blocks: np.ndarray,
    tet_nodes: np.ndarray,
    block_rows: np.ndarray,
    coeff: float,
) -> None:
    if coeff == 0.0:
        return
    for local_idx, node_idx in enumerate(tet_nodes):
        vec = block_rows[local_idx]
        node_blocks[node_idx] += coeff * np.outer(vec, vec)


def reassemble_nodal_k(
    preprocessed_path: Path,
    stiffness_key: str = "stiffness",
    torsion_key: str = "torsion_stiffness",
    kappa_key: str = "volume_kappa",
    volume_key: str = "volume",
    r_axis_key: str = "r_axis",
    r_shear_key: str = "r_shear",
    r_vol_key: str = "r_vol",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reassemble nodal 3x3 stiffness blocks from SMS preprocessing outputs.

    Returns:
        coords_world: (N, 3) node coordinates in world units.
        node_blocks: (N, 3, 3) symmetric stiffness blocks per node.
    """
    data = np.load(preprocessed_path, allow_pickle=True)
    _ensure_keys(
        data,
        [
            "mesh_points",
            "tetrahedra",
            stiffness_key,
            torsion_key,
            kappa_key,
            volume_key,
            r_axis_key,
            r_shear_key,
            r_vol_key,
        ],
        preprocessed_path,
    )

    coords_world = np.asarray(data["mesh_points"], dtype=float)
    tets = np.asarray(data["tetrahedra"], dtype=np.int64)
    vol = np.asarray(data[volume_key], dtype=float)
    stiffness = np.asarray(data[stiffness_key], dtype=float)
    torsion = np.asarray(data[torsion_key], dtype=float)
    kappa = np.asarray(data[kappa_key], dtype=float)
    r_axis = np.asarray(data[r_axis_key], dtype=float)
    r_shear = np.asarray(data[r_shear_key], dtype=float)
    r_vol = np.asarray(data[r_vol_key], dtype=float)

    if stiffness.shape[1] != 3 or torsion.shape[1] != 3:
        raise ValueError(
            f"Expected 3 axial and 3 shear stiffness values per tet; "
            f"got {stiffness.shape} and {torsion.shape}"
        )
    if r_axis.shape[1] != 3 or r_shear.shape[1] != 3 or r_axis.shape[2] != 12:
        raise ValueError(
            f"Unexpected r_axis/r_shear shape: {r_axis.shape}, {r_shear.shape}"
        )
    if r_vol.shape[1] != 12:
        raise ValueError(f"Unexpected r_vol shape: {r_vol.shape}")

    node_blocks = np.zeros((coords_world.shape[0], 3, 3), dtype=float)

    for k in range(tets.shape[0]):
        tet_nodes = tets[k]
        vol_k = float(vol[k])
        if vol_k <= 0.0:
            continue

        for axis_idx in range(3):
            coeff = 4.0 * stiffness[k, axis_idx] * vol_k
            rows = r_axis[k, axis_idx].reshape(4, 3)
            _accumulate_node_block(node_blocks, tet_nodes, rows, coeff)

        for shear_idx in range(3):
            coeff = 4.0 * torsion[k, shear_idx] * vol_k
            rows = r_shear[k, shear_idx].reshape(4, 3)
            _accumulate_node_block(node_blocks, tet_nodes, rows, coeff)

        coeff_vol = float(kappa[k] * vol_k)
        rows_vol = r_vol[k].reshape(4, 3)
        _accumulate_node_block(node_blocks, tet_nodes, rows_vol, coeff_vol)

    return coords_world, node_blocks


def summarize_node_blocks(
    node_blocks: np.ndarray,
    metric: Literal[
        "frobenius", "trace", "max_eig", "min_eig", "det", "xx", "yy", "zz"
    ] = "frobenius",
) -> np.ndarray:
    """Reduce each 3x3 block to a scalar for interpolation."""
    if metric == "frobenius":
        return np.linalg.norm(node_blocks, ord="fro", axis=(1, 2))
    if metric == "trace":
        return np.trace(node_blocks, axis1=1, axis2=2)
    if metric == "max_eig":
        eigvals = np.linalg.eigvalsh(node_blocks)
        return eigvals.max(axis=1)
    if metric == "min_eig":
        eigvals = np.linalg.eigvalsh(node_blocks)
        return eigvals.min(axis=1)
    if metric == "det":
        return np.linalg.det(node_blocks)
    if metric in {"xx", "yy", "zz"}:
        idx = {"xx": 0, "yy": 1, "zz": 2}[metric]
        return node_blocks[:, idx, idx]
    raise ValueError(f"Unsupported node metric: {metric}")


def _resample_mask_to_image(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Nearest-neighbor resample to align an input mask to the reference grid."""
    if (
        mask.GetSize() == reference.GetSize()
        and mask.GetSpacing() == reference.GetSpacing()
        and mask.GetOrigin() == reference.GetOrigin()
        and mask.GetDirection() == reference.GetDirection()
    ):
        return mask

    return sitk.Resample(
        mask,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        reference.GetPixelID(),
    )


def rasterize_to_image(
    image: sitk.Image,
    voxel_coords: np.ndarray,
    values: np.ndarray,
    method: Literal["linear", "nearest", "hybrid"] = "hybrid",
    fill_value: float = np.nan,
) -> sitk.Image:
    """
    Interpolate sparse samples (in voxel space) onto the full image grid.

    Args:
        image: reference SimpleITK image providing size/spacing/origin/direction
        voxel_coords: (N, 3) coordinates expressed in voxel units
        values: (N,) scalar values at voxel_coords
    """
    interpolator = build_interpolator(
        coords=voxel_coords, values=values, method=method, fill_value=fill_value
    )

    size_x, size_y, size_z = image.GetSize()
    output = np.full((size_z, size_y, size_x), fill_value, dtype=float)

    x_idx = np.arange(size_x, dtype=float)
    y_idx = np.arange(size_y, dtype=float)

    for k in range(size_z):
        xx, yy = np.meshgrid(x_idx, y_idx, indexing="ij")
        slice_points = np.column_stack(
            [xx.ravel(), yy.ravel(), np.full(xx.size, float(k))]
        )
        slice_vals = interpolator(slice_points)
        slice_vals = np.asarray(slice_vals, dtype=float).reshape(xx.shape)
        output[k] = slice_vals.T

    mapped = sitk.GetImageFromArray(output)
    mapped.CopyInformation(image)
    return mapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate stiffness (mesh field or reassembled nodal K) onto an image grid."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--mesh",
        type=Path,
        help="Mesh with a scalar field (e.g., alpha_final.xdmf from sms_reg_alpha_lbfgs.py).",
    )
    source_group.add_argument(
        "--preprocessed",
        type=Path,
        help="SMS preprocessing npz containing stiffness fields and r_* rows (for K reassembly).",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="alpha",
        help="Name of the mesh field to map (point or cell data).",
    )
    parser.add_argument(
        "--location",
        choices=["auto", "point", "cell"],
        default="auto",
        help="Where to read the field from when using --mesh.",
    )
    parser.add_argument(
        "--stiffness-key",
        type=str,
        default="stiffness",
        help="npz key for axial stiffness (3 values per tet) when reassembling.",
    )
    parser.add_argument(
        "--torsion-key",
        type=str,
        default="torsion_stiffness",
        help="npz key for shear stiffness (3 values per tet) when reassembling.",
    )
    parser.add_argument(
        "--kappa-key",
        type=str,
        default="volume_kappa",
        help="npz key for volumetric stiffness when reassembling.",
    )
    parser.add_argument(
        "--volume-key",
        type=str,
        default="volume",
        help="npz key for tetra volumes when reassembling.",
    )
    parser.add_argument(
        "--r-axis-key",
        type=str,
        default="r_axis",
        help="npz key for axial row vectors (shape Mx3x12) when reassembling.",
    )
    parser.add_argument(
        "--r-shear-key",
        type=str,
        default="r_shear",
        help="npz key for shear row vectors (shape Mx3x12) when reassembling.",
    )
    parser.add_argument(
        "--r-vol-key",
        type=str,
        default="r_vol",
        help="npz key for volumetric row vectors (shape Mx12) when reassembling.",
    )
    parser.add_argument(
        "--node-metric",
        choices=["frobenius", "trace", "max_eig", "min_eig", "det", "xx", "yy", "zz"],
        default="frobenius",
        help="Scalar summary for each nodal 3x3 K block (reassembly mode).",
    )
    parser.add_argument(
        "--save-node-blocks",
        type=Path,
        default=None,
        help="Optional path to save assembled nodal 3x3 K blocks (.npy).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Reference image (.mha/.nii/etc) defining the target grid.",
    )
    parser.add_argument(
        "--method",
        choices=["linear", "nearest", "hybrid"],
        default="hybrid",
        help="Interpolation method; hybrid falls back to nearest outside hull.",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=np.nan,
        help="Value to use where interpolation is undefined.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional mask image path; values outside mask set to fill-value.",
    )
    parser.add_argument(
        "--infer-mask",
        action="store_true",
        help=(
            "Infer a mask from the image path (expects TotalSegment/<stem>/"
            "lung_combined_mask.nii.gz alongside the CT)."
        ),
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=None,
        help="Override the base directory for mask inference (default: sibling TotalSegment).",
    )
    parser.add_argument(
        "--mask-name",
        type=str,
        default="lung_combined_mask.nii.gz",
        help="Mask filename used during inference (default: lung_combined_mask.nii.gz).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path; defaults to <mesh stem>_<field>_map.nii.gz (mesh mode) "
            "or <preprocessed stem>_node_k_<metric>.nii.gz (reassembly mode)."
        ),
    )
    parser.add_argument(
        "--save-npy",
        type=Path,
        default=None,
        help="Optional path to also save the mapped volume as a .npy array.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    _add_lung_project_git_to_path(repo_root)

    from project.core import transforms

    image = sitk.ReadImage(str(args.image))
    affine = build_affine_from_image(image)

    mask_image: sitk.Image | None = None
    if args.mask is not None:
        mask_image = sitk.ReadImage(str(args.mask))
    elif args.infer_mask:
        inferred_mask = infer_mask_path(
            image_path=args.image, mask_root=args.mask_root, mask_name=args.mask_name
        )
        print(f"Inferred mask path: {inferred_mask}")
        mask_image = sitk.ReadImage(str(inferred_mask))

    if args.mesh is not None:
        coords_world, values = load_mesh_field(
            mesh_path=args.mesh, field=args.field, location=args.location
        )
    else:
        coords_world, node_blocks = reassemble_nodal_k(
            preprocessed_path=args.preprocessed,
            stiffness_key=args.stiffness_key,
            torsion_key=args.torsion_key,
            kappa_key=args.kappa_key,
            volume_key=args.volume_key,
            r_axis_key=args.r_axis_key,
            r_shear_key=args.r_shear_key,
            r_vol_key=args.r_vol_key,
        )
        values = summarize_node_blocks(node_blocks, metric=args.node_metric)
        if args.save_node_blocks is not None:
            np.save(args.save_node_blocks, node_blocks)
            print(f"Saved nodal 3x3 K blocks -> {args.save_node_blocks}")
    voxel_coords = transforms.world_to_voxel_coords(coords_world, affine)

    size = np.array(image.GetSize(), dtype=float)
    in_bounds = (
        (voxel_coords >= 0.0) & (voxel_coords <= (size - 1.0))
    ).all(axis=1)
    pct_in_bounds = 100.0 * in_bounds.sum() / len(voxel_coords)
    print(
        f"{in_bounds.sum()}/{len(voxel_coords)} samples "
        f"({pct_in_bounds:.1f}%) fall inside the image grid"
    )

    mapped = rasterize_to_image(
        image=image,
        voxel_coords=voxel_coords,
        values=values,
        method=args.method,
        fill_value=args.fill_value,
    )

    if mask_image is not None:
        mask_resampled = _resample_mask_to_image(mask_image, image)
        mask_arr = sitk.GetArrayFromImage(mask_resampled) > 0
        mapped_arr = sitk.GetArrayFromImage(mapped)
        mapped_arr[~mask_arr] = args.fill_value
        mapped = sitk.GetImageFromArray(mapped_arr)
        mapped.CopyInformation(image)
        print("Applied mask to interpolated volume.")

    if args.output is not None:
        out_path = args.output
    elif args.mesh is not None:
        out_path = args.mesh.with_name(f"{args.mesh.stem}_{args.field}_map.nii.gz")
    else:
        out_path = args.preprocessed.with_name(
            f"{args.preprocessed.stem}_node_k_{args.node_metric}.nii.gz"
        )
    sitk.WriteImage(mapped, str(out_path))
    print(f"Wrote interpolated volume -> {out_path}")

    if args.save_npy is not None:
        np.save(args.save_npy, sitk.GetArrayFromImage(mapped))
        print(f"Saved numpy volume -> {args.save_npy}")


if __name__ == "__main__":
    main()
