from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from .map_alpha_to_ct import NU, load_alpha, tet_to_node


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

    # World (m) -> voxel indices
    affine_inv = np.linalg.inv(ct_aff)
    pts_mm = pts_m * 1e3
    pts_h = np.concatenate(
        [pts_mm, np.ones((pts_mm.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    vox = (affine_inv @ pts_h.T).T[:, :3].astype(np.float32)  # (N, 3)

    return ct_vol, vox, E_node.astype(np.float32), ct_aff


def interpolate_to_ct(
    mesh_npz: str | Path,
    ct_path: str | Path,
    use_log_alpha: bool = True,
    base_time: str = "T00",
    return_kpa: bool = False,
    mask_path: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate per-tet alpha / log_alpha from a mesh NPZ onto the CT voxel grid.
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

    # Nearest neighbour interpolation via KDTree (CPU)
    tree = cKDTree(vox)
    _, idx = tree.query(grid_points, k=1, workers=-1)
    E_vals = E_node[idx].astype(np.float32)

    E_vol = np.zeros(ct_shape, dtype=np.float32)
    E_vol[np.ix_(gi, gj, gk)] = E_vals.reshape(len(gi), len(gj), len(gk))

    # Optional: restrict to lung mask
    if mask_path is not None:
        mask_img = nib.load(str(mask_path))
        mask_data = mask_img.get_fdata()
        if mask_data.shape[:3] != ct_shape:
            raise ValueError(
                f"Mask shape {mask_data.shape[:3]} does not match CT shape {ct_shape}"
            )
        lung = mask_data > 0.5
        E_vol[~lung] = 0.0

    return ct_vol, E_vol, ct_aff


def interactive_slicer(
    ct_vol: np.ndarray,
    param_vol: np.ndarray,
    axis: int = 2,
    cmap: str = "bwr",
    alpha: float = 0.5,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Interactive slice viewer for CT + parameter overlay in a Jupyter notebook.

    Parameters
    ----------
    ct_vol:
        CT volume ``(I, J, K)``.
    param_vol:
        Parameter volume (e.g. E) on the same grid.
    axis:
        0=sagittal, 1=coronal, 2=axial.
    cmap:
        Matplotlib colormap name (default: blue->red 'bwr').
    alpha:
        Overlay opacity in ``[0, 1]``.
    vmin, vmax:
        Optional fixed color limits; if omitted, they are inferred from
        ``param_vol`` (ignoring zeros).
    """
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

    # Fix color range to [0.5, 20] (e.g. kPa) unless overridden
    if vmin is None:
        vmin = 0.5
    if vmax is None:
        vmax = 20.0

    max_index = ct_vol.shape[axis] - 1
    slider = widgets.IntSlider(
        value=max_index // 2,
        min=0,
        max=max_index,
        step=1,
        description="slice",
        continuous_update=False,
    )

    def _get_slice(vol: np.ndarray, idx: int) -> np.ndarray:
        if axis == 0:
            return vol[idx, :, :]
        if axis == 1:
            return vol[:, idx, :]
        return vol[:, :, idx]

    def _update(idx: int) -> None:
        ct_slice = _get_slice(ct_vol, idx)
        param_slice = _get_slice(param_vol, idx)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(ct_slice.T, cmap="gray", origin="lower")
        # Treat zeros as background: make them fully transparent
        import numpy.ma as ma

        masked = ma.masked_where(param_slice == 0.0, param_slice)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(alpha=0.0)
        im = ax.imshow(
            masked.T,
            cmap=cmap_obj,
            origin="lower",
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_axis_off()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Stiffness E(KPa)")
        plt.show()

    out = widgets.interactive_output(_update, {"idx": slider})
    display(slider, out)


def interactive_mesh_slicer(
    ct_vol: np.ndarray,
    vox_coords: np.ndarray,
    values: np.ndarray,
    axis: int = 2,
    cmap: str = "bwr",
    alpha: float = 0.8,
    vmin: float | None = None,
    vmax: float | None = None,
    thickness: float = 0.5,
    shuffle: bool = True,
) -> None:
    """
    Interactive slicer that overlays mesh nodes on CT slices.
    """
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

    values = np.asarray(values, dtype=np.float32)
    coords = np.asarray(vox_coords, dtype=np.float32)

    # Fix color range to [0.5, 20] (e.g. kPa) unless overridden
    if vmin is None:
        vmin = 0.5
    if vmax is None:
        vmax = 20.0

    max_index = ct_vol.shape[axis] - 1
    slider = widgets.IntSlider(
        value=max_index // 2,
        min=0,
        max=max_index,
        step=1,
        description="slice",
        continuous_update=False,
    )

    def _get_slice(vol: np.ndarray, idx: int) -> np.ndarray:
        if axis == 0:
            return vol[idx, :, :]
        if axis == 1:
            return vol[:, idx, :]
        return vol[:, :, idx]

    def _update(idx: int) -> None:
        ct_slice = _get_slice(ct_vol, idx)
        coord_axis = coords[:, axis]
        mask = np.abs(coord_axis - float(idx)) <= thickness

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(ct_slice.T, cmap="gray", origin="lower")

        if mask.any():
            sub = coords[mask]
            vals = values[mask]
            if shuffle:
                order = np.arange(sub.shape[0])
                np.random.shuffle(order)
                sub = sub[order]
                vals = vals[order]
            # Map voxel indices (i, j, k) to display coordinates consistently
            if axis == 2:
                # axial: plane (i, j), ct_slice = vol[:, :, idx], shown as ct_slice.T
                x = sub[:, 0]  # i
                y = sub[:, 1]  # j
            elif axis == 1:
                # coronal: plane (i, k), ct_slice = vol[:, idx, :], shown as ct_slice.T
                x = sub[:, 0]  # i
                y = sub[:, 2]  # k
            else:
                # sagittal: plane (j, k), ct_slice = vol[idx, :, :], shown as ct_slice.T
                x = sub[:, 1]  # j
                y = sub[:, 2]  # k
            sc = ax.scatter(
                x,
                y,
                c=vals,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=3,
                alpha=alpha,
            )
            ax.set_axis_off()
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Stiffness E(KPa)")
        else:
            ax.set_axis_off()

        plt.show()

    out = widgets.interactive_output(_update, {"idx": slider})
    display(slider, out)


__all__ = ["interpolate_to_ct", "interactive_slicer", "mesh_to_ct_coords", "interactive_mesh_slicer"]
