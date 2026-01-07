"""
Generate a cone mesh with per-tet labels for heterogeneous verification.

SPECIAL REGION DEFINITION:
The cone is partitioned into background (label=0) and special (label=1) material regions.
The special region is constructed by selecting two height layers, forming a frustum of a 
smaller cone between them, and trimming by an azimuthal wedge:

1. HEIGHT LAYERS: Choose lower/upper layers via `mid_center ± band_width / 2` (normalized)
   Default: mid_center=0.50, band_width=0.40 → z=0.30 to z=0.70 (40% of cone height)
2. RADIAL RANGE: Keep centroids between `r_frac_min` and `r_frac_max` of local cone radius
   Default: r_frac_min=0.0, r_frac_max=1.0 → full radial extent (axis to surface)
3. AZIMUTHAL WEDGE: Angular sector with center `theta_center` and opening `wedge_width`
   Default: theta_center=0.0, wedge_width=120° → ±60° around +X axis

The intersection yields label=1 (special region), all other tetrahedra are label=0 (background).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import gmsh
import meshio
import numpy as np


@dataclass
class HeteroSectionParameters:
    """
    Parameters defining the heterogeneous special region within the cone.

    Spatial parameters normalized to [0, 1]:
    - Z coordinates: 0 = cone base, 1 = cone top
    - Radial coordinates: fraction of local surface radius
    - Angular coordinates: 0 = +X axis direction

    The special region is created by:
    - Selecting two layers from `mid_center ± band_width / 2`
    - Keeping centroids whose local radius fraction lies between `r_frac_min` and `r_frac_max`
    - Clipping to azimuthal wedge centered at `theta_center` with opening `wedge_width`
    """
    mid_center: float = 0.50      # Center of special band (0=base, 1=top)
    band_width: float = 0.20      # Height of special band (normalized)
    theta_center: float = 0.0     # Center angle of wedge (radians)
    wedge_width: float = math.radians(60.0)  # Angular width of wedge (radians)
    r_frac_min: float = 0.20      # Minimum radius (normalized, 0=axis)
    r_frac_max: float = 1.00      # Maximum radius (normalized, 1=surface)

    def clamp(self) -> "HeteroSectionParameters":
        """Return a copy with parameters clamped to valid ranges."""
        return HeteroSectionParameters(
            mid_center=float(np.clip(self.mid_center, 0.0, 1.0)),
            band_width=float(max(self.band_width, 0.0)),
            theta_center=float(self.theta_center),
            wedge_width=float(max(self.wedge_width, 0.0)),
            r_frac_min=float(np.clip(self.r_frac_min, 0.0, 1.0)),
            r_frac_max=float(np.clip(self.r_frac_max, 0.0, 1.0)),
        )

    def as_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def compute_section_labels(
    points: np.ndarray,
    tetrahedra: np.ndarray, 
    params: HeteroSectionParameters,
) -> np.ndarray:
    """
    Build per-tetrahedron labels using the "two layers -> small cone -> wedge" recipe.

    Workflow:
    1. Two height layers chosen from `mid_center ± band_width / 2`. Centroids
       outside this window are labeled as background.
    2. Within the window estimate local cone surface radius, compute each
       centroid's radius fraction, keep only those between `r_frac_min` and `r_frac_max`.
    3. Retained centroids clipped to azimuthal wedge defined by `theta_center` and `wedge_width`.

    Args:
        points: Vertex coordinates (N_vertices, 3)
        tetrahedra: Connectivity (N_tets, 4)
        params: Region definition parameters

    Returns:
        labels: Integer array (N_tets,) with 0 = background and 1 = special region
    """
    params = params.clamp()

    tet_points = points[tetrahedra]  # (N_tets, 4, 3)
    centroids = tet_points.mean(axis=1)  # (N_tets, 3)

    # Normalize centroid heights to [0, 1]
    z_vals = centroids[:, 2]
    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max())
    z_range = max(z_max - z_min, 1e-9)
    z_normalized = (z_vals - z_min) / z_range

    # Determine the two layers (lower/upper limits)
    half_band = params.band_width * 0.5
    z_lower = max(params.mid_center - half_band, 0.0)
    z_upper = min(params.mid_center + half_band, 1.0)
    if z_upper < z_lower:
        z_lower = z_upper = params.mid_center

    if math.isclose(z_upper, z_lower, rel_tol=0.0, abs_tol=1e-6):
        height_condition = np.abs(z_normalized - z_lower) <= 1e-6
    else:
        height_condition = (z_normalized >= z_lower) & (z_normalized <= z_upper)

    # Estimate cone surface radius as function of height for local radial fractions
    def _estimate_surface_profile(
        pts: np.ndarray, z0: float, zlen: float, bins: int = 80
    ) -> Tuple[np.ndarray, np.ndarray]:
        z_norm_pts = np.clip((pts[:, 2] - z0) / zlen, 0.0, 1.0)
        radius_pts = np.linalg.norm(pts[:, :2], axis=1)
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_indices = np.clip(np.digitize(z_norm_pts, bin_edges) - 1, 0, bins - 1)

        surface_profile = np.zeros(bins, dtype=np.float64)
        np.maximum.at(surface_profile, bin_indices, radius_pts)

        counts = np.bincount(bin_indices, minlength=bins)
        valid = counts > 0
        if not np.all(valid):
            valid_idx = np.where(valid)[0]
            if valid_idx.size == 0:
                raise ValueError("Unable to estimate cone surface radius from mesh points")
            surface_profile = np.interp(
                bin_centers, bin_centers[valid_idx], surface_profile[valid_idx]
            )
        return bin_centers, surface_profile

    bins_z, surface_radius = _estimate_surface_profile(points, z_min, z_range)
    centroid_surface_radius = np.interp(z_normalized, bins_z, surface_radius)

    radius = np.linalg.norm(centroids[:, :2], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        local_radius_fraction = np.divide(
            radius,
            centroid_surface_radius,
            out=np.zeros_like(radius),
            where=centroid_surface_radius > 1e-12,
        )
    local_radius_fraction = np.clip(local_radius_fraction, 0.0, 1.0)

    lower_frac = min(params.r_frac_min, params.r_frac_max)
    upper_frac = max(params.r_frac_min, params.r_frac_max)
    if lower_frac <= 0.0 and upper_frac >= 1.0:
        radial_condition = np.ones_like(height_condition, dtype=bool)
    else:
        radial_condition = (local_radius_fraction >= lower_frac) & (
            local_radius_fraction <= upper_frac
        )

    # Azimuthal wedge selection
    theta = np.arctan2(centroids[:, 1], centroids[:, 0])
    dtheta = (theta - params.theta_center + math.pi) % (2.0 * math.pi) - math.pi
    wedge_condition = np.abs(dtheta) <= params.wedge_width * 0.5

    labels = (height_condition & radial_condition & wedge_condition).astype(np.int32)

    return labels


def summarize_labels(labels: np.ndarray) -> Dict[str, float]:
    """Analyze the distribution of background vs special region elements."""
    n_total = int(labels.size)
    n_special = int(labels.sum())  # Count of label=1 elements
    n_background = n_total - n_special  # Count of label=0 elements
    
    return {
        "n_total": n_total,
        "n_background": n_background, 
        "n_special": n_special,
        "fraction_special": float(n_special / n_total) if n_total > 0 else 0.0,
        "fraction_background": float(n_background / n_total) if n_total > 0 else 0.0,
    }


def build_cone_mesh(length: float, radius_base: float, radius_top: float, hmax: float) -> meshio.Mesh:
    gmsh.initialize()
    gmsh.model.add("cone_hetero")
    volume = gmsh.model.occ.addCone(0.0, 0.0, 0.0, 0.0, 0.0, length, radius_base, radius_top)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", hmax)
    gmsh.model.mesh.generate(3)

    tmp_path = Path("cone_tmp.msh")
    gmsh.write(str(tmp_path))
    gmsh.finalize()

    msh = meshio.read(tmp_path)
    tmp_path.unlink(missing_ok=True)

    tets = msh.get_cells_type("tetra")
    if tets is None or tets.size == 0:
        raise RuntimeError("Generated mesh does not contain tetrahedra")

    return meshio.Mesh(points=msh.points, cells=[("tetra", tets)])


def write_cone_with_labels(
    output_dir: Path,
    params: HeteroSectionParameters,
    mesh: meshio.Mesh,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    xdmf_dir = output_dir / "xdmf_visualization"
    xdmf_dir.mkdir(parents=True, exist_ok=True)
    xdmf_path = xdmf_dir / "cone.xdmf"
    tets = mesh.get_cells_type("tetra")
    labels = compute_section_labels(mesh.points, tets, params)

    meshio.write(
        xdmf_path,
        meshio.Mesh(points=mesh.points, cells=[("tetra", tets)], cell_data={"labels": [labels]}),
        data_format="XML",
    )

    stats = summarize_labels(labels)
    print(
        f"Saved {xdmf_path} with {stats['n_special']} special cells "
        f"({stats['fraction_special']:.3%} of {stats['n_total']})"
    )
    
    return xdmf_path



def main():
    """
    Generate cone mesh with heterogeneous region labels.
    
    SPECIAL REGION PARAMETERS (hardcoded, documented here for reference):
    These parameters define how the special material region (label=1) is created:
    
    CONE GEOMETRY:
      - length = 0.6 m (cone height)
      - radius_base = 0.20 m (base radius)
      - radius_top = 0.10 m (top radius)
      - hmax = 0.03 m (max mesh element size)
    
    SPECIAL REGION DEFINITION (see module docstring for details):
      - mid_center = 0.50 (center of height band, normalized 0=base to 1=top)
      - band_width = 0.40 (height range of special region, normalized)
      - theta_center = 0.0 rad (azimuthal angle of wedge center, 0=+X axis)
      - wedge_width = 2.0944 rad (120°, angular opening of wedge)
      - r_frac_min = 0.0 (min radial fraction of local cone radius, 0=axis)
      - r_frac_max = 1.00 (max radial fraction, 1.0=cone surface)
    
    The special region is the intersection of:
      1. Height band from z=0.30 to z=0.70 (normalized)
      2. Full radial extent from 0% to 100% of local cone radius
      3. Azimuthal wedge ±60° around +X axis (total 120° opening)
    """
    length = 0.6
    radius_base = 0.20
    radius_top = 0.10
    hmax = 0.03
    mid_center = 0.50
    band_width = 0.40
    theta_center = 0.0
    wedge_width = np.deg2rad(120.0)
    radial_min = 0.0
    radial_max = 1.00
    output_dir = Path(__file__).resolve().parent
    
    params = HeteroSectionParameters(
        mid_center=mid_center,
        band_width=band_width,
        theta_center=theta_center,
        wedge_width=wedge_width,
        r_frac_min=radial_min,
        r_frac_max=radial_max,
    ).clamp()

    print("Generating cone mesh with heterogeneous region parameters:")
    for key, value in params.as_dict().items():
        if key.startswith("theta") or key.startswith("wedge"):
            print(f"  {key}: {value:.4f} rad ({np.rad2deg(value):.1f}°)")
        else:
            print(f"  {key}: {value:.4f}")

    mesh = build_cone_mesh(length, radius_base, radius_top, hmax)
    write_cone_with_labels(output_dir, params, mesh)


if __name__ == "__main__":
    main()
