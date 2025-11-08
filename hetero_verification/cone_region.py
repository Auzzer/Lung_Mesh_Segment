"""
Heterogeneous Cone Region Definition Module

OVERVIEW:
This module defines how to create a heterogeneous cone with two distinct material regions
for finite element simulation. It provides geometric criteria to divide a cone mesh into
background and special regions, enabling heterogeneous material property assignment.

TWO-REGION HETEROGENEOUS MATERIAL CONCEPT:
1. Background region (label=0): Majority of cone volume with standard properties
2. Special region (label=1): Localized inclusion with modified properties

GEOMETRIC REGION DEFINITION ALGORITHM:
The special region is defined as the intersection of THREE geometric conditions applied
to tetrahedron centroids:

1. Z-AXIS BAND (Horizontal Belt):
   - Creates horizontal "belt" around cone at specified height
   - Parameters: mid_center, band_width (normalized to cone height)
   - Condition: |z_normalized - mid_center| <= band_width/2

2. AZIMUTHAL WEDGE (Angular Sector):
   - Creates "pie slice" sector around cone axis  
   - Parameters: theta_center, wedge_width (in radians)
   - Condition: |angle - theta_center| <= wedge_width/2

3. RADIAL RANGE (Annular Ring):
   - Controls distance from cone central axis
   - Parameters: r_frac_min, r_frac_max (normalized to max radius)
   - Condition: r_frac_min <= r_normalized <= r_frac_max

FINAL REGION ASSIGNMENT:
- Special region (label=1): Tetrahedra satisfying ALL three geometric conditions  
- Background region (label=0): All other tetrahedra

MATERIAL PROPERTY USAGE:
The resulting labels are used by hetero_cone_fem.py to assign different material
properties to each region:
- Background region (label=0): E = E_base, rho = rho_base, nu = nu  
- Special region (label=1): E = E_base * E_special_mult, rho = rho_base * rho_special_mult, nu = nu

DEFAULT CONFIGURATION:
With default parameters, the special region forms a wedge-shaped inclusion in the
middle portion of the cone, creating a localized "stiff patch" for testing
heterogeneous material behavior and homogenization methods.
"""
from __future__ import annotations

import json
import math
import numpy as np
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable


@dataclass
class HeteroSectionParameters:
    """
    Parameters defining the heterogeneous special region within the cone.
    
    All spatial parameters are normalized to [0,1] for generality:
    - Z-coordinates: 0 = cone base, 1 = cone apex
    - Radial coordinates: 0 = cone axis, 1 = cone surface
    - Angular coordinates: 0 = +X axis direction
    
    The special region is defined as:
    Z-band: |z_normalized - mid_center| <= band_width/2
    Wedge: |angle - theta_center| <= wedge_width/2  
    Radial: r_frac_min <= r_normalized <= r_frac_max
    """
    # Z-axis band definition (horizontal belt around cone)
    mid_center: float = 0.50      # Center of special band (0=base, 1=apex)
    band_width: float = 0.20      # Height of special band (normalized)
    
    # Azimuthal wedge definition (pie slice sector)  
    theta_center: float = 0.0             # Center angle of wedge (radians)
    wedge_width: float = math.radians(60.0)  # Angular width of wedge (radians)
    
    # Radial range definition (distance from cone axis)
    r_frac_min: float = 0.20      # Minimum radius (normalized, 0=axis)
    r_frac_max: float = 1.00      # Maximum radius (normalized, 1=surface)

    def clamp(self) -> "HeteroSectionParameters":
        """Return a copy with parameters clamped to valid ranges."""
        return HeteroSectionParameters(
            mid_center=float(np.clip(self.mid_center, 0.0, 1.0)),
            band_width=float(max(self.band_width, 0.0)),
            theta_center=float(self.theta_center),  # No limits on angle
            wedge_width=float(max(self.wedge_width, 0.0)),
            r_frac_min=float(np.clip(self.r_frac_min, 0.0, 1.0)),
            r_frac_max=float(np.clip(self.r_frac_max, 0.0, 1.0)),
        )

    def as_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "HeteroSectionParameters":
        """Create instance from dictionary with validation."""
        params = cls(**{k: float(v) for k, v in data.items() if hasattr(cls, k)})
        return params.clamp()


def compute_section_labels(
    points: np.ndarray,
    tetrahedra: np.ndarray, 
    params: HeteroSectionParameters,
) -> np.ndarray:
    """
    Assign labels to tetrahedra based on their centroid locations.
    
    HETEROGENEOUS REGION DEFINITION ALGORITHM:
    This function determines which tetrahedra belong to the "special region" (label=1) 
    versus "background region" (label=0) based on three geometric criteria:
    
    1. Z-AXIS BAND (Horizontal Belt):
       - Normalized z-coordinate: z_hat = (z - z_min) / (z_max - z_min) in [0,1]
       - Condition: |z_hat - mid_center| <= band_width/2
       - Creates horizontal "belt" around cone at specified height
       
    2. AZIMUTHAL WEDGE (Angular Sector): 
       - Angular coordinate: theta = arctan2(y, x) in [-pi, pi]
       - Angular distance: Delta_theta = wrap_to_[-pi,pi](theta - theta_center)
       - Condition: |Delta_theta| <= wedge_width/2
       - Creates "pie slice" sector around cone axis
       
    3. RADIAL RANGE (Annular Region):
       - Normalized radius: r_hat = sqrt(x^2 + y^2) / r_max in [0,1]
       - Condition: r_frac_min <= r_hat <= r_frac_max
       - Creates annular ring (can exclude center or outer regions)
    
    FINAL LABELING RULE:
    - Special region (label=1): Tetrahedron centroid satisfies ALL three conditions
    - Background region (label=0): Everything else
    
    DEFAULT PARAMETERS (from cone_region_params.json):
    - mid_center=0.5, band_width=0.2    -> Middle 20% height of cone
    - theta_center=0.0, wedge_width=60 degrees -> 60 degree wedge starting from +x axis  
    - r_frac_min=0.2, r_frac_max=1.0    -> Exclude central 20% radius
    
    GEOMETRIC RESULT:
    The special region forms a wedge-shaped sector in the middle portion of the cone,
    excluding the central axis region. This creates a localized "inclusion" with
    different material properties for heterogeneous material testing.
    
    Args:
        points: Vertex coordinates (N_vertices, 3)  
        tetrahedra: Connectivity (N_tets, 4)
        params: Region definition parameters from HeteroSectionParameters
        
    Returns:
        labels: Integer array (N_tets,) with 0=background, 1=special region
    """
    params = params.clamp()
    
    # Compute tetrahedron centroids
    tet_points = points[tetrahedra]  # (N_tets, 4, 3)
    centroids = tet_points.mean(axis=1)  # (N_tets, 3)
    
    # === Z-AXIS NORMALIZATION ===
    # Convert absolute Z coordinates to normalized [0,1] range
    z_vals = centroids[:, 2]
    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max()) 
    z_range = max(z_max - z_min, 1e-9)  # Avoid division by zero
    z_normalized = (z_vals - z_min) / z_range  # 0=base, 1=apex
    
    # === RADIAL NORMALIZATION === 
    # Convert absolute radial distance to normalized [0,1] range
    radius = np.linalg.norm(centroids[:, :2], axis=1)  # Distance from Z-axis
    r_max = max(radius.max(), 1e-12)  # Maximum radius in mesh
    r_normalized = radius / r_max  # 0=axis, 1=max_radius
    
    # === ANGULAR COORDINATES ===
    # Compute azimuthal angle around Z-axis
    theta = np.arctan2(centroids[:, 1], centroids[:, 0])  # [-pi, pi]
    
    # Compute angular distance from wedge center (wrapped to [-pi, pi])
    dtheta = (theta - params.theta_center + math.pi) % (2.0 * math.pi) - math.pi
    
    # === GEOMETRIC CONDITIONS ===
    # Condition 1: Z-axis band (horizontal belt)
    z_condition = np.abs(z_normalized - params.mid_center) <= params.band_width * 0.5
    
    # Condition 2: Azimuthal wedge (angular sector)
    wedge_condition = np.abs(dtheta) <= params.wedge_width * 0.5
    
    # Condition 3: Radial range (annular region)
    if params.r_frac_min <= 0.0 and params.r_frac_max >= 1.0:
        # Full radial range - no radial constraint
        radial_condition = np.ones_like(z_condition)
    else:
        radial_condition = ((r_normalized >= params.r_frac_min) & 
                           (r_normalized <= params.r_frac_max))
    
    # === FINAL LABELING ===
    # Special region: must satisfy ALL three conditions
    # Background region: everything else
    labels = (z_condition & wedge_condition & radial_condition).astype(np.int32)
    
    return labels


# =============================================================================
# UTILITY FUNCTIONS (simplified from original overly-complex design)
# =============================================================================

def create_default_parameters() -> HeteroSectionParameters:
    """Create default heterogeneous region parameters."""
    return HeteroSectionParameters().clamp()


def save_parameters_to_file(path: Path, params: HeteroSectionParameters) -> None:
    """Save parameters to JSON file."""
    path.write_text(json.dumps(params.as_dict(), indent=2))


def load_parameters_from_file(path: Path) -> HeteroSectionParameters:
    """Load parameters from JSON file, return defaults if file missing."""
    if not path.exists():
        return create_default_parameters()
    
    with path.open("r") as f:
        data = json.load(f)
    return HeteroSectionParameters.from_dict(data)


def analyze_labeling_results(labels: Iterable[int]) -> Dict[str, float]:
    """Analyze the distribution of background vs special region elements."""
    labels_arr = np.asarray(labels, dtype=np.int32)
    n_total = int(labels_arr.size)
    n_special = int(labels_arr.sum())  # Count of label=1 elements
    n_background = n_total - n_special  # Count of label=0 elements
    
    return {
        "n_total": n_total,
        "n_background": n_background, 
        "n_special": n_special,
        "fraction_special": float(n_special / n_total) if n_total > 0 else 0.0,
        "fraction_background": float(n_background / n_total) if n_total > 0 else 0.0,
    }


# =============================================================================
# COMPATIBILITY ALIASES (for existing code that uses old function names)
# =============================================================================
default_section_parameters = create_default_parameters
save_parameters = save_parameters_to_file  
load_parameters = load_parameters_from_file
summarize_labels = analyze_labeling_results