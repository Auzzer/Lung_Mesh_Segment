"""Generate a cone mesh with per-tet labels for heterogeneous verification."""
from __future__ import annotations

from pathlib import Path
import sys
import json

import gmsh
import meshio
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from hetero_verification.cone_region import (  # type: ignore
        AxisTrapezoidParameters,
        HeteroSectionParameters,
        compute_section_labels,
        compute_section_labels_axis_trapezoid,
        save_parameters,
        summarize_labels,
    )
else:
    from .cone_region import (
        AxisTrapezoidParameters,
        HeteroSectionParameters,
        compute_section_labels,
        compute_section_labels_axis_trapezoid,
        save_parameters,
        summarize_labels,
    )


def build_cone_mesh(length: float, radius_base: float, radius_apex: float, hmax: float) -> meshio.Mesh:
    gmsh.initialize()
    gmsh.model.add("cone_hetero")
    volume = gmsh.model.occ.addCone(0.0, 0.0, 0.0, 0.0, 0.0, length, radius_base, radius_apex)
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


def _axis_params_to_json(params: AxisTrapezoidParameters) -> dict:
    return {
        "type": "axis_trapezoid",
        "base_point": params.base_point.tolist(),
        "apex_point": params.apex_point.tolist(),
        "s_min": params.s_min,
        "s_max": params.s_max,
        "wedge_center": params.wedge_center,
        "wedge_width": params.wedge_width,
        "r_min": params.r_min,
        "r_max": params.r_max,
        "base_radius": params.base_radius,
        "apex_radius": params.apex_radius,
    }


def write_cone_with_labels(
    output_dir: Path,
    mesh: meshio.Mesh,
    *,
    region_mode: str,
    region_params,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    xdmf_path = output_dir / "cone.xdmf"
    tets = mesh.get_cells_type("tetra")

    if region_mode == "axis_trapezoid":
        labels = compute_section_labels_axis_trapezoid(mesh.points, tets, region_params)
        params_path = output_dir / "cone_axis_trapezoid_params.json"
        params_path.write_text(json.dumps(_axis_params_to_json(region_params), indent=2))
    elif region_mode == "legacy":
        labels = compute_section_labels(mesh.points, tets, region_params)
        params_path = output_dir / "cone_region_params.json"
        save_parameters(params_path, region_params)
    else:
        raise ValueError(f"Unsupported region mode: {region_mode}")

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
    print(f"Stored region parameters in {params_path}")
    return xdmf_path



def main() -> None:
    # Base cone geometry (matches hetero verification case)
    length = 0.6
    radius_base = 0.20
    radius_apex = 0.10
    hmax = 0.03
    output_dir = Path(__file__).resolve().parent

    # Toggle between legacy "belt + wedge" parameters and the new axial trapezoid region
    region_mode = "axis_trapezoid"  # set to "legacy" to reproduce the original behaviour

    if region_mode == "axis_trapezoid":
        wedge_width_deg = 60.0
        axis_params = AxisTrapezoidParameters(
            base_point=np.array([0.0, 0.0, 0.0], dtype=float),
            apex_point=np.array([0.0, 0.0, length], dtype=float),
            s_min=0.45,
            s_max=0.65,
            wedge_center=0.0,
            wedge_width=np.deg2rad(wedge_width_deg),
            r_min=0.0,
            r_max=1.0,
            base_radius=radius_base,
            apex_radius=radius_apex,
        )

        print("Generating cone mesh with axial trapezoid parameters:")
        axis_dict = _axis_params_to_json(axis_params)
        for key, value in axis_dict.items():
            if key in {"base_point", "apex_point"}:
                print(f"  {key}: {value}")
            elif key == "type":
                continue
            elif key.startswith("wedge"):
                print(f"  {key}: {value:.4f} rad")
            else:
                print(f"  {key}: {value:.4f}")

        region_params = axis_params

    else:
        params = HeteroSectionParameters(
            mid_center=0.50,
            band_width=0.20,
            theta_center=0.0,
            wedge_width=np.deg2rad(60.0),
            r_frac_min=0.20,
            r_frac_max=1.00,
        ).clamp()

        print("Generating cone mesh with legacy belt/wedge parameters:")
        for key, value in params.as_dict().items():
            if key.startswith("theta") or key.startswith("wedge"):
                print(f"  {key}: {value:.4f} rad")
            else:
                print(f"  {key}: {value:.4f}")

        region_params = params

    mesh = build_cone_mesh(length, radius_base, radius_apex, hmax)
    write_cone_with_labels(
        output_dir,
        mesh,
        region_mode=region_mode,
        region_params=region_params,
    )


if __name__ == "__main__":
    main()
