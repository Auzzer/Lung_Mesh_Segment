#!/usr/bin/env python
"""
Helper script to generate SMS preprocessing NPZ files from Emory mesh + displacement.
Runs in a separate process to avoid Taichi re-initialization conflicts.
"""

import argparse
import sys
from pathlib import Path

import taichi as ti

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from toy_verification.deformation_processor_v2 import (
    DeformationProcessor,
    load_mesh_with_displacement,
)


def main():
    parser = argparse.ArgumentParser(description="Generate SMS preprocessing NPZ from mesh + displacement")
    parser.add_argument("--mesh", required=True, help="Path to pygalmesh .xdmf file")
    parser.add_argument("--displacement", required=True, help="Path to CorrField displacement .nii.gz")
    parser.add_argument("--output", required=True, help="Output NPZ file path")
    parser.add_argument("--subject", help="Subject identifier for metadata validation")
    parser.add_argument("--fixed-state", dest="fixed_state", help="Fixed respiratory state")
    parser.add_argument("--moving-state", dest="moving_state", help="Moving respiratory state")
    parser.add_argument("--mask-name", dest="mask_name", help="Mask/segmentation name")
    parser.add_argument("--mesh-tag", dest="mesh_tag", help="Mesh tag suffix")
    parser.add_argument("--variant", help="Dataset variant identifier")
    args = parser.parse_args()

    mesh_path = Path(args.mesh).resolve()
    disp_path = Path(args.displacement).resolve()
    output_path = Path(args.output).resolve()

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    if not disp_path.exists():
        raise FileNotFoundError(f"Displacement file not found: {disp_path}")

    print(f"[sms_precompute] Loading mesh: {mesh_path}")
    print(f"[sms_precompute] Loading displacement: {disp_path}")
    try:
        pts, tets, lbls, displacement_vectors = load_mesh_with_displacement(str(mesh_path), str(disp_path))

        # Convert mesh to metres (pygalmesh outputs millimetres)
        pts_m = pts.astype(float) * 1e-3

        processor = DeformationProcessor(pts_m, tets, lbls)
        processor.set_metadata(
            subject=args.subject,
            fixed_state=args.fixed_state,
            moving_state=args.moving_state,
            mask_name=args.mask_name,
            mesh_tag=args.mesh_tag,
            variant=args.variant,
        )
        processor.set_displacement_field(displacement_vectors)

        # DeformationProcessor expects path without extension
        output_base = output_path
        if output_base.suffix == ".npz":
            output_base = output_base.with_suffix("")
        processor.save_results(str(output_base))
        print(f"[sms_precompute] Saved preprocessing results to {output_base}.npz")
    finally:
        # Release Taichi CUDA context between subprocess invocations
        ti.reset()


if __name__ == "__main__":
    main()
