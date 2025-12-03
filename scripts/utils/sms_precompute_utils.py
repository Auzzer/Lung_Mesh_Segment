#!/usr/bin/env python
"""
Utility helpers for SMS preprocessing artifacts.
Shared between the main optimizer and batch preprocessing scripts so we keep
metadata handling and dataset resolution consistent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Ensure the repository and project roots are on sys.path for dataset imports.
_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CURRENT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PROJECT_ROOT = _REPO_ROOT / "lung_project_git"
if _PROJECT_ROOT.is_dir() and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lung_project_git.project.datasets.emory4dct import Emory4DCTDataset


def normalize_mesh_tag(mask_name: str, mesh_tag: str) -> str:
    """Strip the mask-name prefix from a mesh tag if present."""
    prefix = f"{mask_name}_"
    if mesh_tag.startswith(prefix):
        return mesh_tag[len(prefix):]
    return mesh_tag


def get_emory_example_paths(
    data_root: str,
    subject: str,
    variant: str,
    mask_name: str,
    mesh_tag: str,
    fixed_state: str,
    moving_state: str,
) -> dict:
    """Resolve filesystem paths for a single Emory 4DCT example."""
    ds = Emory4DCTDataset(data_root)
    ex_iter = ds.examples(
        subjects=[subject],
        variant=variant,
        state_pairs=[(fixed_state, moving_state)],
        mask_name=mask_name,
        mesh_tag=mesh_tag,
        source_variant='Images',
        ref_state=fixed_state,
    )
    try:
        example = next(ex_iter)
    except StopIteration as exc:
        raise RuntimeError(
            f"Failed to find Emory example for {subject} {fixed_state}->{moving_state} "
            f"(variant={variant}, mask={mask_name}, mesh={mesh_tag})"
        ) from exc
    return example.paths


def run_sms_preprocessor(
    mesh_path: Path,
    displacement_path: Path,
    output_npz: Path,
    metadata: Optional[dict[str, str]] = None,
    log_fn: Callable[[str], None] = print,
) -> None:
    """Invoke the helper script that builds SMS NPZ blobs."""
    helper_script = _CURRENT_DIR / "sms_precompute.py"
    if not helper_script.exists():
        raise FileNotFoundError(f"Helper script not found: {helper_script}")
    cmd = [
        sys.executable,
        str(helper_script),
        "--mesh",
        str(mesh_path),
        "--displacement",
        str(displacement_path),
        "--output",
        str(output_npz),
    ]
    if metadata:
        for key, value in metadata.items():
            if value is None:
                continue
            flag = f"--{key.replace('_', '-')}"
            cmd.extend([flag, str(value)])
    log_fn(f"Running deformation preprocessing via helper: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_mesh_info(npz_path: Path) -> dict[str, Any]:
    """Load the mesh_info dict from a preprocessing NPZ, if available."""
    with np.load(npz_path, allow_pickle=True) as data:
        if 'mesh_info' not in data.files:
            return {}
        mesh_info_arr = data['mesh_info']
        if isinstance(mesh_info_arr, dict):
            return dict(mesh_info_arr)
        try:
            mesh_info = mesh_info_arr.item()
            if isinstance(mesh_info, dict):
                return dict(mesh_info)
        except Exception:
            pass
    return {}


def infer_states_from_filename(npz_path: Path) -> dict[str, str]:
    """Fallback parsing of subject/states from the NPZ filename."""
    stem = npz_path.stem
    if "_to_" not in stem:
        return {}
    before, after = stem.split("_to_", 1)
    if "_" not in before:
        return {}
    subject, fixed_state = before.rsplit("_", 1)
    moving_state = after.split("_", 1)[0]
    return {
        "subject": subject,
        "fixed_state": fixed_state,
        "moving_state": moving_state,
    }


def validate_preprocessed_states(
    npz_path: Path,
    subject: str,
    fixed_state: str,
    moving_state: str,
    log_fn: Callable[[str], None],
) -> None:
    """
    Ensure the NPZ metadata aligns with the CLI subject/state selection.
    Raises if metadata is missing or mismatched.
    """
    expected = {
        "subject": subject,
        "fixed_state": fixed_state,
        "moving_state": moving_state,
    }
    mesh_info = load_mesh_info(npz_path)
    provided = {k: mesh_info.get(k) for k in expected.keys()}
    missing = [k for k, v in provided.items() if not v]
    used_fallback = False
    if missing:
        inferred = infer_states_from_filename(npz_path)
        for key in missing:
            if inferred.get(key):
                provided[key] = inferred[key]
                used_fallback = True
    still_missing = [k for k, v in provided.items() if not v]
    if still_missing:
        raise RuntimeError(
            f"Preprocessed file {npz_path} lacks metadata for {', '.join(still_missing)}; "
            "regenerate it so subject/states are embedded."
        )
    mismatches = [
        key for key, expected_value in expected.items()
        if expected_value and provided.get(key) and provided[key] != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"Preprocessed file {npz_path} was generated for "
            f"{provided['subject']} {provided['fixed_state']}->{provided['moving_state']} "
            f"but CLI requested {subject} {fixed_state}->{moving_state}."
        )
    if used_fallback and not all(mesh_info.get(k) for k in expected):
        log_fn(
            f"Warning: {npz_path} does not store subject/state metadata; "
            "validated using filename pattern. Re-run preprocessing to embed metadata explicitly."
        )
