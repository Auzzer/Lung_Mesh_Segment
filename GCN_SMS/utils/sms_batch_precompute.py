#!/usr/bin/env python
"""
Batch precomputation harness for SMS NPZ artifacts.

Features:
  * Enumerates subject/state combinations from the Emory 4DCT dataset.
  * Displays a tqdm progress bar while generating artifacts serially.
  * Maintains a JSONL manifest so reruns automatically resume from
    the first unfinished combination (skipping completed outputs).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple

from tqdm.auto import tqdm

_CURRENT_DIR = Path(__file__).resolve().parent
_SCRIPTS_ROOT = _CURRENT_DIR.parent
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from utils.sms_precompute_utils import (
    Emory4DCTDataset,
    get_emory_example_paths,
    normalize_mesh_tag,
    run_sms_preprocessor,
    validate_preprocessed_states,
)


DEFAULT_STATE_PAIR = ("T00", "T10")


@dataclass(frozen=True)
class PrecomputeTask:
    subject: str
    fixed_state: str
    moving_state: str
    mesh_path: Path
    disp_path: Path
    output_path: Path
    mask_name: str
    mesh_tag_raw: str
    metadata: dict

    @property
    def task_id(self) -> str:
        return "|".join(
            [
                self.subject,
                self.fixed_state,
                self.moving_state,
                self.mask_name,
                self.mesh_tag_raw,
            ]
        )


def parse_state_pairs(raw_pairs: Sequence[str]) -> List[Tuple[str, str]]:
    """Parse CLI provided fixed:moving strings."""
    pairs: List[Tuple[str, str]] = []
    for token in raw_pairs:
        if ":" not in token:
            raise ValueError(f"Invalid state pair '{token}'. Expected format FIXED:MOVING (e.g., T00:T50).")
        fixed, moving = token.split(":", 1)
        fixed = fixed.strip()
        moving = moving.strip()
        if not fixed or not moving:
            raise ValueError(f"Invalid state pair '{token}'. Fixed or moving component missing.")
        pairs.append((fixed, moving))
    return pairs


def adjacency_state_pairs(states: Sequence[str] | None) -> List[Tuple[str, str]]:
    """Return adjacent state pairs (wrap-around) or the fallback default."""
    if not states:
        return [DEFAULT_STATE_PAIR]
    ordered = list(states)
    if len(ordered) == 1:
        return [(ordered[0], ordered[0])]
    return [
        (ordered[i], ordered[(i + 1) % len(ordered)])
        for i in range(len(ordered))
    ]


def discover_subjects(ds: Emory4DCTDataset, subset: Sequence[str] | None) -> List[str]:
    if subset:
        return list(subset)
    subjects = ds.subjects()
    if not subjects:
        raise RuntimeError(f"No subjects found under {ds.root}")
    return subjects


def build_tasks(args) -> List[PrecomputeTask]:
    ds = Emory4DCTDataset(args.data_root)
    subjects = discover_subjects(ds, args.subjects)
    if args.state_pairs:
        state_pairs = parse_state_pairs(args.state_pairs)
    else:
        state_pairs = adjacency_state_pairs(getattr(ds, "VALID_STATES", None))
    mesh_tag_trim = normalize_mesh_tag(args.mask_name, args.mesh_tag)

    tasks: List[PrecomputeTask] = []
    for subject in subjects:
        for fixed_state, moving_state in state_pairs:
            try:
                paths = get_emory_example_paths(
                    data_root=args.data_root,
                    subject=subject,
                    variant=args.variant,
                    mask_name=args.mask_name,
                    mesh_tag=mesh_tag_trim,
                    fixed_state=fixed_state,
                    moving_state=moving_state,
                )
            except RuntimeError as exc:
                print(f"[skip] {subject} {fixed_state}->{moving_state}: {exc}")
                continue

            mesh_path = Path(paths['fixed_mesh'])
            disp_path = Path(paths['disp_field'])
            if not mesh_path.exists():
                print(f"[skip] Missing mesh for {subject} {fixed_state}: {mesh_path}")
                continue
            if not disp_path.exists():
                print(f"[skip] Missing displacement for {subject} {fixed_state}->{moving_state}: {disp_path}")
                continue

            output_dir = Path(args.cache_dir) / subject
            output_dir.mkdir(parents=True, exist_ok=True)
            output_name = f"{subject}_{fixed_state}_to_{moving_state}_{args.mask_name}_{mesh_tag_trim}"
            output_npz = output_dir / f"{output_name}.npz"

            metadata = {
                "subject": subject,
                "fixed_state": fixed_state,
                "moving_state": moving_state,
                "mask_name": args.mask_name,
                "mesh_tag": args.mesh_tag,
                "variant": args.variant,
            }
            tasks.append(
                PrecomputeTask(
                    subject=subject,
                    fixed_state=fixed_state,
                    moving_state=moving_state,
                    mesh_path=mesh_path,
                    disp_path=disp_path,
                    output_path=output_npz,
                    mask_name=args.mask_name,
                    mesh_tag_raw=args.mesh_tag,
                    metadata=metadata,
                )
            )
    return tasks


def load_progress(progress_file: Path) -> dict[str, dict]:
    if not progress_file.exists():
        return {}
    progress: dict[str, dict] = {}
    with progress_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = entry.get("task_id")
            if task_id:
                progress[task_id] = entry
    return progress


def append_progress(progress_file: Path, entry: dict) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with progress_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry))
        handle.write("\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mark_status(progress_file: Path, progress: dict, task: PrecomputeTask, status: str, message: str | None = None) -> None:
    entry = {
        "task_id": task.task_id,
        "status": status,
        "output": str(task.output_path),
        "timestamp": now_iso(),
    }
    if message:
        entry["message"] = message
    append_progress(progress_file, entry)
    progress[task.task_id] = entry


def is_task_completed(task: PrecomputeTask, progress: dict, force: bool, log_fn, progress_file: Path) -> bool:
    if force:
        return False

    entry = progress.get(task.task_id)
    if entry and entry.get("status") == "completed":
        if Path(entry.get("output", "")).exists():
            return True

    if task.output_path.exists():
        try:
            validate_preprocessed_states(task.output_path, task.subject, task.fixed_state, task.moving_state, log_fn)
        except Exception as exc:
            log_fn(f"[resume] Existing artifact {task.output_path} failed validation: {exc}. Will rebuild.")
            return False
        mark_status(progress_file, progress, task, "completed", "found existing file")
        return True

    return False


def run_task(task: PrecomputeTask) -> tuple[bool, str | None]:
    print(f"[run] {task.subject} {task.fixed_state}->{task.moving_state}: {task.output_path}")
    try:
        run_sms_preprocessor(
            mesh_path=task.mesh_path,
            displacement_path=task.disp_path,
            output_npz=task.output_path,
            metadata=task.metadata,
            log_fn=lambda msg: print(f"    {msg}"),
        )
    except Exception as exc:
        return False, f"preprocessor failed: {exc}"

    try:
        validate_preprocessed_states(
            task.output_path,
            task.subject,
            task.fixed_state,
            task.moving_state,
            log_fn=lambda msg: print(f"    {msg}"),
        )
    except Exception as exc:
        return False, f"validation failed: {exc}"

    return True, None


def main():
    parser = argparse.ArgumentParser(description="Batch SMS preprocessing with resume + progress bar.")
    parser.add_argument("--data-root", default="data/Emory-4DCT", help="Root directory for Emory 4DCT data.")
    parser.add_argument("--subjects", nargs="+", help="Optional list of subjects to process (default: all).")
    parser.add_argument("--state-pairs", nargs="+", help="State transitions (FIXED:MOVING). Default: T00:T50.")
    parser.add_argument("--variant", default="NIFTI", help="Dataset variant for meshes/masks.")
    parser.add_argument("--mask-name", default="lung_regions", help="Mask name used for meshing.")
    parser.add_argument(
        "--mesh-tag",
        default="lung_regions_11",
        help="Mesh tag suffix (with or without mask_name prefix).",
    )
    parser.add_argument("--cache-dir", default="data_processed_deformation", help="Where to store preprocessing NPZ files.")
    parser.add_argument(
        "--progress-file",
        help="Resume manifest path (default: <cache_dir>/sms_precompute_manifest.jsonl).",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild artifacts even if outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="List pending tasks without running preprocessing.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure.")
    args = parser.parse_args()

    tasks = build_tasks(args)
    if not tasks:
        print("No preprocessing tasks to run.")
        return

    progress_file = Path(args.progress_file) if args.progress_file else Path(args.cache_dir) / "sms_precompute_manifest.jsonl"
    progress = load_progress(progress_file)

    log_fn = lambda msg: print(f"    {msg}")

    pending: List[PrecomputeTask] = []
    completed_count = 0
    for task in tasks:
        if is_task_completed(task, progress, args.force, log_fn, progress_file):
            completed_count += 1
            continue
        pending.append(task)

    if args.dry_run:
        print("Pending preprocessing tasks:")
        for task in pending:
            print(f"  - {task.subject} {task.fixed_state}->{task.moving_state} -> {task.output_path}")
        print(f"{len(pending)} task(s) would run.")
        return

    pbar = tqdm(total=len(tasks), initial=completed_count, desc="SMS preprocessing")
    failures = 0
    for task in pending:
        ok, message = run_task(task)
        if ok:
            mark_status(progress_file, progress, task, "completed")
        else:
            failures += 1
            mark_status(progress_file, progress, task, "failed", message)
            print(f"[fail] {task.subject} {task.fixed_state}->{task.moving_state}: {message}")
            if args.fail_fast:
                pbar.close()
                raise SystemExit(1)
        pbar.update(1)
    pbar.close()

    if failures:
        print(f"Completed with {failures} failure(s). Check {progress_file} for details.")
    else:
        print("All preprocessing tasks completed successfully.")


if __name__ == "__main__":
    main()
