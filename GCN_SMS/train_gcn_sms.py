"""
Training skeleton: Pack_Mesh_HU -> TetGCN -> SMSLayer over all cases/timestamps.
Matches doc/gcn.md: HU->log-alpha linear baseline + residual GCN, one update per case.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import time
import subprocess

import numpy as np
import torch
from torch import optim

# Ensure repo root is on sys.path so we can import GCN_SMS modules when running from scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GCN_SMS.GCN_nn import TetGCN, build_neighbor_tensors
from GCN_SMS.sms_torch_layer import ConeStaticEquilibrium, SMSLayer, reset_taichi
from GCN_SMS.utils.Pack_Mesh_HU import pack_mesh_hu_sequence
from GCN_SMS.utils.sms_precompute_utils import run_sms_preprocessor
import re

# -----------------------------
# Config
# -----------------------------
CASE_IDS: Sequence[str] = [f"Case{i}Pack" for i in range(1, 11)]
HU_ROOT = REPO_ROOT / "data_processed_deformation"
SMS_ROOT = REPO_ROOT / "data_processed_deformation"
CT_ROOT = REPO_ROOT / "data" / "Emory-4DCT"
HU_SUFFIX = "_T00_forward_hu.npz"
# Adjust to match your naming; pattern used below.
SMS_PATTERN = "{case_id}_{fixed}_to_{moving}_lung_regions_11.npz"
TIMESTEP_PAIRS: Sequence[Tuple[str, str]] = [
    ("T00", "T10"),
    ("T10", "T20"),
    ("T20", "T30"),
    ("T30", "T40"),
    ("T40", "T50"),
    ("T50", "T60"),
    ("T60", "T70"),
    ("T70", "T80"),
    ("T80", "T90"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PACK_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE.type != "cuda":
    print("[WARN] CUDA not available; SMSLayer expects GPU.")


# -----------------------------
# 1) Scan HU range over all cases
# -----------------------------
def collect_hu_range(case_ids: Sequence[str]) -> Tuple[float, float]:
    global_min = float("inf")
    global_max = float("-inf")

    for case_id in case_ids:
        hu_npz_path, _ = ensure_hu_pack(case_id)
        data = np.load(hu_npz_path, allow_pickle=True)
        hu_mean = np.asarray(data["hu_tetra_mean"])[:, 0]  # (M,)
        case_min = float(hu_mean.min())
        case_max = float(hu_mean.max())
        global_min = min(global_min, case_min)
        global_max = max(global_max, case_max)
        print(f"[HU] {case_id}: min={case_min:.2f}, max={case_max:.2f}")

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError("No HU packs found; cannot compute global HU range.")

    print(f"[HU] Global range: min={global_min:.2f}, max={global_max:.2f}")
    return global_min, global_max


# -----------------------------
# HU pack helper (compute if missing)
# -----------------------------
def _case_subject_prefix(case_id: str) -> str:
    m = re.match(r"Case(\d+)Pack", case_id, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse case id: {case_id}")
    return f"case{m.group(1)}"


def _discover_corrfield_edges(case_id: str) -> Tuple[Dict[Tuple[str, str], Path], List[str]]:
    corr_dir = CT_ROOT / case_id / "CorrField"
    if not corr_dir.exists():
        raise FileNotFoundError(f"CorrField dir missing for {case_id}: {corr_dir}")
    edges: Dict[Tuple[str, str], Path] = {}
    times: set[str] = set()
    for f in corr_dir.glob("*.nii.gz"):
        stem = f.stem
        m = re.search(r"_T(\d{2})_T(\d{2})", stem, re.IGNORECASE)
        if not m:
            continue
        src = f"T{int(m.group(1)):02d}"
        tgt = f"T{int(m.group(2)):02d}"
        edges[(src, tgt)] = f
        times.update([src, tgt])
    if not edges:
        raise FileNotFoundError(f"No CorrField NIfTI found in {corr_dir}")
    ordered = sorted(times, key=lambda t: int(t[1:]))
    return edges, ordered


def _discover_adjacent_steps(case_id: str) -> List[Tuple[str, str, Path]]:
    edges, times = _discover_corrfield_edges(case_id)
    if "T00" not in times:
        raise RuntimeError(f"T00 not found in CorrField times for {case_id}")
    idx = times.index("T00")
    if idx == len(times) - 1:
        raise RuntimeError(f"No forward step after T00 for {case_id}")
    tgt = times[idx + 1]
    key = ("T00", tgt)
    if key not in edges:
        raise FileNotFoundError(f"Missing CorrField for step T00->{tgt} in {case_id}")
    return [("T00", tgt, edges[key])]


def ensure_sms_npz(
    case_id: str,
    fixed_t: str,
    moving_t: str,
    mask_name: str = "lung_regions_11",
    run_if_missing: bool = False,
    disp_path: Path | None = None,
) -> Tuple[Path, bool]:
    """
    Ensure SMS preprocessed NPZ exists.
    Returns (path, created_flag). If run_if_missing is False and file is absent, raises.
    """
    npz_name = SMS_PATTERN.format(case_id=case_id, fixed=fixed_t, moving=moving_t)
    pre_npz_path = SMS_ROOT / case_id / npz_name
    if pre_npz_path.exists():
        return pre_npz_path, False

    subj = _case_subject_prefix(case_id)
    # Mesh files use uppercase time tokens (e.g., case2_T00_lung_regions_11.xdmf)
    mesh_path = CT_ROOT / case_id / "pygalmesh" / f"{subj}_{fixed_t.upper()}_{mask_name}.xdmf"
    # CorrField files can be mixed case; try uppercase then lowercase
    if disp_path is None:
        disp_candidates = [
            CT_ROOT / case_id / "CorrField" / f"{subj}_{fixed_t.upper()}_{moving_t.upper()}.nii.gz",
            CT_ROOT / case_id / "CorrField" / f"{subj}_{fixed_t.lower()}_{moving_t.lower()}.nii.gz",
        ]
        disp_path = next((p for p in disp_candidates if p.exists()), disp_candidates[0])
    pre_npz_path.parent.mkdir(parents=True, exist_ok=True)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing mesh for SMS precompute: {mesh_path}")
    if not disp_path.exists():
        raise FileNotFoundError(f"Missing displacement for SMS precompute: {disp_path}")

    if not run_if_missing:
        raise FileNotFoundError(f"Missing SMS preprocessed file: {pre_npz_path}")

    max_retries = 6
    for attempt in range(1, max_retries + 1):
        try:
            run_sms_preprocessor(
                mesh_path=mesh_path,
                displacement_path=disp_path,
                output_npz=pre_npz_path,
                metadata={
                    "subject": case_id,
                    "fixed_state": fixed_t,
                    "moving_state": moving_t,
                    "mask_name": mask_name,
                    "mesh_tag": mask_name,
                    "variant": "Emory-4DCT",
                },
                log_fn=lambda msg: print(f"[sms_precompute] {msg}"),
            )
            break
        except subprocess.CalledProcessError as exc:
            # Kill any lingering sms_precompute subprocesses that may hold CUDA context
            subprocess.run(
                ["pkill", "-f", str(REPO_ROOT / "GCN_SMS/utils/sms_precompute.py")],
                check=False,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt == max_retries:
                raise RuntimeError(
                    f"SMS precompute failed after {max_retries} CUDA attempts for {pre_npz_path}"
                ) from exc
            wait_s = 10 * attempt
            print(f"[sms_precompute] Retry {attempt}/{max_retries} after CUDA error (sleep {wait_s}s)")
            time.sleep(wait_s)
    return pre_npz_path, True


def ensure_hu_pack(case_id: str, run_if_missing: bool = False) -> Tuple[Path, bool]:
    out_path = HU_ROOT / case_id / f"{case_id}{HU_SUFFIX}"
    if out_path.exists():
        return out_path, False

    steps = _discover_adjacent_steps(case_id)
    base_step = next((s for s in steps if s[0] == "T00"), None)
    if base_step is None:
        raise FileNotFoundError(f"No CorrField step starting at T00 for {case_id}")
    src, tgt, disp_path = base_step
    base_npz, _ = ensure_sms_npz(case_id, src, tgt, run_if_missing=run_if_missing, disp_path=disp_path)
    corrfield_dir = CT_ROOT / case_id / "CorrField"
    ct_dir = CT_ROOT / case_id / "NIFTI"

    if not base_npz.exists():
        raise FileNotFoundError(f"Missing base NPZ for HU pack: {base_npz}")
    if not corrfield_dir.exists():
        raise FileNotFoundError(f"Missing CorrField dir for {case_id}: {corrfield_dir}")
    if not ct_dir.exists():
        raise FileNotFoundError(f"Missing CT dir for {case_id}: {ct_dir}")

    if not run_if_missing:
        raise FileNotFoundError(f"Missing HU pack: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[HU] Computing forward HU pack for {case_id} -> {out_path}")
    pack_mesh_hu_sequence(
        base_npz=base_npz,
        corrfield_dir=corrfield_dir,
        ct_dir=ct_dir,
        output=out_path,
        base_time=None,
        device=PACK_DEVICE,
        chunk_size=200_000,
        mask_output=None,
    )
    return out_path, True


def precompute_if_missing() -> None:
    """
    Run all required precomputations (SMS NPZ for all timestep pairs and HU packs).
    If anything was created, exit so the user can rerun training on a clean start.
    """
    created_any = False
    for case_id in CASE_IDS:
        steps = _discover_adjacent_steps(case_id)
        for fixed_t, moving_t, disp_path in steps:
            _, created = ensure_sms_npz(case_id, fixed_t, moving_t, run_if_missing=True, disp_path=disp_path)
            created_any = created_any or created
        _, created = ensure_hu_pack(case_id, run_if_missing=True)
        created_any = created_any or created
    if created_any:
        print("[setup] Generated missing SMS/HU precomputations. Please rerun the training script.")
        sys.exit(0)


# -----------------------------
# 2) Per-case container
# -----------------------------
class CaseData:
    def __init__(
        self,
        case_id: str,
        hu_npz_path: Path,
        sms_root: Path,
    ):
        self.case_id = case_id
        self.hu_npz_path = hu_npz_path
        self.sms_root = sms_root
        self.timestep_pairs: List[Tuple[str, str]] = []
        self.sms_npz_paths: List[Path] = []

        data = np.load(hu_npz_path, allow_pickle=True)
        hu_mean = np.asarray(data["hu_tetra_mean"])[:, 0].astype(np.float32)  # (M,)
        tet_neighbor_indices = data["tet_neighbor_indices"].astype(np.int64)
        tet_neighbor_offsets = data["tet_neighbor_offsets"].astype(np.int64)

        self.hu = torch.from_numpy(hu_mean).to(device=DEVICE)
        self.neigh_idx, self.neigh_off = build_neighbor_tensors(
            torch.from_numpy(tet_neighbor_indices),
            torch.from_numpy(tet_neighbor_offsets),
            device=DEVICE,
        )
        # Discover adjacent CorrField steps (sequential times) and store SMS NPZ paths.
        steps = _discover_adjacent_steps(case_id)
        self.timestep_pairs = [(src, tgt) for src, tgt, _ in steps]
        for fixed_t, moving_t, disp_path in steps:
            pre_npz_path, _ = ensure_sms_npz(self.case_id, fixed_t, moving_t, disp_path=disp_path)
            self.sms_npz_paths.append(pre_npz_path)


# -----------------------------
# 3) Model/optimizer helpers
# -----------------------------
def build_model_and_optimizer(hu_min: float, hu_max: float, hidden_dim: int = 16, lr: float = 1e-3) -> Tuple[TetGCN, optim.Optimizer]:
    model = TetGCN(hidden_dim=hidden_dim).to(device=DEVICE)
    model.init_linear_from_hu_range(hu_min, hu_max)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


# -----------------------------
# 4) Training loop
# -----------------------------
def train(num_epochs: int = 50) -> None:
    # Precompute any missing artifacts, then exit to rerun if new files were created.
    precompute_if_missing()
    # 4.1 global HU range
    hu_min, hu_max = collect_hu_range(CASE_IDS)
    # 4.2 build model + optimizer
    model, optimizer = build_model_and_optimizer(hu_min, hu_max, hidden_dim=16, lr=1e-3)
    # 4.3 enforce load or compute data
    case_data: Dict[str, CaseData] = {}
    for case_id in CASE_IDS:
        hu_npz_path, _ = ensure_hu_pack(case_id)
        case_data[case_id] = CaseData(case_id, hu_npz_path, SMS_ROOT)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_cases_used = 0

        for case_id in CASE_IDS:
            if case_id not in case_data:
                continue
            cd = case_data[case_id]
            if not cd.sms_npz_paths:
                continue  # skip if no SMS layers

            optimizer.zero_grad()
            # # 4.3.1: HU -> log-alpha via GCN (one field per case)
            alpha, log_alpha = model(cd.hu, cd.neigh_idx, cd.neigh_off)# shapes: (M,), (M,)
            # SMS expects delta = per-tet log-parameter, theta = scalar offset
            delta = log_alpha.to(dtype=torch.float64, device=DEVICE)
            theta = torch.tensor(0.0, dtype=torch.float64, device=DEVICE) # global log-param (kept 0, not trained)
            # 4.3.2: accumulate loss over timestamps (graph-conv "RNN" over time)
            loss_case = 0.0
            n_steps = len(cd.sms_npz_paths)
            for sms_npz in cd.sms_npz_paths:
                sim = ConeStaticEquilibrium(str(sms_npz))
                u_obs_free = sim.get_observed_free()
                sms_layer = SMSLayer(
                    sim,
                    u_obs_free=u_obs_free,
                    omega_u=0.5,
                    omega_alpha=0.5,
                    return_components=True,
                )
                loss_t, _extras = sms_layer(theta, delta)
                loss_case = loss_case + loss_t / n_steps
                del sms_layer
                del sim
                # Release Taichi state after each SMS solve iteration
                #reset_taichi()
            # 4.3.3: backprop and update
            loss_case.backward()
            optimizer.step()

            epoch_loss += float(loss_case.item())
            n_cases_used += 1
            # Release Taichi state between cases to avoid snode tree buildup
            #reset_taichi()

        if n_cases_used > 0:
            epoch_loss /= n_cases_used
        print(f"[Epoch {epoch:03d}] mean case loss = {epoch_loss:.6f}")
        # Release Taichi between epochs to avoid snode buildup
        reset_taichi()


if __name__ == "__main__":
    train(num_epochs=50)
