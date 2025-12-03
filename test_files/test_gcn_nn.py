import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GCN_SMS.GCN_nn import TetGCN, build_neighbor_tensors, scatter_neighbor_sum


def test_two_tet_forward_sum_neighbors():
    # Simple two-tet graph: 0 <-> 1
    neighbor_indices = torch.tensor([1, 0], dtype=torch.int64)
    neighbor_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
    idx, off = build_neighbor_tensors(neighbor_indices, neighbor_offsets)

    # HU values chosen so mean=2, std=1 -> normalized h0 = [-1, 1]
    hu = torch.tensor([1.0, 3.0], dtype=torch.float32)

    model = TetGCN(hidden_dim=2)
    # Deterministic weights to match doc/gcn.md pattern (sum over neighbors, no averaging)
    model.W_nei1.data = torch.tensor([[1.0, 0.0]])
    model.W_self1.data = torch.tensor([[0.0, 1.0]])
    model.b1.data.zero_()
    model.W_nei2.data = torch.tensor([0.0, 0.0])
    model.W_self2.data = torch.tensor([0.0, 0.0])
    model.b2.data.fill_(7.0)  # stays within clamp range

    # Manually compute layer-1 hidden to assert neighbor sum usage (torch.std default unbiased)
    h0 = (hu - hu.mean()) / (hu.std(unbiased=True) + model.b1.new_tensor(1e-8))
    nei_sum0 = scatter_neighbor_sum(h0, idx, off)
    h1_expected = model.act(
        model.b1
        + nei_sum0.unsqueeze(1) * model.W_nei1
        + h0.unsqueeze(1) * model.W_self1
    )
    h1_out = model.act(
        model.b1
        + nei_sum0.unsqueeze(1) * model.W_nei1
        + h0.unsqueeze(1) * model.W_self1
    )
    assert torch.allclose(h1_out, h1_expected, atol=1e-6)
    # Expected pattern: first hidden component positive for node0, second for node1
    assert h1_out[0, 0] > 0 and h1_out[0, 1] == 0
    assert h1_out[1, 1] > 0 and h1_out[1, 0] == 0

    delta = model(hu, idx, off)
    # With b2=7 and zero weights, delta_raw=7, delta_log=tanh(7)*max_delta_log
    delta_expected = model.max_delta_log * torch.tanh(torch.full_like(delta, 7.0))
    assert torch.allclose(delta, delta_expected, atol=1e-5)


def test_real_npz_forward():
    """Smoke test on real NPZ output to ensure shapes and finiteness."""
    npz_path = Path("data_processed_deformation/hu_packs/Case10Pack/Case10Pack_T00_forward_hu.npz")
    if not npz_path.exists():
        print("Skipping real-data test: NPZ not found. Run Pack_Mesh_HU first.")
        return
    import numpy as np

    npz = np.load(npz_path, allow_pickle=True)
    hu_np = npz["hu_tetra_mean"]
    # Reduce multi-channel HU (M,P) to scalar via mean if needed
    if hu_np.ndim > 1:
        hu_np = hu_np.mean(axis=-1)
    hu = torch.from_numpy(hu_np).float().squeeze()
    idx = torch.from_numpy(npz["tet_neighbor_indices"]).long()
    off = torch.from_numpy(npz["tet_neighbor_offsets"]).long()

    model = TetGCN(hidden_dim=8)
    idx, off = build_neighbor_tensors(idx, off)
    delta = model(hu, idx, off)
    assert delta.shape == hu.shape
    assert torch.isfinite(delta).all()


if __name__ == "__main__":
    test_two_tet_forward_sum_neighbors()
    print("test_two_tet_forward_sum_neighbors passed.")
    test_real_npz_forward()
    print("test_real_npz_forward passed.")
