"""Per-neuron linking and echo freezing hold with and without grad checkpointing."""
import torch

from train_srnn.config import compose_config
from train_srnn.models.factory import build_model

VARIANTS = ["srnn", "srnn-per-neuron", "srnn-echo"]


def _grads(grad_checkpoint: bool):
    torch.manual_seed(0)
    cfg = compose_config(["model=srnn", "task=synthetic", "seed=0", "model.num_units=16",
                          f"model.variants=[{','.join(VARIANTS)}]"])
    model = build_model(cfg)
    x = torch.randn(2, 30, 4, generator=torch.Generator().manual_seed(1))
    y = model(x, readout_idx=slice(10, 30), bptt_start_idx=5, bptt_chunk_len=5,
              grad_checkpoint=grad_checkpoint, grad_checkpoint_segment_len=5)
    y.pow(2).sum().backward()
    return {n: p.grad.clone() for n, p in model.cell.named_parameters() if p.grad is not None}


def test_linked_rows_get_no_gradient():
    for ckpt in (False, True):
        g = _grads(ckpt)
        for name, grad in g.items():
            if name.endswith("_vec"):
                assert torch.count_nonzero(grad[0]) == 0, (ckpt, name)   # srnn: linked
                assert torch.count_nonzero(grad[2]) == 0, (ckpt, name)   # echo: linked
                assert torch.count_nonzero(grad[1]) > 0, (ckpt, name)    # per-neuron trains
        assert torch.count_nonzero(g["W_raw"][2]) == 0                    # echo: frozen W
        assert torch.count_nonzero(g["W_raw"][0]) > 0
        assert torch.count_nonzero(g["W_raw_gain"][2]) > 0                # echo: gain still trains


def test_checkpointing_does_not_change_gradients():
    a, b = _grads(False), _grads(True)
    for name in a:
        assert torch.allclose(a[name], b[name], atol=1e-6, rtol=1e-5), name


def test_freeze_groups():
    cfg = compose_config(["model=srnn", "task=synthetic", "model.num_units=16",
                          "model.variants=[srnn-no-adapt]"])
    cell = build_model(cfg).cell
    assert cell.freeze(["a_0", "tau_d"]) == ["a_0_vec", "a_0_scalar", "isp_tau_d_vec", "log_tau_d_gain"]
    assert not cell.a_0_vec.requires_grad
    try:
        cell.freeze(["tau_a_E"])          # ablated away in srnn-no-adapt
    except ValueError as e:
        assert "no parameters" in str(e)
    else:
        raise AssertionError("expected ValueError")
