"""Record golden snapshots: forward outputs, gradients, and three trainer steps per cell.

Regenerate only when a numerical change is intended, and say so in the
commit; ``tests/test_golden_*.py`` replay the ``*.pt`` files.

    PYTHONPATH=. uv run python tests/golden/make_golden.py
"""
from __future__ import annotations

import tempfile
import pathlib

import torch

from tests.golden import harness as api


def snapshot_model(name: str, model) -> None:
    x, alpha = api.inputs()
    rec = {
        "torch": str(torch.__version__),
        "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "x": x, "alpha": alpha,
        "eval_last": api.run_eval(model, x),
    }
    for mode in api.FORWARD_MODES:
        y, grads = api.run_forward(model, x, alpha, mode)
        rec[mode] = {"y": y, "grads": grads}
    torch.save(rec, api.GOLDEN_DIR / f"cell_{name}.pt")
    print(f"cell_{name}: {len(rec['state_dict'])} tensors, "
          f"y{tuple(rec['open']['y'].shape)}")


def main() -> None:
    torch.use_deterministic_algorithms(True)
    for name in api.SINGLE_MODELS:
        snapshot_model(name, api.build_single(name))
    snapshot_model("batched", api.build_batched(api.BATCHED_VARIANTS))

    for cl in (False, True):
        tag = "cl" if cl else "tf"
        rec = api.run_windowed_epoch(closed_loop=cl)
        rec["torch"] = str(torch.__version__)
        torch.save(rec, api.GOLDEN_DIR / f"trainer_windowed_{tag}.pt")
        print(f"trainer_windowed_{tag}: loss={rec['train_loss']}")
        with tempfile.TemporaryDirectory() as d:
            rec = api.run_continuous_epoch(closed_loop=cl, out_dir=pathlib.Path(d))
        rec["torch"] = str(torch.__version__)
        torch.save(rec, api.GOLDEN_DIR / f"trainer_continuous_{tag}.pt")
        print(f"trainer_continuous_{tag}:\n{rec['training_history']}")


if __name__ == "__main__":
    main()
