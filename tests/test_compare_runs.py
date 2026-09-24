"""scripts/compare_runs.py: config filtering, history and checkpoint differences."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import compare_runs  # noqa: E402


def _run(path, loss, weight, output_dir):
    path.mkdir()
    (path / "training_history.csv").write_text(
        "epoch,variant,train_loss,lr,timestamp\n"
        f"0,srnn,{loss},0.001,2026-01-01T00:00:00Z\n")
    (path / "test_history.csv").write_text(
        "epoch,tag,variant,test_loss,timestamp\n0,last,srnn,0.5,2026-01-01T00:00:00Z\n")
    config = {"lr": 0.001, "output_dir": output_dir, "paths": {"results": output_dir},
              "model": {"num_units": 300}}
    for name in ("init.pt", "last.pt"):
        torch.save({"model_state_dict": {"w": torch.tensor([1.0, weight])}, "config": config},
                   path / name)


def test_compare_ignores_location_and_reports_differences(tmp_path):
    _run(tmp_path / "a", 0.25, 2.0, "/opt/a")
    _run(tmp_path / "b", 0.2501, 2.5, "/vol/b")
    report = compare_runs.compare(tmp_path / "a", tmp_path / "b")
    assert report["config_diff"] == {}
    assert abs(report["training_history.csv"]["train_loss"]["max_abs"] - 1e-4) < 1e-12
    assert report["test_history.csv"]["test_loss"]["max_abs"] == 0
    assert report["checkpoints"]["last.pt"]["max_abs"] == 0.5
    assert report["checkpoints"]["last.pt"]["worst_tensor"] == "w"
    assert not compare_runs.has_errors(report)


def test_compare_flags_missing_rows(tmp_path):
    _run(tmp_path / "a", 0.25, 2.0, "/a")
    _run(tmp_path / "b", 0.25, 2.0, "/b")
    with (tmp_path / "b" / "training_history.csv").open("a") as f:
        f.write("1,srnn,0.2,0.001,2026-01-01T00:00:00Z\n")
    assert compare_runs.has_errors(compare_runs.compare(tmp_path / "a", tmp_path / "b"))
