"""Typed config composition: every task/model composes; unknown keys fail."""
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from train_srnn.config import MODEL_CONFIGS, TASK_CONFIGS, compose_config


@pytest.mark.parametrize("task", sorted(TASK_CONFIGS))
def test_every_task_composes(task):
    cfg = compose_config([f"task={task}"])
    assert cfg.task.name == task
    assert cfg.task.data_dir.endswith(f"/{cfg.task.dataset}")


@pytest.mark.parametrize("model", sorted(MODEL_CONFIGS))
def test_every_model_composes(model):
    cfg = compose_config([f"model={model}", "task=har"])
    assert cfg.model.name == model
    OmegaConf.to_container(cfg, resolve=True)


def test_model_h_follows_task():
    cfg = compose_config(["model=srnn", "task=cheetah100"])
    assert cfg.model.h == cfg.task.h == 0.01
    cfg = compose_config(["model=node", "task=har", "task.h=0.05"])
    assert cfg.model.h == 0.05


@pytest.mark.parametrize("override", ["task.bogus=1", "model.sparsity=0.5", "batched_ablations=[srnn]"])
def test_unknown_keys_fail(override):
    with pytest.raises(Exception) as e:
        compose_config([override])
    assert "not in" in str(e.value) or "Could not override" in str(e.value)


def test_output_dir_uses_run_name():
    cfg = compose_config(["run_name=abc", "paths.results_dir=/r"])
    assert cfg.output_dir == "/r/cheetah100/abc"
