"""Typed configuration for training runs.

Every task and model is a dataclass registered with Hydra's ConfigStore, so
``python train.py task=cheetah100 model=srnn model.num_units=64`` composes a
fully typed config and a misspelled key fails at startup. ``conf/config.yaml``
only lists the defaults and Hydra's run-dir settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING, DictConfig

CONF_DIR = Path(__file__).resolve().parent.parent / "conf"


# Run-level blocks

@dataclass
class PathsConfig:
    """Resolved from $SRNN_HOME by default; see train_srnn/paths.py."""
    data_dir: str = "${srnn_path:data}"
    results_dir: str = "${srnn_path:results}"


@dataclass
class CompileConfig:
    enabled: bool = True          # torch.compile on CUDA; ignored elsewhere
    cell_only: bool = True        # compile model.cell in the continuous trainer, not the whole model
    mode: Optional[str] = None    # torch.compile(mode=...); only the default mode is supported
    dynamic: Optional[bool] = None
    log_recompiles: bool = False


@dataclass
class ClosedLoopConfig:
    """Variable teacher forcing: x_in = (1 - alpha) * x_real + alpha * y_pred[t-1]."""
    enabled: bool = False
    teacher_forcing_batch_frac: float = 0.2   # fraction of batches with alpha = 0
    alpha_baseline: float = 0.3
    alpha_baseline_start: Optional[float] = None   # ramp from here to alpha_baseline over the run
    alpha_baseline_jitter: float = 0.0        # half-range of uniform jitter on the baseline
    alpha_rnd_density: float = 0.0            # per-channel sparse Gaussian perturbation
    alpha_rnd_sigma: float = 0.0
    t_warm: int = 2500                        # half-cosine ramp length in samples (windowed trainer only)
    alpha_rnd_period_epochs: int = 10         # per-channel rotation period (continuous trainer only)


# Tasks

@dataclass
class TaskConfig:
    name: str = MISSING
    dataset: str = MISSING            # folder under paths.data_dir
    data_dir: str = "${paths.data_dir}/${task.dataset}"
    task_type: str = MISSING          # classification | regression
    input_size: int = MISSING
    output_size: int = MISSING
    per_timestep_labels: bool = True
    trainer: str = "windowed"         # windowed | continuous
    batch_size: int = 128
    seq_len: int = MISSING            # native sequence length produced by the loader
    h: float = 0.02                   # ODE step (s); models read it via ${task.h}
    ode_unfolds: int = 1
    window_len: int = 1024            # samples fed to the model per window
    bptt_len: int = 512               # gradient horizon in samples
    bptt_chunk_len: Optional[int] = 25   # state.detach() every N steps; None = one graph
    loss_over_bptt: bool = False      # loss at every step of the grad region
    no_augment: bool = False          # skip stretch + palindrome; requires seq_len == window_len
    stretch_lo: float = 0.8
    stretch_hi: float = 1.2


@dataclass
class ClassicTaskConfig(TaskConfig):
    """Benchmark tasks from Hasani et al.; windowed trainer with augmentation."""


@dataclass
class HarConfig(ClassicTaskConfig):
    name: str = "har"; dataset: str = "har"; task_type: str = "classification"
    input_size: int = 561; output_size: int = 6; seq_len: int = 16


@dataclass
class SmnistConfig(ClassicTaskConfig):
    name: str = "smnist"; dataset: str = "smnist"; task_type: str = "classification"
    input_size: int = 28; output_size: int = 10; seq_len: int = 28
    per_timestep_labels: bool = False


@dataclass
class GestureConfig(ClassicTaskConfig):
    name: str = "gesture"; dataset: str = "gesture"; task_type: str = "classification"
    input_size: int = 32; output_size: int = 5; seq_len: int = 32; batch_size: int = 16


@dataclass
class OccupancyConfig(ClassicTaskConfig):
    name: str = "occupancy"; dataset: str = "occupancy"; task_type: str = "classification"
    input_size: int = 5; output_size: int = 2; seq_len: int = 16


@dataclass
class OzoneConfig(ClassicTaskConfig):
    name: str = "ozone"; dataset: str = "ozone"; task_type: str = "classification"
    input_size: int = 72; output_size: int = 2; seq_len: int = 32; batch_size: int = 16


@dataclass
class PersonConfig(ClassicTaskConfig):
    name: str = "person"; dataset: str = "person"; task_type: str = "classification"
    input_size: int = 7; output_size: int = 7; seq_len: int = 32


@dataclass
class PowerConfig(ClassicTaskConfig):
    name: str = "power"; dataset: str = "power"; task_type: str = "regression"
    input_size: int = 6; output_size: int = 1; seq_len: int = 32


@dataclass
class TrafficConfig(ClassicTaskConfig):
    name: str = "traffic"; dataset: str = "traffic"; task_type: str = "regression"
    input_size: int = 7; output_size: int = 1; seq_len: int = 32


@dataclass
class TraceTaskConfig(TaskConfig):
    """One long trace per split, trained with the continuous ring trainer."""
    trainer: str = "continuous"
    task_type: str = "regression"
    stride: int = 500                 # window stride for the valid/test arrays
    normalize: bool = True            # per-channel z-score with train-split stats
    train_trace_max_len: Optional[int] = None   # truncate the train trace (pick a prime)
    no_augment: bool = True
    loss_over_bptt: bool = True
    stretch_lo: float = 1.0
    stretch_hi: float = 1.0


@dataclass
class Cheetah100Config(TraceTaskConfig):
    name: str = "cheetah100"; dataset: str = "cheetah100"
    include_actions: bool = False     # append the 6 control channels
    skip_transient_s: float = 10.0    # drop the standstill-to-cruise ramp at the head of each trace
    sample_rate_hz: float = 100.0
    train_trace_max_len: Optional[int] = 118973   # prime, coprime to batch_size * bptt_chunk_len
    h: float = 0.01                   # model time == physical time at 100 Hz
    input_size: int = 17; output_size: int = 17
    seq_len: int = 1500; stride: int = 500       # 15 s windows, 5 s stride
    batch_size: int = 24              # parallel readers around the ring
    window_len: int = 1500; bptt_len: int = 1000; bptt_chunk_len: Optional[int] = 250


@dataclass
class Cheetah100ActConfig(Cheetah100Config):
    name: str = "cheetah100_act"
    include_actions: bool = True
    input_size: int = 23; output_size: int = 23


@dataclass
class SyntheticConfig(TraceTaskConfig):
    """Deterministic sinusoid trace for data-free smoke tests."""
    name: str = "synthetic"; dataset: str = "synthetic"
    seed: int = 0
    train_samples: int = 4001; eval_samples: int = 401
    sample_rate_hz: float = 100.0
    train_trace_max_len: Optional[int] = 3989   # prime
    h: float = 0.01
    input_size: int = 4; output_size: int = 4
    seq_len: int = 100; stride: int = 50
    batch_size: int = 4
    window_len: int = 100; bptt_len: int = 50; bptt_chunk_len: Optional[int] = 10


TASK_CONFIGS: dict[str, type[TaskConfig]] = {
    "har": HarConfig, "smnist": SmnistConfig, "gesture": GestureConfig,
    "occupancy": OccupancyConfig, "ozone": OzoneConfig, "person": PersonConfig,
    "power": PowerConfig, "traffic": TrafficConfig,
    "cheetah100": Cheetah100Config, "cheetah100_act": Cheetah100ActConfig,
    "synthetic": SyntheticConfig,
}


# Models

@dataclass
class ModelConfig:
    name: str = MISSING
    type: str = MISSING
    num_units: int = 300


@dataclass
class LSTMModelConfig(ModelConfig):
    name: str = "lstm"; type: str = "lstm"


@dataclass
class LTCModelConfig(ModelConfig):
    name: str = "ltc"; type: str = "ltc"
    solver: str = "semi_implicit"     # semi_implicit | explicit | rk4
    ode_unfolds: int = 6
    h: float = 0.1                    # step for the explicit solvers
    erev_init_factor: float = 1.0
    w_init_min: float = 0.01
    w_init_max: float = 1.0
    gleak_init_min: float = 0.001
    gleak_init_max: float = 1.0
    cm_init_min: float = 0.4
    cm_init_max: float = 0.6
    fix_vleak: bool = False
    fix_gleak: bool = False
    fix_cm: bool = False


@dataclass
class LTCRKModelConfig(LTCModelConfig):
    name: str = "ltc_rk"; solver: str = "rk4"


@dataclass
class LTCExplicitModelConfig(LTCModelConfig):
    name: str = "ltc_ex"; solver: str = "explicit"


@dataclass
class CTRNNModelConfig(ModelConfig):
    name: str = "ctrnn"; type: str = "ctrnn"
    solver: str = "euler"             # euler | rk4
    global_feedback: bool = True
    cell_clip: float = 0.0
    unfolds: int = 6
    h: float = 0.1
    fix_tau: bool = True
    tau: float = 1.0


@dataclass
class NODEModelConfig(CTRNNModelConfig):
    """Neural ODE: RK4 on the leak-free CTRNN vector field, step from the task."""
    name: str = "node"; type: str = "node"
    solver: str = "rk4"
    h: float = II("task.h")


@dataclass
class CTGRUModelConfig(ModelConfig):
    name: str = "ctgru"; type: str = "ctgru"
    M: int = 8                        # parallel timescales
    tau_base: float = 1.0
    cell_clip: float = -1.0           # negative disables


@dataclass
class RMTConfig:
    """Random recurrent connectivity (Harris et al. 2023)."""
    density: float = 1.0 / 3.0
    level_of_chaos: float = 1.0


@dataclass
class SRNNModelConfig(ModelConfig):
    """Shared settings for every SRNN variant in the batch.

    Variant names (``model.variants``) toggle the flags below per network; see
    train_srnn/models/variants.py for the grammar. Per-variant flags listed
    here are the base values that the name tokens modify.
    """
    name: str = "srnn"; type: str = "srnn"
    variants: list[str] = field(default_factory=lambda: ["srnn"])
    variant_seeds: Optional[list[int]] = None   # cross variants with recurrent-matrix seeds
    dales: bool = True
    n_a_E: int = 3                    # SFA timescales on E neurons (0 = off)
    n_a_I: int = 3
    n_b_E: int = 1                    # STD on E neurons (0/1)
    n_b_I: int = 1
    per_neuron: bool = False
    echo: bool = False                # frozen recurrent weights (reservoir)
    skip: bool = False                # y = readout(state) + x; autoregressive tasks only
    solver: str = "semi_implicit"     # semi_implicit | explicit | rk4
    h: float = II("task.h")
    ode_unfolds: int = II("task.ode_unfolds")
    readout: str = "synaptic"
    tau_global_init: float = 1.0
    tau_a_lo_init: float = 0.25       # fastest SFA timescale (s)
    tau_a_hi_init: float = 4.0        # slowest SFA timescale (s)
    std_zero_floor: bool = True       # rescale STD state so synaptic gain reaches 0 at saturation
    rmt: RMTConfig = field(default_factory=RMTConfig)


MODEL_CONFIGS: dict[str, type[ModelConfig]] = {
    "lstm": LSTMModelConfig, "ltc": LTCModelConfig, "ltc_rk": LTCRKModelConfig,
    "ltc_ex": LTCExplicitModelConfig, "ctrnn": CTRNNModelConfig,
    "node": NODEModelConfig, "ctgru": CTGRUModelConfig, "srnn": SRNNModelConfig,
}


# Run

@dataclass
class TrainConfig:
    model: Any = MISSING
    task: Any = MISSING
    paths: PathsConfig = field(default_factory=PathsConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    closed_loop: ClosedLoopConfig = field(default_factory=ClosedLoopConfig)

    seed: int = 1
    epochs: int = 200
    lr: float = 5e-4
    cosine_decay: bool = False        # cosine-decay to lr/20 after the hold phase
    warmup_epochs: Optional[int] = None   # None: 2 (windowed) or round(B/2) (continuous)
    grad_clip: float = 1.0            # per-variant gradient-norm clip; 0 disables
    grad_checkpoint: bool = True      # torch.utils.checkpoint over grad-region segments
    grad_checkpoint_segment_len: Optional[int] = 5
    burn_in: float = 10.0             # seconds of unforced dynamics to initialise the IC
    burn_in_every: int = 1            # re-run burn-in every N epochs; 0 = init only
    freeze_ic_after_burnin: bool = True
    log_interval: Optional[int] = None        # None: trainer default
    checkpoint_interval: Optional[int] = None
    device: str = "auto"              # auto | cpu | cuda | mps
    amp: str = "fp32"                 # fp32 | bf16
    freeze_params: list[str] = field(default_factory=list)   # SRNN params pinned at init
    init_ckpt: Optional[str] = None   # local path or gs:// URL to resume from
    early_exit_after_init: bool = False
    profile: bool = False             # per-phase timing in the continuous trainer

    run_name: str = "${model.name}_${model.num_units}"
    output_dir: str = "${paths.results_dir}/${task.name}/${run_name}"


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=TrainConfig)
    for name, cls in TASK_CONFIGS.items():
        cs.store(group="task", name=name, node=cls)
    for name, cls in MODEL_CONFIGS.items():
        cs.store(group="model", name=name, node=cls)


def compose_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose the training config outside ``@hydra.main`` (tests, scripts)."""
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return compose("config", overrides=list(overrides or []))
