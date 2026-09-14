"""Training loops."""

from train_srnn.training.continuous import ContinuousTrainer
from train_srnn.training.trainer import EpochStats, Trainer
from train_srnn.training.windowed import WindowedTrainer

TRAINERS = {"windowed": WindowedTrainer, "continuous": ContinuousTrainer}

__all__ = ["ContinuousTrainer", "EpochStats", "Trainer", "WindowedTrainer", "TRAINERS"]
