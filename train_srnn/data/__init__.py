"""Tasks, datasets, and augmentation.

Importing the submodules registers every task in ``TASKS``.
"""

from train_srnn.data import cheetah100, classic, synthetic  # noqa: F401  (registration)
from train_srnn.data.task import TASKS, Batch, Dataset, Task, TraceTask, build_task

__all__ = ["TASKS", "Batch", "Dataset", "Task", "TraceTask", "build_task"]
