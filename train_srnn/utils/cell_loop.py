"""Two helpers for the per-timestep cell loop."""
import torch


def mark_cudagraph_step() -> None:
    """Tell the CUDA-graph allocator a step ended; a no-op outside graph capture."""
    torch.compiler.cudagraph_mark_step_begin()


def empty_time_buffer(value: torch.Tensor, T: int) -> torch.Tensor:
    """Uninitialised ``(*value.shape[:-1], T, value.shape[-1])`` buffer to fill by slice-assign."""
    return torch.empty(value.shape[:-1] + (T, value.shape[-1]), device=value.device, dtype=value.dtype)
