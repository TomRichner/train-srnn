"""Windowed trainer: shuffled mini-batches of augmented windows, state reset per batch."""
from __future__ import annotations

import time

import torch

from train_srnn.training.closed_loop import sample_alpha_schedule, summarize_alpha
from train_srnn.training.trainer import EpochStats, Trainer


class WindowedTrainer(Trainer):
    def steps_per_epoch(self) -> int:
        return len(self.data.train[0]) // int(self.cfg.task.batch_size) + 1

    def default_warmup_epochs(self) -> int:
        return 2

    def train_epoch(self, epoch: int) -> EpochStats:
        cfg, task = self.cfg, self.task
        cl_cfg = self.epoch_closed_loop_cfg(epoch)
        x_all, y_all = self.data.train
        loss_sum, metric_sum, n_total = [0.0] * self.K, [0.0] * self.K, 0
        alpha_sum, alpha_max, pure_tf, n_batches = 0.0, 0.0, 0, 0
        t0 = time.time()
        self.model.train()
        for idx in self.batches(x_all, training=True):
            b = task.train_batch(x_all[idx], y_all[idx], self.rng)
            x = torch.tensor(b.x, dtype=torch.float32, device=self.device)
            y = task.target_tensor(b.y, self.device)

            alpha = None
            if cl_cfg.enabled:
                alpha = sample_alpha_schedule(cl_cfg, T=x.shape[1], C=x.shape[2], device=self.device,
                                              generator=self.cl_gen, dtype=x.dtype)
                s = summarize_alpha(alpha)
                alpha_sum += s["alpha_mean"]
                alpha_max = max(alpha_max, s["alpha_max"])
                pure_tf += int(s["is_pure_tf"])
                n_batches += 1

            with self.autocast():
                logits = self.model(
                    x, readout_idx=b.readout_idx, bptt_start_idx=b.bptt_start_idx,
                    bptt_chunk_len=cfg.task.bptt_chunk_len, grad_checkpoint=cfg.grad_checkpoint,
                    grad_checkpoint_segment_len=cfg.grad_checkpoint_segment_len,
                    alpha_schedule=alpha)
                loss, losses, metrics = self.loss_and_metrics(logits, y)
            self.optimizer_step(loss)

            n = len(idx)
            for k in range(self.K):
                loss_sum[k] += losses[k] * n
                metric_sum[k] += metrics[k] * n
            n_total += n

        alpha_stats = None
        if cl_cfg.enabled and n_batches:
            alpha_stats = {"alpha_mean": alpha_sum / n_batches, "alpha_max": alpha_max,
                           "pure_tf_frac": pure_tf / n_batches, "n_batches": n_batches}
        return EpochStats([s / n_total for s in loss_sum], [s / n_total for s in metric_sum],
                          wall_s=time.time() - t0, alpha=alpha_stats)

    def run_epoch_and_log(self, epoch: int) -> None:
        train = self.train_epoch(epoch)
        valid = self.evaluate("valid")
        if epoch % self.log_interval == 0:
            self.log_epoch(epoch, train, valid)
        if epoch % self.checkpoint_interval == 0:
            tag = f"epoch_{epoch:03d}"
            self.checkpoint(epoch, tag)
            self.test(epoch, tag)
        self.record(epoch, train, valid)
