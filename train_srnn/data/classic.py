"""The eight benchmark tasks of Hasani et al.

Ported from https://github.com/raminmh/liquid_time_constant_networks
(experiments_with_ltcs/{har,smnist,gesture,occupancy,traffic,power,ozone,person}.py),
Apache License 2.0. Copyright (c) the original authors. Modifications
copyright (c) 2026 Thomas Richner. Windowing, splits, and split seeds match
the original scripts so results stay comparable.
"""
from __future__ import annotations

import datetime as dt
import gzip
import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from train_srnn.data.task import TASKS, Dataset, Task


# ---------------------------------------------------------------------------
# Windowing and splits shared by the loaders
# ---------------------------------------------------------------------------

def cut_in_sequences(data, labels, seq_len, inc=1):
    """Cut ``(N, F)`` data and its labels into overlapping windows of ``seq_len``."""
    xs, ys = [], []
    for s in range(0, len(data) - seq_len, inc):
        xs.append(data[s:s + seq_len])
        ys.append(labels[s:s + seq_len])
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def _split_75_10_15(x, y, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(x))
    n_valid, n_test = int(0.1 * len(x)), int(0.15 * len(x))
    valid, test, train = perm[:n_valid], perm[n_valid:n_valid + n_test], perm[n_valid + n_test:]
    return x[train], y[train], x[valid], y[valid], x[test], y[test]


def _split_90_10(x, y, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(x))
    n_valid = int(0.1 * len(x))
    valid, train = perm[:n_valid], perm[n_valid:]
    return x[train], y[train], x[valid], y[valid]


def _zscore_columns(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = std.copy()
    std[std < 1e-8] = 1.0
    return (x - mean) / std


class ClassicTask(Task):
    """Windowed classification/regression task with the standard split sizes."""

    def _dataset(self, train, valid, test) -> Dataset:
        c = self.cfg
        return Dataset(train=train, valid=valid, test=test,
                       input_size=c.input_size, output_size=c.output_size)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def _read_idx_gz(path):
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        dtype = {0x08: np.uint8, 0x09: np.int8}[(magic >> 8) & 0xFF]
        ndim = magic & 0xFF
        dims = [struct.unpack(">I", f.read(4))[0] for _ in range(ndim)]
        return np.frombuffer(f.read(), dtype=dtype).reshape(dims)


@TASKS.register("smnist")
class SmnistTask(ClassicTask):
    """Row-wise sequential MNIST: 28 steps of 28 pixels, one label per image."""

    def load(self, data_dir: Path) -> Dataset:
        idx = Path(data_dir) / "train-images-idx3-ubyte.gz"
        if idx.is_file():
            d = Path(data_dir)
            train_x = _read_idx_gz(idx).astype(np.float32) / 255.0
            train_y = _read_idx_gz(d / "train-labels-idx1-ubyte.gz").astype(np.int64)
            test_x = _read_idx_gz(d / "t10k-images-idx3-ubyte.gz").astype(np.float32) / 255.0
            test_y = _read_idx_gz(d / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
        else:
            from torchvision import datasets
            cache = os.path.join(data_dir, "mnist_cache")
            train_ds = datasets.MNIST(cache, train=True, download=True)
            test_ds = datasets.MNIST(cache, train=False, download=True)
            train_x = train_ds.data.numpy().astype(np.float32) / 255.0
            train_y = train_ds.targets.numpy().astype(np.int64)
            test_x = test_ds.data.numpy().astype(np.float32) / 255.0
            test_y = test_ds.targets.numpy().astype(np.int64)
        split = int(0.9 * len(train_x))
        return self._dataset((train_x[:split], train_y[:split]),
                             (train_x[split:], train_y[split:]),
                             (test_x, test_y))


@TASKS.register("har")
class HarTask(ClassicTask):
    """UCI Human Activity Recognition, 16-step windows over the feature stream."""

    def load(self, data_dir: Path) -> Dataset:
        base = Path(data_dir) / "UCI HAR Dataset"
        train_x = np.loadtxt(base / "train" / "X_train.txt").astype(np.float32)
        train_y = (np.loadtxt(base / "train" / "y_train.txt") - 1).astype(np.int64)
        test_x = np.loadtxt(base / "test" / "X_test.txt").astype(np.float32)
        test_y = (np.loadtxt(base / "test" / "y_test.txt") - 1).astype(np.int64)
        seq_len = self.cfg.seq_len
        train_x, train_y = cut_in_sequences(train_x, train_y, seq_len, inc=1)
        test_x, test_y = cut_in_sequences(test_x, test_y, seq_len, inc=8)
        train_x, train_y, valid_x, valid_y = _split_90_10(train_x, train_y, seed=893429)
        return self._dataset((train_x, train_y), (valid_x, valid_y), (test_x, test_y))


@TASKS.register("gesture")
class GestureTask(ClassicTask):
    """Gesture phase segmentation from 32-channel motion features."""

    FILES = ["a3_va3.csv", "b1_va3.csv", "b3_va3.csv", "c1_va3.csv",
             "c3_va3.csv", "a2_va3.csv", "a1_va3.csv"]
    PHASES = {"D": 0, "P": 1, "S": 2, "H": 3, "R": 4}

    def load(self, data_dir: Path) -> Dataset:
        seq_len = self.cfg.seq_len
        windows = []
        for f in self.FILES:
            df = pd.read_csv(Path(data_dir) / f, header=0)
            y = np.array([self.PHASES[s] for s in df["Phase"].values], dtype=np.int64)
            x = df.values[:, :-1].astype(np.float32)
            n = x.shape[0] // seq_len
            for s in range(n):
                start = seq_len * s
                windows.append((x[start:start + seq_len], y[start:start + seq_len]))
                if s < n - 1:   # interleaved half-offset window
                    start2 = start + seq_len // 2
                    windows.append((x[start2:start2 + seq_len], y[start2:start2 + seq_len]))
        all_x = np.stack([w[0] for w in windows], axis=0)
        all_y = np.stack([w[1] for w in windows], axis=0)
        flat = all_x.reshape(-1, all_x.shape[-1])
        all_x = _zscore_columns(all_x, flat.mean(axis=0), flat.std(axis=0))
        tr_x, tr_y, va_x, va_y, te_x, te_y = _split_75_10_15(all_x, all_y, seed=23489)
        return self._dataset((tr_x, tr_y), (va_x, va_y), (te_x, te_y))


@TASKS.register("occupancy")
class OccupancyTask(ClassicTask):
    """Room occupancy from five environmental sensors."""

    COLUMNS = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

    def _read(self, path: Path):
        df = pd.read_csv(path)
        x = np.stack([df[c].values for c in self.COLUMNS], axis=-1).astype(np.float32)
        return x, df["Occupancy"].values.astype(np.int64)

    def load(self, data_dir: Path) -> Dataset:
        d, seq_len = Path(data_dir), self.cfg.seq_len
        train_x, train_y = self._read(d / "datatraining.txt")
        test0_x, test0_y = self._read(d / "datatest.txt")
        test1_x, test1_y = self._read(d / "datatest2.txt")
        mean, std = train_x.mean(axis=0), train_x.std(axis=0)
        train_x, test0_x, test1_x = (_zscore_columns(a, mean, std) for a in (train_x, test0_x, test1_x))
        train_x, train_y = cut_in_sequences(train_x, train_y, seq_len, inc=1)
        test0_x, test0_y = cut_in_sequences(test0_x, test0_y, seq_len, inc=8)
        test1_x, test1_y = cut_in_sequences(test1_x, test1_y, seq_len, inc=8)
        train_x, train_y, valid_x, valid_y = _split_90_10(train_x, train_y, seed=893429)
        test_x = np.concatenate([test0_x, test1_x], axis=0)
        test_y = np.concatenate([test0_y, test1_y], axis=0)
        return self._dataset((train_x, train_y), (valid_x, valid_y), (test_x, test_y))


@TASKS.register("traffic")
class TrafficTask(ClassicTask):
    """Metro Interstate traffic volume regression from weather and time features."""

    def load(self, data_dir: Path) -> Dataset:
        df = pd.read_csv(Path(data_dir) / "Metro_Interstate_Traffic_Volume.csv")
        holiday = (df["holiday"].values == None).astype(np.float32)  # noqa: E711 (matches upstream)
        temp = df["temp"].values.astype(np.float32)
        temp -= np.mean(temp)
        rain = df["rain_1h"].values.astype(np.float32)
        snow = df["snow_1h"].values.astype(np.float32)
        clouds = df["clouds_all"].values.astype(np.float32)
        stamps = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in df["date_time"].values]
        weekday = np.array([d.weekday() for d in stamps]).astype(np.float32)
        noon = np.sin(np.array([d.hour for d in stamps]).astype(np.float32) * np.pi / 24)
        features = np.stack([holiday, temp, rain, snow, clouds, weekday, noon], axis=-1).astype(np.float32)
        volume = df["traffic_volume"].values.astype(np.float32)
        volume = (volume - np.mean(volume)) / np.std(volume)
        all_x, all_y = cut_in_sequences(features, volume, self.cfg.seq_len, inc=4)
        tr_x, tr_y, va_x, va_y, te_x, te_y = _split_75_10_15(all_x, all_y, seed=23489)
        return self._dataset((tr_x, tr_y), (va_x, va_y), (te_x, te_y))


@TASKS.register("power")
class PowerTask(ClassicTask):
    """Household power: predict global active power from the other six channels."""

    def load(self, data_dir: Path) -> Dataset:
        rows, memory = [], [float(i) for i in range(7)]
        with open(Path(data_dir) / "household_power_consumption.txt") as f:
            for lineno, line in enumerate(f):
                if lineno == 0:
                    continue
                arr = line.split(";")
                if len(arr) < 8:
                    continue
                cols = arr[2:]
                for i, val in enumerate(cols):
                    val = val.strip()
                    if val in ("?", ""):
                        cols[i] = memory[i]       # carry the last valid value forward
                    else:
                        cols[i] = float(val)
                        memory[i] = cols[i]
                rows.append(np.array(cols, dtype=np.float32))
        all_x = np.stack(rows, axis=0)
        all_x = _zscore_columns(all_x, all_x.mean(axis=0), all_x.std(axis=0))
        all_y = all_x[:, 0].reshape(-1, 1).astype(np.float32)
        all_x = all_x[:, 1:]
        seq_len = self.cfg.seq_len
        all_x, all_y = cut_in_sequences(all_x, all_y, seq_len, inc=seq_len)
        tr_x, tr_y, va_x, va_y, te_x, te_y = _split_75_10_15(all_x, all_y, seed=23489)
        return self._dataset((tr_x, tr_y), (va_x, va_y), (te_x, te_y))


@TASKS.register("ozone")
class OzoneTask(ClassicTask):
    """Eight-hour ozone level detection."""

    def load(self, data_dir: Path) -> Dataset:
        xs, ys = [], []
        with open(Path(data_dir) / "eighthr.data") as f:
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) != 74:
                    continue
                ys.append(int(float(parts[-1])))
                xs.append(np.array([0.0 if p == "?" else float(p) for p in parts[1:-1]],
                                   dtype=np.float32))
        all_x = np.stack(xs, axis=0)
        all_y = np.array(ys, dtype=np.int64)
        all_x = (all_x - all_x.mean()) / all_x.std()
        all_x, all_y = cut_in_sequences(all_x, all_y, self.cfg.seq_len, inc=4)
        tr_x, tr_y, va_x, va_y, te_x, te_y = _split_75_10_15(all_x, all_y, seed=23489)
        return self._dataset((tr_x, tr_y), (va_x, va_y), (te_x, te_y))


@TASKS.register("person")
class PersonTask(ClassicTask):
    """Activity recognition from four body-worn accelerometer tags."""

    CLASSES = {
        "lying down": 0, "lying": 0, "sitting down": 1, "sitting": 1,
        "standing up from lying": 2, "standing up from sitting": 2,
        "standing up from sitting on the ground": 2, "walking": 3, "falling": 4,
        "on all fours": 5, "sitting on the ground": 6,
    }
    SENSORS = {"010-000-024-033": 0, "010-000-030-096": 1,
               "020-000-033-111": 2, "020-000-032-221": 3}

    def load(self, data_dir: Path) -> Dataset:
        seq_len = self.cfg.seq_len
        people_x, people_y, series_x, series_y = [], [], [], []
        current = "A01"
        with open(Path(data_dir) / "ConfLongDemo_JSI.txt") as f:
            for line in f:
                arr = line.split(",")
                if len(arr) < 6:
                    break
                if arr[0] != current:
                    if series_x:
                        people_x.append(np.stack(series_x, axis=0))
                        people_y.append(np.array(series_y, dtype=np.int64))
                    series_x, series_y = [], []
                current = arr[0]
                onehot = np.zeros(4, dtype=np.float32)
                onehot[self.SENSORS[arr[1]]] = 1.0
                series_x.append(np.concatenate([onehot, np.array(arr[4:7], dtype=np.float32)]))
                series_y.append(self.CLASSES[arr[7].replace("\n", "")])
        if series_x:
            people_x.append(np.stack(series_x, axis=0))
            people_y.append(np.array(series_y, dtype=np.int64))
        xs, ys = [], []
        for px, py in zip(people_x, people_y):
            for s in range(0, px.shape[0] - seq_len, seq_len // 2):
                xs.append(px[s:s + seq_len])
                ys.append(py[s:s + seq_len])
        all_x, all_y = np.stack(xs, axis=0), np.stack(ys, axis=0)
        tr_x, tr_y, va_x, va_y, te_x, te_y = _split_75_10_15(all_x, all_y, seed=27731)
        return self._dataset((tr_x, tr_y), (va_x, va_y), (te_x, te_y))
