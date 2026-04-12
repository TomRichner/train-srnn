"""Unified dataset loaders for all 9 tasks.

Each loader returns a dict:
    {"train": (x, y), "valid": (x, y), "test": (x, y), "meta": {...}}
where x is numpy array (num_sequences, seq_len, features) and y is numpy
array of labels (num_sequences, seq_len) for per-timestep tasks or
(num_sequences,) for single-label tasks.

Meta includes: input_size, output_size, task_type, seq_len.
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

def cut_in_sequences(data, labels, seq_len, inc=1):
    """Cut time series into overlapping windows.

    Args:
        data: (N, features) array.
        labels: (N,) or (N, label_dim) array.
        seq_len: Window length.
        inc: Stride between windows.

    Returns:
        sequences: (num_windows, seq_len, features)
        seq_labels: (num_windows, seq_len) or (num_windows, seq_len, label_dim)
    """
    sequences = []
    seq_labels = []
    for s in range(0, len(data) - seq_len, inc):
        sequences.append(data[s:s + seq_len])
        seq_labels.append(labels[s:s + seq_len])
    return np.array(sequences), np.array(seq_labels)


def _split_75_10_15(x, y, seed=23489):
    """Split data into 75% train, 10% valid, 15% test."""
    n = x.shape[0]
    perm = np.random.RandomState(seed).permutation(n)
    valid_size = int(0.1 * n)
    test_size = int(0.15 * n)

    valid_x = x[perm[:valid_size]]
    valid_y = y[perm[:valid_size]]
    test_x = x[perm[valid_size:valid_size + test_size]]
    test_y = y[perm[valid_size:valid_size + test_size]]
    train_x = x[perm[valid_size + test_size:]]
    train_y = y[perm[valid_size + test_size:]]

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def _split_90_10(x, y, seed=893429):
    """Split data into 90% train, 10% valid."""
    n = x.shape[0]
    perm = np.random.RandomState(seed).permutation(n)
    valid_size = int(0.1 * n)

    valid_x = x[perm[:valid_size]]
    valid_y = y[perm[:valid_size]]
    train_x = x[perm[valid_size:]]
    train_y = y[perm[valid_size:]]

    return train_x, train_y, valid_x, valid_y


# ---------------------------------------------------------------------------
# 1. SMNIST (Sequential MNIST) - row-by-row
# ---------------------------------------------------------------------------

def load_smnist(data_dir=None):
    """Load Sequential MNIST (row-by-row, seq_len=28, features=28).

    Uses torchvision to download/load MNIST.  Labels are single per sequence.
    """
    from torchvision import datasets

    cache = os.path.join(data_dir, "mnist_cache") if data_dir else "/tmp/mnist_cache"
    train_ds = datasets.MNIST(cache, train=True, download=True)
    test_ds = datasets.MNIST(cache, train=False, download=True)

    train_x = train_ds.data.numpy().astype(np.float32) / 255.0  # (60000, 28, 28)
    train_y = train_ds.targets.numpy().astype(np.int64)
    test_x = test_ds.data.numpy().astype(np.float32) / 255.0
    test_y = test_ds.targets.numpy().astype(np.int64)

    # 90/10 split of training set
    split = int(0.9 * len(train_x))
    valid_x, valid_y = train_x[split:], train_y[split:]
    train_x, train_y = train_x[:split], train_y[:split]

    # Shape: (num_sequences, 28, 28) — already row-by-row
    return {
        "train": (train_x, train_y),
        "valid": (valid_x, valid_y),
        "test": (test_x, test_y),
        "meta": {
            "input_size": 28,
            "output_size": 10,
            "task_type": "classification",
            "seq_len": 28,
            "per_timestep_labels": False,
        },
    }


# ---------------------------------------------------------------------------
# 2. HAR (Human Activity Recognition)
# ---------------------------------------------------------------------------

def load_har(data_dir="data/har"):
    """Load UCI HAR Dataset.

    Expects files at:
        data_dir/UCI HAR Dataset/train/X_train.txt
        data_dir/UCI HAR Dataset/train/y_train.txt
        data_dir/UCI HAR Dataset/test/X_test.txt
        data_dir/UCI HAR Dataset/test/y_test.txt
    """
    seq_len = 16
    base = os.path.join(data_dir, "UCI HAR Dataset")

    train_x = np.loadtxt(os.path.join(base, "train", "X_train.txt")).astype(np.float32)
    train_y = (np.loadtxt(os.path.join(base, "train", "y_train.txt")) - 1).astype(np.int64)
    test_x = np.loadtxt(os.path.join(base, "test", "X_test.txt")).astype(np.float32)
    test_y = (np.loadtxt(os.path.join(base, "test", "y_test.txt")) - 1).astype(np.int64)

    # Cut into 16-timestep windows
    train_x, train_y = cut_in_sequences(train_x, train_y, seq_len, inc=1)
    test_x, test_y = cut_in_sequences(test_x, test_y, seq_len, inc=8)

    # 90/10 valid from training
    train_x, train_y, valid_x, valid_y = _split_90_10(
        train_x, train_y, seed=893429)

    return {
        "train": (train_x, train_y),
        "valid": (valid_x, valid_y),
        "test": (test_x, test_y),
        "meta": {
            "input_size": 561,
            "output_size": 6,
            "task_type": "classification",
            "seq_len": seq_len,
            "per_timestep_labels": True,
        },
    }


# ---------------------------------------------------------------------------
# 3. Gesture
# ---------------------------------------------------------------------------

def _load_gesture_trace(filename):
    """Load a single gesture CSV file."""
    df = pd.read_csv(filename, header=0)
    str_y = df["Phase"].values
    convert = {"D": 0, "P": 1, "S": 2, "H": 3, "R": 4}
    y = np.array([convert[s] for s in str_y], dtype=np.int64)
    x = df.values[:, :-1].astype(np.float32)
    return x, y


def _cut_gesture_sequences(x, y, seq_len, interleaved=False):
    """Cut gesture trace into non-overlapping (+ optional interleaved) windows."""
    sequences = []
    num_sequences = x.shape[0] // seq_len
    for s in range(num_sequences):
        start = seq_len * s
        end = start + seq_len
        sequences.append((x[start:end], y[start:end]))
        if interleaved and s < num_sequences - 1:
            start2 = start + seq_len // 2
            end2 = start2 + seq_len
            sequences.append((x[start2:end2], y[start2:end2]))
    return sequences


def load_gesture(data_dir="data/gesture"):
    """Load gesture dataset from CSV files."""
    seq_len = 32

    training_files = [
        "a3_va3.csv", "b1_va3.csv", "b3_va3.csv",
        "c1_va3.csv", "c3_va3.csv", "a2_va3.csv", "a1_va3.csv",
    ]

    all_traces = []
    for f in training_files:
        x, y = _load_gesture_trace(os.path.join(data_dir, f))
        all_traces.extend(_cut_gesture_sequences(x, y, seq_len, interleaved=True))

    all_x = np.stack([t[0] for t in all_traces], axis=0)  # (N, seq_len, 32)
    all_y = np.stack([t[1] for t in all_traces], axis=0)  # (N, seq_len)

    # Normalize using stats from all data (before splitting)
    flat_x = all_x.reshape(-1, all_x.shape[-1])
    mean_x = np.mean(flat_x, axis=0)
    std_x = np.std(flat_x, axis=0)
    std_x[std_x < 1e-8] = 1.0
    all_x = (all_x - mean_x) / std_x

    # Split: 10% valid, 15% test, 75% train
    train_x, train_y, valid_x, valid_y, test_x, test_y = _split_75_10_15(
        all_x, all_y, seed=23489)

    return {
        "train": (train_x, train_y),
        "valid": (valid_x, valid_y),
        "test": (test_x, test_y),
        "meta": {
            "input_size": 32,
            "output_size": 5,
            "task_type": "classification",
            "seq_len": seq_len,
            "per_timestep_labels": True,
        },
    }


# ---------------------------------------------------------------------------
# 4. Occupancy
# ---------------------------------------------------------------------------

def _read_occupancy_file(filename):
    """Read an occupancy CSV file."""
    df = pd.read_csv(filename)
    data_x = np.stack([
        df["Temperature"].values,
        df["Humidity"].values,
        df["Light"].values,
        df["CO2"].values,
        df["HumidityRatio"].values,
    ], axis=-1).astype(np.float32)
    data_y = df["Occupancy"].values.astype(np.int64)
    return data_x, data_y


def load_occupancy(data_dir="data/occupancy"):
    """Load occupancy detection dataset."""
    seq_len = 16

    train_x, train_y = _read_occupancy_file(
        os.path.join(data_dir, "datatraining.txt"))
    test0_x, test0_y = _read_occupancy_file(
        os.path.join(data_dir, "datatest.txt"))
    test1_x, test1_y = _read_occupancy_file(
        os.path.join(data_dir, "datatest2.txt"))

    # Normalize using training stats
    mean_x = np.mean(train_x, axis=0)
    std_x = np.std(train_x, axis=0)
    std_x[std_x < 1e-8] = 1.0
    train_x = (train_x - mean_x) / std_x
    test0_x = (test0_x - mean_x) / std_x
    test1_x = (test1_x - mean_x) / std_x

    # Cut into windows
    train_x, train_y = cut_in_sequences(train_x, train_y, seq_len, inc=1)
    test0_x, test0_y = cut_in_sequences(test0_x, test0_y, seq_len, inc=8)
    test1_x, test1_y = cut_in_sequences(test1_x, test1_y, seq_len, inc=8)

    # 90/10 valid from training
    train_x, train_y, valid_x, valid_y = _split_90_10(
        train_x, train_y, seed=893429)

    # Combine both test sets
    test_x = np.concatenate([test0_x, test1_x], axis=0)
    test_y = np.concatenate([test0_y, test1_y], axis=0)

    return {
        "train": (train_x, train_y),
        "valid": (valid_x, valid_y),
        "test": (test_x, test_y),
        "meta": {
            "input_size": 5,
            "output_size": 2,
            "task_type": "classification",
            "seq_len": seq_len,
            "per_timestep_labels": True,
        },
    }


# ---------------------------------------------------------------------------
# 5. Traffic
# ---------------------------------------------------------------------------

def load_traffic(data_dir="data/traffic"):
    """Load Metro Interstate Traffic Volume dataset."""
    import datetime as dt

    seq_len = 32
    df = pd.read_csv(os.path.join(data_dir, "Metro_Interstate_Traffic_Volume.csv"))

    holiday = (df["holiday"].values == None).astype(np.float32)
    temp = df["temp"].values.astype(np.float32)
    temp -= np.mean(temp)
    rain = df["rain_1h"].values.astype(np.float32)
    snow = df["snow_1h"].values.astype(np.float32)
    clouds = df["clouds_all"].values.astype(np.float32)

    date_time = df["date_time"].values
    date_time = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date_time]
    weekday = np.array([d.weekday() for d in date_time]).astype(np.float32)
    noon = np.array([d.hour for d in date_time]).astype(np.float32)
    noon = np.sin(noon * np.pi / 24)

    features = np.stack([holiday, temp, rain, snow, clouds, weekday, noon],
                        axis=-1).astype(np.float32)

    traffic_volume = df["traffic_volume"].values.astype(np.float32)
    traffic_volume -= np.mean(traffic_volume)
    traffic_volume /= np.std(traffic_volume)

    # Cut into windows
    all_x, all_y = cut_in_sequences(features, traffic_volume, seq_len, inc=4)

    # Split
    train_x, train_y, valid_x, valid_y, test_x, test_y = _split_75_10_15(
        all_x, all_y, seed=23489)

    return {
        "train": (train_x, train_y),
        "valid": (valid_x, valid_y),
        "test": (test_x, test_y),
        "meta": {
            "input_size": 7,
            "output_size": 1,
            "task_type": "regression",
            "seq_len": seq_len,
            "per_timestep_labels": True,
        },
    }


# ---------------------------------------------------------------------------
# 6. Power
# ---------------------------------------------------------------------------

def load_power(data_dir="data/power"):
    """Load household power consumption dataset."""
    seq_len = 32

    all_x = []
    memory = [float(i) for i in range(7)]
    filepath = os.path.join(data_dir, "household_power_consumption.txt")

    with open(filepath, "r") as f:
        for lineno, line in enumerate(f):
            if lineno == 0:
                continue
            arr = line.split(";")
            if len(arr) < 8:
                continue
            feature_col = arr[2:]
            for i in range(len(feature_col)):
                val = feature_col[i].strip()
                if val == "?" or val == "":
                    feature_col[i] = memory[i]
                else:
                    feature_col[i] = float(val)
                    memory[i] = feature_col[i]
            all_x.append(np.array(feature_col, dtype=np.float32))

    all_x = np.stack(all_x, axis=0)

    # Normalize per-feature
    all_x -= np.mean(all_x, axis=0)
    std = np.std(all_x, axis=0)
    std[std < 1e-8] = 1.0
    all_x /= std

    # Target: Global_active_power (first column after date/time = index 0)
    all_y = all_x[:, 0].reshape(-1, 1).astype(np.float32)
    all_x = all_x[:, 1:]  # 6 remaining features

    # Cut with non-overlapping windows
    all_x, all_y = cut_in_sequences(all_x, all_y, seq_len, inc=seq_len)

    # Split
    train_x, train_y, valid_x, valid_y, test_x, test_y = _split_75_10_15(
        all_x, all_y, seed=23489)

    return {
        "train": (train_x, train_y),
        "valid": (valid_x, valid_y),
        "test": (test_x, test_y),
        "meta": {
            "input_size": 6,
            "output_size": 1,
            "task_type": "regression",
            "seq_len": seq_len,
            "per_timestep_labels": True,
        },
    }


# ---------------------------------------------------------------------------
# 7. Ozone
# ---------------------------------------------------------------------------

def load_ozone(data_dir="data/ozone"):
    """Load ozone level detection dataset."""
    seq_len = 32

    all_x = []
    all_y = []

    with open(os.path.join(data_dir, "eighthr.data"), "r") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split(",")
            if len(parts) != 74:
                continue
            label = int(float(parts[-1]))
            feats = []
            for i in range(1, len(parts) - 1):
                feats.append(0.0 if parts[i] == "?" else float(parts[i]))
            all_x.append(np.array(feats, dtype=np.float32))
            all_y.append(label)

    all_x = np.stack(all_x, axis=0)
    all_y = np.array(all_y, dtype=np.int64)

    # Normalize globally
    all_x -= np.mean(all_x)
    all_x /= np.std(all_x)

    # Cut into windows
    all_x, all_y = cut_in_sequences(all_x, all_y, seq_len, inc=4)

    # Split
    train_x, train_y, valid_x, valid_y, test_x, test_y = _split_75_10_15(
        all_x, all_y, seed=23489)

    return {
        "train": (train_x, train_y),
        "valid": (valid_x, valid_y),
        "test": (test_x, test_y),
        "meta": {
            "input_size": 72,
            "output_size": 2,
            "task_type": "classification",
            "seq_len": seq_len,
            "per_timestep_labels": True,
        },
    }


# ---------------------------------------------------------------------------
# 8. Person (Activity Recognition from Accelerometers)
# ---------------------------------------------------------------------------

_PERSON_CLASS_MAP = {
    "lying down": 0,
    "lying": 0,
    "sitting down": 1,
    "sitting": 1,
    "standing up from lying": 2,
    "standing up from sitting": 2,
    "standing up from sitting on the ground": 2,
    "walking": 3,
    "falling": 4,
    "on all fours": 5,
    "sitting on the ground": 6,
}

_PERSON_SENSOR_IDS = {
    "010-000-024-033": 0,
    "010-000-030-096": 1,
    "020-000-033-111": 2,
    "020-000-032-221": 3,
}


def load_person(data_dir="data/person"):
    """Load person activity recognition dataset."""
    seq_len = 32

    all_x = []  # list of per-person arrays
    all_y = []

    series_x = []
    series_y = []

    filepath = os.path.join(data_dir, "ConfLongDemo_JSI.txt")
    current_person = "A01"

    with open(filepath, "r") as f:
        for line in f:
            arr = line.split(",")
            if len(arr) < 6:
                break
            if arr[0] != current_person:
                # Save previous person's data
                if series_x:
                    all_x.append(np.stack(series_x, axis=0))
                    all_y.append(np.array(series_y, dtype=np.int64))
                series_x = []
                series_y = []
            current_person = arr[0]

            sensor_id = _PERSON_SENSOR_IDS[arr[1]]
            label = _PERSON_CLASS_MAP[arr[7].replace("\n", "")]
            accel = np.array(arr[4:7], dtype=np.float32)

            sensor_onehot = np.zeros(4, dtype=np.float32)
            sensor_onehot[sensor_id] = 1.0

            feature = np.concatenate([sensor_onehot, accel])
            series_x.append(feature)
            series_y.append(label)

    # Don't forget the last person
    if series_x:
        all_x.append(np.stack(series_x, axis=0))
        all_y.append(np.array(series_y, dtype=np.int64))

    # Cut each person's data into windows with 50% overlap
    inc = seq_len // 2
    sequences_x = []
    sequences_y = []
    for px, py in zip(all_x, all_y):
        for s in range(0, px.shape[0] - seq_len, inc):
            sequences_x.append(px[s:s + seq_len])
            sequences_y.append(py[s:s + seq_len])

    all_seq_x = np.stack(sequences_x, axis=0)  # (N, seq_len, 7)
    all_seq_y = np.stack(sequences_y, axis=0)  # (N, seq_len)

    # Split
    train_x, train_y, valid_x, valid_y, test_x, test_y = _split_75_10_15(
        all_seq_x, all_seq_y, seed=27731)

    return {
        "train": (train_x, train_y),
        "valid": (valid_x, valid_y),
        "test": (test_x, test_y),
        "meta": {
            "input_size": 7,
            "output_size": 7,
            "task_type": "classification",
            "seq_len": seq_len,
            "per_timestep_labels": True,
        },
    }


# ---------------------------------------------------------------------------
# 9. Cheetah (Half-Cheetah motion capture autoregressive)
# ---------------------------------------------------------------------------

def load_cheetah(data_dir="data/cheetah"):
    """Load half-cheetah motion capture dataset.

    Autoregressive: input = frame[t], target = frame[t+1].
    """
    seq_len = 32
    inc = 10

    all_files = sorted([
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir) if d.endswith(".npy")
    ])

    train_files = all_files[15:25]
    test_files = all_files[5:15]
    valid_files = all_files[:5]

    def _load_files(files):
        xs, ys = [], []
        for f in files:
            arr = np.load(f).astype(np.float32)
            for s in range(0, arr.shape[0] - seq_len - 1, inc):
                xs.append(arr[s:s + seq_len])
                ys.append(arr[s + 1:s + seq_len + 1])
        return np.stack(xs, axis=0), np.stack(ys, axis=0)

    train_x, train_y = _load_files(train_files)
    test_x, test_y = _load_files(test_files)
    valid_x, valid_y = _load_files(valid_files)

    return {
        "train": (train_x, train_y),
        "valid": (valid_x, valid_y),
        "test": (test_x, test_y),
        "meta": {
            "input_size": 17,
            "output_size": 17,
            "task_type": "regression",
            "seq_len": seq_len,
            "per_timestep_labels": True,
        },
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_LOADERS = {
    "smnist": load_smnist,
    "har": load_har,
    "gesture": load_gesture,
    "occupancy": load_occupancy,
    "traffic": load_traffic,
    "power": load_power,
    "ozone": load_ozone,
    "person": load_person,
    "cheetah": load_cheetah,
}


def load_dataset(task_name, data_dir=None):
    """Load a dataset by name.

    Args:
        task_name: One of 'smnist', 'har', 'gesture', 'occupancy',
            'traffic', 'power', 'ozone', 'person', 'cheetah'.
        data_dir: Optional base data directory. If None, uses the default
            path for each loader (e.g. 'data/har').

    Returns:
        Dict with keys 'train', 'valid', 'test', 'meta'.
    """
    if task_name not in _LOADERS:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list(_LOADERS.keys())}")
    loader = _LOADERS[task_name]
    if data_dir is not None:
        return loader(data_dir=data_dir)
    return loader()
