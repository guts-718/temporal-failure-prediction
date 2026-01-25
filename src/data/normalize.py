import numpy as np
from pathlib import Path

DATA_DIR = Path("./data/processed")
OUT_DIR = DATA_DIR

FEATURE_AXIS = (0, 1)  # samples & time

def load(split):
    data = np.load(DATA_DIR / f"{split}.npz", allow_pickle=True)
    return data["X"], data["y"], data["timestamps"]

# load splits
X_train, y_train, ts_train = load("train")
X_val, y_val, ts_val = load("val")
X_test, y_test, ts_test = load("test")

# compute train-only stats
mean = X_train.mean(axis=FEATURE_AXIS, keepdims=True)
std = X_train.std(axis=FEATURE_AXIS, keepdims=True) + 1e-6

def normalize(X):
    return (X - mean) / std

X_train_n = normalize(X_train)
X_val_n = normalize(X_val)
X_test_n = normalize(X_test)

# save normalized datasets + stats
np.savez_compressed(
    OUT_DIR / "train_norm.npz",
    X=X_train_n, y=y_train, timestamps=ts_train
)
np.savez_compressed(
    OUT_DIR / "val_norm.npz",
    X=X_val_n, y=y_val, timestamps=ts_val
)
np.savez_compressed(
    OUT_DIR / "test_norm.npz",
    X=X_test_n, y=y_test, timestamps=ts_test
)

np.savez_compressed(
    OUT_DIR / "norm_stats.npz",
    mean=mean,
    std=std
)

print("Normalization complete")
print("Train mean shape:", mean.shape)
print("Train std shape:", std.shape)
