import numpy as np
import pandas as pd
from pathlib import Path

# Configs
INPUT_WINDOW = 10          # 10 minutes
PRED_HORIZON = 5           # 5 minutes

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RAW_DATA_DIR = Path("synthetic")
OUT_DIR = Path("processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "request_count",
    "error_rate",
    "p50_latency",
    "p95_latency",
    "p99_latency",
    "cpu_usage",
    "memory_usage",
    "network_in",
    "network_out",
    "pod_restarts",
]

# load the data
metrics = pd.read_csv(RAW_DATA_DIR / "metrics.csv", parse_dates=["timestamp"])
failures = pd.read_csv(RAW_DATA_DIR / "failures.csv", parse_dates=["timestamp"])

df = metrics.merge(failures, on="timestamp", how="inner")
df = df.sort_values("timestamp").reset_index(drop=True)

# building windows
X = []
y = []
timestamps = []

num_rows = len(df)

for t in range(INPUT_WINDOW - 1, num_rows - PRED_HORIZON):
    # input window [t-9 , t-8, ... t]
    window = df.loc[t - INPUT_WINDOW + 1 : t, FEATURE_COLUMNS].values

    # label = any failure occurance in [t+1 ... t+5]
    future_failures = df.loc[t + 1 : t + PRED_HORIZON, "failure_active"]
    label = int(future_failures.any())

    X.append(window)
    y.append(label)
    timestamps.append(df.loc[t, "timestamp"])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
timestamps = np.array(timestamps)

# Time based splitting
N = len(X)
train_end = int(N * TRAIN_RATIO)
val_end = int(N * (TRAIN_RATIO + VAL_RATIO))

X_train, y_train, ts_train = X[:train_end], y[:train_end], timestamps[:train_end]
X_val, y_val, ts_val = X[train_end:val_end], y[train_end:val_end], timestamps[train_end:val_end]
X_test, y_test, ts_test = X[val_end:], y[val_end:], timestamps[val_end:]


np.savez_compressed(
    OUT_DIR / "train.npz",
    X=X_train,
    y=y_train,
    timestamps=ts_train,
)

np.savez_compressed(
    OUT_DIR / "val.npz",
    X=X_val,
    y=y_val,
    timestamps=ts_val,
)

np.savez_compressed(
    OUT_DIR / "test.npz",
    X=X_test,
    y=y_test,
    timestamps=ts_test,
)

# logs
def log_split(name, y_split):
    pos = int(y_split.sum())
    total = len(y_split)
    print(f"{name}: samples={total}, positives={pos}, positive_ratio={pos/total:.4f}")

print("Dataset created successfully\n")
print(f"Input shape: {X.shape}")
print(f"Label shape: {y.shape}\n")

log_split("Train", y_train)
log_split("Val", y_val)
log_split("Test", y_test)

print("\nSaved to:", OUT_DIR.resolve())
