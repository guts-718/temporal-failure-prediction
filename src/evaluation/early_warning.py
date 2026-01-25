import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.models.lstm import LSTMModel

DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/synthetic")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD MODEL 
model = LSTMModel(input_size=10)
model.load_state_dict(torch.load("lstm.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

#  LOAD DATA 
test = np.load(DATA_DIR / "test_norm.npz", allow_pickle=True)
X_test = torch.tensor(test["X"], dtype=torch.float32)
timestamps = test["timestamps"]

failures = pd.read_csv(RAW_DIR / "failures.csv", parse_dates=["timestamp"])
failure_series = dict(
    zip(failures["timestamp"], failures["failure_active"])
)

# PREDICT 
with torch.no_grad():
    probs = torch.sigmoid(model(X_test.to(DEVICE))).cpu().numpy()

preds = (probs > 0.5).astype(int)

#  EARLY WARNING 
warning_times = []

for i in range(len(preds)):
    if preds[i] == 1:
        t = timestamps[i]

        for k in range(1, 6):   # next 5 minutes
            future_t = t + np.timedelta64(k, "m")
            if failure_series.get(pd.Timestamp(future_t), 0) == 1:
                warning_times.append(k)
                break

if len(warning_times) == 0:
    print("No early warnings found.")
else:
    print("Average early warning (minutes):",
          sum(warning_times) / len(warning_times))
    print("Median early warning (minutes):",
          sorted(warning_times)[len(warning_times)//2])
