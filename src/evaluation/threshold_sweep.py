import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.models.lstm import LSTMModel

DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/synthetic")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#  LOAD MODEL 
model = LSTMModel(input_size=10)
model.load_state_dict(torch.load("lstm.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

#  LOAD DATA
test = np.load(DATA_DIR / "test_norm.npz", allow_pickle=True)
X = torch.tensor(test["X"], dtype=torch.float32)
y = test["y"]
timestamps = test["timestamps"]

failures = pd.read_csv(RAW_DIR / "failures.csv", parse_dates=["timestamp"])
failure_map = dict(zip(failures["timestamp"], failures["failure_active"]))

# PREDICT PROBS
with torch.no_grad():
    probs = torch.sigmoid(model(X.to(DEVICE))).cpu().numpy()

#  METRICS 
def eval_threshold(th):
    preds = (probs > th).astype(int)

    tp = ((preds == 1) & (y == 1)).sum()
    fp = ((preds == 1) & (y == 0)).sum()
    fn = ((preds == 0) & (y == 1)).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    warnings = []
    for i in range(len(preds)):
        if preds[i] == 1:
            t = timestamps[i]
            for k in range(1, 6):
                ft = t + np.timedelta64(k, "m")
                if failure_map.get(pd.Timestamp(ft), 0) == 1:
                    warnings.append(k)
                    break

    ew = sum(warnings)/len(warnings) if warnings else 0

    return precision, recall, f1, ew

#  SWEEP
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

print("th | precision | recall | f1 | early_warn")
print("-------------------------------------------")

for th in thresholds:
    p, r, f, ew = eval_threshold(th)
    print(f"{th:.2f} | {p:.3f} | {r:.3f} | {f:.3f} | {ew:.2f}")
