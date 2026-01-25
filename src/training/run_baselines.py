import numpy as np
from pathlib import Path
from src.models.baseline import (
    rule_based_predict,
    snapshot_logreg,
    window_logreg,
    eval_preds
)

DATA_DIR = Path("data/processed")

def load(split):
    d = np.load(DATA_DIR / f"{split}_norm.npz", allow_pickle=True)
    return d["X"], d["y"]

X_train, y_train = load("train")
X_test, y_test = load("test")

# error_rate index from feature list
ERROR_RATE_IDX = 1

# Rule baseline
rule_preds = rule_based_predict(X_test, ERROR_RATE_IDX)
eval_preds(y_test, rule_preds, "Rule-based")

# Snapshot logistic regression
snap_preds = snapshot_logreg(X_train, y_train, X_test)
eval_preds(y_test, snap_preds, "Snapshot LogReg")

# Window-aggregated logistic regression
win_preds = window_logreg(X_train, y_train, X_test)
eval_preds(y_test, win_preds, "Window LogReg")

print("Positive rate:", y_test.mean())

