import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

#  RULE BASELINE 
def rule_based_predict(X, error_idx, threshold=1.5):
    """
    X: (N, T, F) normalized or raw
    error_idx: index of error_rate feature
    threshold: threshold on mean error_rate
    """
    avg_error = X[:, -3:, error_idx].mean(axis=1)
    return (avg_error > threshold).astype(int)

#  SNAPSHOT LOGREG 
def snapshot_logreg(X_train, y_train, X_test):
    Xtr = X_train[:, -1, :]
    Xte = X_test[:, -1, :]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    return clf.predict(Xte)

#  WINDOW AGG LOGREG 
def aggregate_features(X):
    mean = X.mean(axis=1)
    max_ = X.max(axis=1)
    slope = X[:, -1, :] - X[:, 0, :]
    return np.concatenate([mean, max_, slope], axis=1)

def window_logreg(X_train, y_train, X_test):
    Xtr = aggregate_features(X_train)
    Xte = aggregate_features(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    return clf.predict(Xte)

# ---------------- METRICS ----------------
def eval_preds(y_true, y_pred, name):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    print(f"{name}: Precision={p:.3f} Recall={r:.3f} F1={f:.3f}")
