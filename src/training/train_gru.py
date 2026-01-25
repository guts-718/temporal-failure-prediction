import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.models.gru import GRUModel

#  CONFIG
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
input_size=19

DATA_DIR = Path("data/processed")

#  DATASET 
class FailureDataset(Dataset):
    def __init__(self, split):
        data = np.load(DATA_DIR / f"{split}_norm.npz")
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32)
        # self.timestamps = data["timestamps"]


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = FailureDataset("train")
val_ds = FailureDataset("val")
test_ds = FailureDataset("test")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

#  MODEL 
device = "cuda" if torch.cuda.is_available() else "cpu"

model = GRUModel(input_size).to(device)

pos = train_ds.y.sum()
neg = len(train_ds) - pos
pos_weight = neg / pos

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#  EVAL 
def evaluate(loader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu())
            labels.append(y.cpu())

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    bin_preds = (preds > 0.5).int()

    tp = ((bin_preds == 1) & (labels == 1)).sum().item()
    fp = ((bin_preds == 1) & (labels == 0)).sum().item()
    fn = ((bin_preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1

def early_warning_time(loader):
    model.eval()
    times = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu()

            for i in range(len(preds)):
                if preds[i] == 1 and y[i] == 1:
                    times.append(0)  # prediction point already inside window

    if len(times) == 0:
        return 0.0

    return sum(times) / len(times)


#  TRAIN 
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_p, val_r, val_f = evaluate(val_loader)

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {total_loss:.3f} | "
        f"Val P: {val_p:.3f} R: {val_r:.3f} F1: {val_f:.3f}"
    )

#  TEST
test_p, test_r, test_f = evaluate(test_loader)
print("\nGRU Test Results:")
print(f"Precision={test_p:.3f} Recall={test_r:.3f} F1={test_f:.3f}")
ew = early_warning_time(test_loader)
print(f"Avg Early Warning Time (mins): {ew:.2f}")
