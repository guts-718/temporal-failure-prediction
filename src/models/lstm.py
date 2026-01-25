import torch
import torch.nn as nn

class LSTMModel64(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        x = self.fc1(last)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze(1)



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        x = self.fc1(last)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze(1)
