import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 50, num_layers: int = 1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=x.device
        )

        c0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=x.device
        )

        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])

        return out