"""GRU architecture for next-day hourly load-profile forecasts."""

from torch import nn


class LoadProfileGRU(nn.Module):
    """Encode a sequence of daily profiles and forecast the next 24-hour profile."""

    def __init__(self, input_size=24, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, input_size))

    def forward(self, x):
        encoded, _ = self.gru(x)
        return self.head(encoded[:, -1, :])
