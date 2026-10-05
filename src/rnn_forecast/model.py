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
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        encoded, _ = self.gru(x)
        return self.head(encoded[:, -1, :])




class ResidualLoadProfileGRU(LoadProfileGRU):
    """Forecast a correction to recent-day and same-weekday profiles."""

    def __init__(self, input_size=24, hidden_size=64, num_layers=2, dropout=0.2,
                 recent_weight=0.8, seasonal_lag=7):
        super().__init__(input_size, hidden_size, num_layers, dropout)
        
        if not 0 <= recent_weight <= 1 or seasonal_lag < 1:
            raise ValueError("recent_weight must be in [0, 1] and seasonal_lag must be positive")
        self.recent_weight = recent_weight
        self.seasonal_lag = seasonal_lag
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        if x.size(1) < self.seasonal_lag:
            raise ValueError("The input sequence is shorter than seasonal_lag")
        recent = x[:, -1, :]
        same_weekday = x[:, -self.seasonal_lag, :]
        baseline = self.recent_weight * recent + (1 - self.recent_weight) * same_weekday
        return baseline + super().forward(x)
