"""Small CNN matched to 24x24 load-profile images."""

from torch import nn


class SmallLoadProfileCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(64 * 3 * 3, 64), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
