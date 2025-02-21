import torch
import torch.nn as nn

class DepthClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool3d(1)
        ])

        self.fc_layers = nn.ModuleList([
            nn.Linear(256, 128),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        ])

    def forward(self, x):
        """
        x : [b, c, t, h, w]
        """
        for layer in self.layers:
            x = torch.relu(layer(x)) if isinstance(layer, nn.Conv3d) else layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = torch.relu(layer(x)) if isinstance(layer, nn.Linear) else layer(x)
        return x