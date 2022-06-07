
import torch

from torch import nn

class LinearModel(nn.Module):
    def __init__(self, in_features, hidden_dim=32):
        super(LinearModel, self).__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ])

        # Linear layer
        self.linear = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.linear(x)
