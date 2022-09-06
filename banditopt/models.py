
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

class LinearLSTMModel(nn.Module):
    def __init__(self, in_features, hidden_dim=32):
        super(LinearLSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        context_features = 1
        objective_features = 1

        # Feature extractor
        self.feature_extractor = nn.Sequential(*[
            nn.Linear(in_features + context_features + objective_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ])

        # LSTM layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Linear layer
        self.linear = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, X, history):

        if X.dim() == 2:
            X = X.unsqueeze(1)

        # Creates an empty hidden state
        hidden = (torch.zeros((1, len(X), self.hidden_dim)), torch.zeros((1, len(X), self.hidden_dim)))
        if next(self.parameters()).is_cuda:
            hidden = tuple(h.cuda() for h in hidden)

        # Calculates the current hidden state from history
        if len(history["X"]) > 0:
            hidden = (torch.zeros((1, 1, self.hidden_dim)), torch.zeros((1, 1, self.hidden_dim)))
            if next(self.parameters()).is_cuda:
                hidden = tuple(h.cuda() for h in hidden)

            for _X, _y, _ctx in zip(history["X"], history["y"], history["ctx"]):
                _X = _X.view(1, 1, -1)
                _y = _y.view(1, 1, -1)
                _ctx = _ctx.view(1, 1, -1)
                _X = torch.cat((_X, _ctx, _y), dim=-1)

                x = self.feature_extractor(_X)
                x, hidden = self.lstm(x, hidden)

            hidden = tuple(h.repeat(1, len(X), 1) for h in hidden)
            ctx = _ctx.repeat(len(X), 1, 1)
            y = _y.repeat(len(X), 1, 1)
        else:
            ctx = torch.zeros((len(X), 1, 1))
            y = torch.zeros((len(X), 1, 1))
            if next(self.parameters()).is_cuda:
                ctx = ctx.cuda()
                y = y.cuda()

        X = torch.cat((X, ctx, y), dim=-1)

        # Calculates
        x = self.feature_extractor(X)
        x, hidden = self.lstm(x, hidden)
        x = self.linear(x)
        return x
