
import torch
import os

from torch import nn
from collections import OrderedDict

class ContextEncoder(nn.Module):
    def __init__(self, shape, n_hidden_dim=32, *args, **kwargs):
        super(ContextEncoder, self).__init__()

        self.shape = shape
        if isinstance(shape, (tuple, list)):
            self.shape = shape[0]
        assert self.shape % 16 == 0, "Image shape should be a multiple of 16."

        self.context_encoder = nn.Sequential(*[
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(4, 4),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            # nn.MaxPool2d(4, 4),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            # nn.Linear(16*(self.shape//16)*(self.shape//16), n_hidden_dim),
            nn.Linear(16, n_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        ])

    def forward(self, x):
        return self.context_encoder(x)

class LinearModel(nn.Module):
    def __init__(self, in_features, hidden_dim=32):
        super(LinearModel, self).__init__()

        # Defines necessary variable
        self.is_sampling = False
        self.sampling_cache = None

        # Feature extractor
        self.feature_extractor = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        ])

        # Linear layer
        self.linear = nn.Linear(hidden_dim, 1, bias=False)

    def sampling(self, is_sampling):
        """
        Activates/Deactivates the sampling mode of the model. This mode can
        be efficient during sampling when the current context doesn't change.
        """
        if not self.is_sampling or not is_sampling:
            self.sampling_cache = None
        self.is_sampling = is_sampling

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.linear(x)

    def load_pretrained(self, path="./pre-trained/model.pt"):
        """
        Loads a context encoder model
        """
        def gattr(obj, names):
            if len(names) == 1:
                return getattr(obj, names[0])
            else:
                return gattr(getattr(obj, names[0]), names[1:])

        def sattr(obj, names, val):
            if len(names) == 1:
                return setattr(obj, names[0], val)
            else:
                return sattr(getattr(obj, names[0]), names[1:], val)

        if not os.path.isfile(path):
            print("[!!!!] Model {} does not exist...".format(path))
            return

        model = torch.load(path, map_location=lambda storage, loc: storage)
        if not isinstance(model, (OrderedDict, dict)):
            model = model.model.state_dict()

        new_state_dict = {}
        for key, value in model.items():
            current = gattr(self, key.split("."))
            if current.shape == value.shape:
                new_state_dict[key] = value

        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)

class ContextLinearModel(LinearModel):
    def __init__(self, in_features, hidden_dim=32, every_step_decision=False):
        super(ContextLinearModel, self).__init__(in_features, hidden_dim=hidden_dim)

        self.every_step_decision = every_step_decision

    def forward(self, X, history):
        if X.dim() == 2:
            X = X.unsqueeze(1)
        # Extracts context from history.
        if self.every_step_decision:
            # We use the most recent context
            ctx = history["ctx"][-1].view(1, 1, -1)
        else:
            # We use the first context
            ctx = history["ctx"][0].view(1, 1, -1)

        # Repeat ctx in cases of sampling
        ctx = ctx.repeat(len(X), 1, 1)

        X = torch.cat((X, ctx), dim=-1)

        x = self.feature_extractor(X)
        return self.linear(x)

class ImageContextLinearModel(ContextLinearModel):
    def __init__(
        self, in_features, image_shape, hidden_dim=32, pretrained_opts=None,
        every_step_decision=False, full_gradient=False, *args, **kwargs
    ):
        super(ImageContextLinearModel, self).__init__(in_features=in_features)

        self.in_features = in_features
        self.image_shape = image_shape
        self.full_gradient = full_gradient
        if isinstance(image_shape, (tuple, list)):
            self.image_shape = image_shape[0]
        assert self.image_shape % 16 == 0, "Image shape should be a multiple of 16."
        self.pretrained_opts = pretrained_opts
        self.every_step_decision = every_step_decision

        # Feature extractor
        self.feature_extractor = nn.Sequential(*[
            nn.Linear(self.in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        ])

        # Context encoder
        self.context_encoder = ContextEncoder(self.image_shape, hidden_dim)

        # Linear layer
        self.pre = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.linear = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, X, history):
        if self.training or self.full_gradient:
            X = self.extract_features(X, history)
        else:
            with torch.no_grad():
                X = self.extract_features(X, history)

        return self.linear(X)

    def ctx_features(self, ctx):
        if self.is_sampling and isinstance(self.sampling_cache, torch.Tensor):
            return self.sampling_cache

        # Since context is constant we only calculate with a single ctx
        if ctx.dim() == 3:
            # Ensures ctx is 4 dimensions [batch, channels, height, width]
            ctx = ctx.unsqueeze(0)
        if self.every_step_decision:
            # We use the most recent context
            ctx = ctx[:, [-1]]
        else:
            # We use the first context
            ctx = ctx[:, [0]]

        if self.pretrained_opts.get("use", False):
            if self.pretrained_opts.get("update", True):
                ctx = self.context_encoder(ctx)
            else:
                with torch.no_grad():
                    ctx = self.context_encoder(ctx)
        else:
            ctx = self.context_encoder(ctx)

        # ctx should have [batch, ..., ...]
        ctx = ctx.unsqueeze(1)

        if self.is_sampling:
            self.sampling_cache = ctx

        return ctx

    def extract_features(self, X, history):
        if X.dim() == 2:
            X = X.unsqueeze(1)

        X = self.feature_extractor(X)

        # Extracts context from history.
        ctx = history["ctx"]
        ctx = self.ctx_features(ctx)

        # Repeat ctx in cases of sampling and concatenates with parameter features
        if len(X) != len(ctx):
            ctx = ctx.repeat(len(X), 1, 1)

        X = torch.cat((X, ctx), dim=-1)

        X = nn.functional.relu(self.pre(X))

        return X

class LSTMLinearModel(nn.Module):
    def __init__(self, in_features, hidden_dim=32):
        super(LSTMLinearModel, self).__init__()

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
