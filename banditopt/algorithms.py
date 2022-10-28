#This file is largely based on:
#https://github.com/PDKlab/STED-Optimization/blob/master/src/algorithms.py
"""This module contains classes to generate options presented to the user in the online
multi-objective bandits optimization problem.
"""

import numpy
import torch
import os
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures
import sklearn
from sklearn.linear_model import BayesianRidge
import numpy as np
from torch import nn, optim
from tqdm.auto import trange, tqdm
from collections import defaultdict

from .models import LinearModel, LSTMLinearModel, ContextLinearModel, ImageContextLinearModel

import copy

from sklearn.preprocessing import StandardScaler

from inspect import currentframe, getframeinfo

def to_cuda(elements):
    """
    Recursively sends data to GPU
    """
    if isinstance(elements, (tuple)):
        elements = list(elements)
    if isinstance(elements, (list)):
        for i, element in enumerate(elements):
            elements[i] = to_cuda(element)
    elif isinstance(elements, dict):
        for key, value in elements.items():
            elements[key] = to_cuda(value)
    else:
        if isinstance(elements, numpy.ndarray):
            elements = torch.tensor(elements, dtype=torch.float32)
        if isinstance(elements, torch.Tensor):
            elements = elements.cuda()
            return elements
    return elements

class Scaler:
    def __init__(self, _min, _max):
        if isinstance(_min, type(None)):
            self._min = 0.
            self._max = 1.
        else:
            self._min = numpy.array(_min)[:, numpy.newaxis]
            self._max = numpy.array(_max)[:, numpy.newaxis]

    def fit_transform(self, X):
        return self.transform(X)

    def transform(self, X):
        return (X - self._min) / (self._max - self._min)

    def inverse_transform(self, X, std=False):
        if std:
            return X * (self._max - self._min)
        else:
            return X * (self._max - self._min) + self._min

class MO_function_sample():
    """
    This class creates a function sample with randomly generated random seeds

    algos: list of TS_sampler objects
    with_time: Optimize the (dwell)time also
    param_names: list of parameters to optimize

    individual: an array like object with parameter values
    """
    def __init__(self, algos, with_time, param_names, time_limit=None, borders=None, *args, **kwargs):
        self.seeds = [np.random.randint(2**31) for i in range(len(algos))]
        self.algos = algos
        self.with_time = with_time
        self.time_limit = time_limit
        self.borders = borders
        self.param_names = param_names

        self.history = kwargs.get("history", {})

    def evaluate(self, individuals, params_to_round=[], weights=None):
        X = numpy.array(individuals)
        for param in params_to_round:
            X[:, self.param_names.index(param)] = numpy.round(X[:, self.param_names.index(param)])

        ys = numpy.array([self.algos[i].sample(X, seed=self.seeds[i], history=self.history) for i in range(len(self.algos))]).squeeze(axis=-1)
        # ys = numpy.array([self.algos[i].predict(X)[0] for i in range(len(self.algos))]).squeeze(axis=-1)
        if self.time_limit is not None:
            pixeltimes = X[:, self.param_names.index("dwelltime")] * X[:, self.param_names.index("line_step")] * X[:, self.param_names.index("pixelsize")]**2/(20e-9)**2
            for i, bounds in enumerate(self.borders):
                if weights[i] < 0:
                    ys[i, :][pixeltimes > self.time_limit] = bounds[1] + (bounds[1]-bounds[0])*1.5
                elif weights[i] > 0:
                    ys[i, :][pixeltimes > self.time_limit] = bounds[0] - (bounds[1]-bounds[0])*1.5

        if self.with_time:
            dwelltime_pos = list(X.flatten()).index('dwelltime')
            return tuple(ys + X[dwelltime_pos])
        else:
            return list(map(tuple, ys.T))

def rescale_X(X, param_space_bounds):
    param_space_bounds = numpy.array(param_space_bounds).T
    xmin, xmax = param_space_bounds
    xmean = (xmax + xmin) / 2
    if X.ndim != 2:
        return []
    X = (X - xmean[numpy.newaxis]) / (0.5 * (xmax - xmin)[numpy.newaxis])
    return X
    # X = copy.deepcopy(X)
    # for col in range(X.shape[1]):
    #     xmin, xmax = param_space_bounds[col]
    #     xmean = (xmax+xmin)/2
    #     X[:,col] = (X[:,col] - xmean)/(0.5 * (xmax - xmin))
    # return X

class sklearn_GP(GaussianProcessRegressor):
    """This class is meant to be used as a the regressor argument of the TS_sampler
    class. It uses a scikit-learn implementation of Gaussian process regression. The
    regularization is fixed (no adaptative regularization).
    """
    def __init__(self, bandwidth, s_lb, s_ub, param_space_bounds=None, normalize_y=False, **kwargs):

        self.bandwidth = bandwidth
        self.s_lb = s_lb
        self.s_ub = s_ub
        self.norm_bound = 5.0
        self.lambda_ = s_ub ** 2 / self.norm_bound ** 2

        super().__init__(RBF(length_scale=self.bandwidth), alpha=self.lambda_, optimizer=None, normalize_y=normalize_y)

        self.param_space_bounds = param_space_bounds
        self.scaler = StandardScaler(with_mean=True, with_std=True)

    def update(self, X, y, *args, **kwargs):
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
        if y.ndim == 1:
            y = y[:, numpy.newaxis]
        y = self.scaler.fit_transform(y)

        self.fit(X,y)

    def get_mean_std(self, X):
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
        mean, sqrt_k = self.predict(X, return_std=True)
        std = self.s_ub / numpy.sqrt(self.lambda_) * sqrt_k

        # Rescales sampled mean
        if mean.ndim == 1:
            mean = mean[:, numpy.newaxis]
        mean = self.scaler.inverse_transform(mean)
        std = std * self.scaler.scale_
        return mean, std

    def sample(self, X, seed=None):
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)

        mean, k = self.predict(X, return_cov=True)
        cov = self.s_ub ** 2 / self.lambda_ * k
        # Rescales sampled mean
        rng = numpy.random.default_rng(seed)
        f_tilde = rng.multivariate_normal(mean.flatten(), cov, method='eigh')[:,numpy.newaxis]
        f_tilde = self.scaler.inverse_transform(f_tilde)
        return f_tilde

class sklearn_BayesRidge(BayesianRidge):
    """This class is meant to be used as a the regressor argument of the TS_sampler
    class. It uses a scikit-learn implementation of bayesian linear regression to fit a
    polynomial of a certain degree. fit_intercept=False should be used.

    :param degree: degree of the polynomial
    :param other parameters: see sklearn.linear_model.BayesianRidge documentation
    """
    def __init__(self, degree, param_space_bounds=None,
                 tol=1e-6, fit_intercept=False,
                 compute_score=True,alpha_init=None,
                 lambda_init=None,
                 alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06,
                 N0_w=None, std0_w=None, N0_n=None, std0_n=None,
                 **kwargs):
        if (N0_w is not None) or (std0_w is not None) or (N0_n is not None) or (std0_n is not None):
            lambda_1 = N0_w/2
            lambda_2 = lambda_1*std0_w**2
            alpha_1 = N0_n/2
            alpha_2 = alpha_1*std0_n**2
        super().__init__(tol=tol, fit_intercept=fit_intercept,
                         compute_score=compute_score, alpha_init=alpha_init,
                         lambda_init=lambda_init,
                        alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2)
        self.degree=degree
        self.param_space_bounds=param_space_bounds

        self.scaler = StandardScaler(with_mean=True, with_std=True)

    def update(self, X, y, *args, **kwargs):
        """Update the regression model using the observations *y* acquired at
        locations *X*.

        :param action: A 2d array of locations.
        :param reward: A 1-D array of observations.
        """
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
        if self.fit_intercept:
            X = PolynomialFeatures(self.degree).fit_transform(X)[:,1:]
        else:
            X = PolynomialFeatures(self.degree).fit_transform(X)
            if y.ndim == 1:
                y = y[:, numpy.newaxis]
            y = self.scaler.fit_transform(y)

        self.fit(X,y.flatten())

    def get_mean_std(self, X, return_withnoise=False):
        """Predict mean and standard deviation at given points *X*.

        :param : A 2d array of locations at which to predict.
        :returns: An array of means and an array of standard deviations.
        """
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
        if self.fit_intercept:
            X = PolynomialFeatures(self.degree).fit_transform(X)[:,1:]
        else:
            X = PolynomialFeatures(self.degree).fit_transform(X)
        mean, std_withnoise = self.predict(X, return_std=True)[:, numpy.newaxis]
        std = np.sqrt(std_withnoise**2 - (1/self.alpha_))
        if not self.fit_intercept:
            if mean.ndim == 1:
                mean = mean[:, numpy.newaxis]
            mean = self.scaler.inverse_transform(mean)
        if return_withnoise:
            return mean, std, std_withnoise
        else:
            return mean, std


    def sample(self, X, seed=None):
        """Sample a function evaluated at points *X*.

        :param X: A 2d array locations at which to evaluate the sampled function.
        :returns: A 1-D array of the pointwise evaluation of a sampled function.
        """
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
        w_sample = np.random.default_rng(seed).multivariate_normal(self.coef_, self.sigma_)
        if self.fit_intercept:
            X = PolynomialFeatures(self.degree).fit_transform(X)[:,1:]
            return X@w_sample[:,np.newaxis] + self.intercept_
        else:
            X = PolynomialFeatures(self.degree).fit_transform(X)
            y = X@w_sample[:,np.newaxis]
            return self.scaler.inverse_transform(y)

class LinearBanditDiag:
    """
    Implements a `LinearBanditDiag` solver. This solver automatically learns the
    features to extract using a NN model.

    This code comes from https://github.com/ZeroWeight/NeuralTS/blob/master/learner_diag_linear.py
    """
    def __init__(
            self, n_features, n_hidden_dim=32, param_space_bounds=None,
            _lambda=1, nu=1, style="TS", learning_rate=1e-2, update_exploration=False,
            *args, **kwargs
        ):
        self.n_features = n_features
        self.n_hidden_dim = n_hidden_dim
        self.param_space_bounds = param_space_bounds
        self._lambda = _lambda
        self.nu = nu

        self.default_nu = nu
        self.default_lambda = _lambda

        self.update_exploration = update_exploration
        self.style = style
        self.learning_rate = learning_rate

        self.min_features = kwargs.get("min_features", None)
        self.max_features = kwargs.get("max_features", None)
        self.scaler = Scaler(self.min_features, self.max_features)

        self.update_gradient = kwargs.get("update_gradient", True)

        self.__cache = {}

        self.reset()

    def reset(self):
        self.model = LinearModel(self.n_features, self.n_hidden_dim)
        self.total_param = sum(p.numel() for key, p in self.model.named_parameters() if (p.requires_grad) and ("linear" in key))
        self.U = self._lambda * torch.ones((self.total_param,))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.U = self.U.cuda()

    def update(self, X, y, weights=None, *args, **kwargs):
        """
        Updates the weights of the model

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param y: A `numpy.ndarray` of observed rewards
        """
        self.model.train()

        self.clear_cache()

        if self.update_exploration:
            self.nu = max(self.default_nu * 1 / numpy.sqrt(len(X)), 1e-4)
            self._lambda = max(self.default_lambda * 1 / numpy.sqrt(len(X)), 1e-4)

        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
        if y.ndim == 1:
            y = y[:, numpy.newaxis]
        y = self.scaler.fit_transform(y)

        if isinstance(weights, type(None)):
            weights = numpy.ones_like(y)
        assert len(weights) == len(y), "Weights and y should have the same length"

        if self.update_gradient:
            self.add_gradient(X)

        # Convert X, y to torch
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
            weights = weights.cuda()

        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        length = len(y)
        index = numpy.arange(length)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            numpy.random.shuffle(index)
            for idx in index:
                c = X[idx]
                r = y[idx]
                optimizer.zero_grad()
                delta = self.model(c) - r
                loss = delta * delta * weights[idx]
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length

    def add_gradient(self, X):
        """
        Calculate the gradient on sample X and add it to the U matrix
        """
        X = torch.from_numpy(X).float()
        if torch.cuda.is_available():
            X = X.cuda()
        y = self.model(X)

        fx = y[-1]
        self.model.zero_grad()
        fx.backward(retain_graph=True)
        g = torch.cat([p.grad.flatten().detach() for key, p in self.model.named_parameters() if "linear" in key])
        self.U += g * g

    def get_mean(self, X):
        """
        Predicts mean at the given points

        :param X: A `numpy.ndarray` of points with shape (N, features)

        :returns: A `numpy.ndarray` of the mean at X
        """
        self.model.eval()
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
        X = torch.from_numpy(X).float()
        if torch.cuda.is_available():
            X = X.cuda()
        y = self.model(X)
        return self.scaler.inverse_transform(y.cpu().data.numpy())

    def get_mean_std(self, X):
        """
        Predicts mean and standard deviation at the given points

        :param X: A `numpy.ndarray` of points with shape (N, features)

        :returns: A `numpy.ndarray` of the mean at X
                  A `numpy.ndarray` of the std at X
        """
        self.model.eval()
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
        X = torch.from_numpy(X).float()
        if torch.cuda.is_available():
            X = X.cuda()
        y = self.model(X)
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        for fx in y:
            self.model.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for key, p in self.model.named_parameters() if "linear" in key])
            # sigma2 = self._lambda * self.nu * g * g / self.U
            sigma2 = self._lambda * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            sampled.append(sigma.item())

        # TODO: Verify how to update this value properly
        # self.U += g_list[arm] * g_list[arm]
        std = self.scaler.inverse_transform(numpy.array(sampled)[:, numpy.newaxis], std=True)
        return self.scaler.inverse_transform(y.cpu().data.numpy()), std

    def sample(self, X, seed=None, *args, **kwargs):
        """
        Samples the function at points X

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param seed: (optional) An `int` of the random seed

        :returns: A `numpy.ndarray` of the sampled function at the specified points
        """
        self.model.eval()
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
        X = torch.from_numpy(X).float()
        if torch.cuda.is_available():
            X = X.cuda()
        y = self.model(X)

        g_list = []
        sampled, sigmas = [], []
        ave_sigma = 0
        ave_rew = 0
        for i, fx in enumerate(y):
            g = self.cache(X[i], fx)
            # self.model.zero_grad()
            # fx.backward(retain_graph=True)
            # g = torch.cat([p.grad.flatten().detach() for key, p in self.model.named_parameters() if "linear" in key])
            g_list.append(g)

            # sigma2 = self._lambda * self.nu * g * g / self.U
            sigma2 = self._lambda * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            if self.style == 'TS':
                rng = numpy.random.default_rng(seed)
                sample_r = rng.normal(loc=fx.item(), scale=sigma.item())
            elif self.style == 'UCB':
                sample_r = fx.item() + sigma.item()
            else:
                raise RuntimeError('Exploration style not set')
            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r

        # TODO: Verify how to update this value
        # self.U += g_list[arm] * g_list[arm]
        sampled = numpy.array(sampled)[:, numpy.newaxis]
        return self.scaler.inverse_transform(sampled)

    def cache(self, key, value):
        if not self._isin_cache(key):
            self.model.zero_grad()
            value.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for key, p in self.model.named_parameters() if "linear" in key])
            self.__cache[self._convert_to_key(key)] = g
        return self.__cache[self._convert_to_key(key)]

    def clear_cache(self):
        self.__cache = {}

    def _isin_cache(self, value):
        return self._convert_to_key(value) in self.__cache

    def _convert_to_key(self, value):
        if isinstance(value, torch.Tensor):
            value = value.cpu().data.numpy().tolist()
        if isinstance(value, list):
            return str([round(val, 1) for val in value])
        else:
            return str(round(value.item(), 1))

    def get_U(self):
        """
        Gets the exploration matrix

        :returns : A `numpy.ndarray` of the exploration matrix
        """
        return self.U.cpu().data.numpy()

    def set_U(self, U):
        """
        Sets the exploration matrix

        :param: A `numpy.ndarray` of the exploration matrix
        """
        if isinstance(U, numpy.ndarray):
            self.U = torch.tensor(U)
        else:
            self.U = U
        if torch.cuda.is_available():
            self.U = self.U.cuda()

    def update_params(self, **kwargs):
        """
        Updates the regressor parameters

        :param params: A `dict` of the parameters to update
        """
        for key, value in kwargs.items():
            if key == "U":
                self.set_U(value)
            elif key == "model":
                # Sends model on gpu if applicable
                if not next(value.parameters()).is_cuda and torch.cuda.is_available():
                    value = value.cuda()
                setattr(self, key, value)
            else:
                setattr(self, key, value)

    def save_ckpt(self, path, prefix="", trial=""):
        """
        Saves a checkpoint of the model

        :param path: A `str` of the model path
        :param prefix: A `str` of the model name
        """
        path = os.path.join(path, "models")
        if trial:
            path = os.path.join(path, trial)

        os.makedirs(path, exist_ok=True)
        savename = f"{prefix}_model.ckpt" if prefix else "model.ckpt"
        torch.save(self, os.path.join(path, savename))

    def load_ckpt(self, path, prefix="", trial=""):
        """
        Loads a checkpoint of the model

        :param path: A `str` of the model path
        :param prefix: A `str` of the model name
        """
        path = os.path.join(path, "models")
        if trial:
            path = os.path.join(path, trial)

        savename = f"{prefix}_model.ckpt" if prefix else "model.ckpt"
        params = torch.load(os.path.join(path, savename), map_location="cpu")
        self.update_params(**vars(params))

        # self.model.load_state_dict(params.model.state_dict())

class ContextualLinearBanditDiag(LinearBanditDiag):
    """
    Implements a `ContextualLinearBanditDiag` solver. This solver automatically learns the
    features to extract using a NN model.

    This code comes from https://github.com/ZeroWeight/NeuralTS/blob/master/learner_diag_linear.py
    """
    def __init__(
            self, n_features, n_hidden_dim=32, param_space_bounds=None,
            _lambda=1, nu=1, style="TS", learning_rate=1e-2, update_exploration=False,
            ctx_features=1,
            *args, **kwargs
        ):
        self.ctx_features = ctx_features
        self.every_step_decision = kwargs.get("every-step-decision", False)

        super(ContextualLinearBanditDiag, self).__init__(
            n_features, n_hidden_dim, param_space_bounds, _lambda, nu, style,
            learning_rate, update_exploration, *args, **kwargs
        )

        self.histories = []
        self.idx = kwargs.get("idx")

    def reset(self):
        self.histories = []
        self.model = ContextLinearModel(self.n_features + self.ctx_features, self.n_hidden_dim, every_step_decision=self.every_step_decision)
        self.total_param = sum(p.numel() for key, p in self.model.named_parameters() if (p.requires_grad) and ("linear" in key))
        self.U = self._lambda * torch.ones((self.total_param,))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.U = self.U.cuda()

    def update(self, X, y, history, weights=None, *args, **kwargs):
        """
        Updates the weights of the model

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param y: A `numpy.ndarray` of observed rewards
        """
        self.model.train()

        self.clear_cache()
        if self.update_exploration:
            self.nu = max(self.default_nu * 1 / numpy.sqrt(len(X)), 1e-4)
            self._lambda = max(self.default_lambda * 1 / numpy.sqrt(len(X)), 1e-4)

        # We will be training with all the acquired histories
        # To do so, we must recreate the sequence for each data point
        history = copy.deepcopy(history)
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
            if len(history["X"]) > 0:
                history["X"] = rescale_X(numpy.concatenate(history["X"], axis=1).T, self.param_space_bounds)
                history["y"] = self.scaler.fit_transform(numpy.array(history["y"]))[:, [self.idx]]
                history["ctx"] = numpy.array(history["ctx"])
        self.histories.append(history)

        histories = []
        for history_ in self.histories:
            if self.every_step_decision:
                for i in range(len(history["X"])):
                    histories.append([
                        history_["X"][[i]], # Keeps a [1, N] shape
                        history_["y"][[i]], # Keeps a [1, 1] shape
                        history_["ctx"][[i]],
                        {
                            "X" : history_["X"][:i],
                            "y" : history_["y"][:i],
                            "ctx" : history_["ctx"][:i + 1] # Uses the first context
                        }
                    ])
            else:
                histories.append([
                    history_["X"][[0]], # Keeps a [1, N] shape
                    history_["y"][[-1]], # Keeps a [1, 1] shape; Only last reward is kept
                    history_["ctx"][[0]],
                    {
                        "X" : history_["X"][[0]],
                        "y" : history_["y"][[-1]],
                        "ctx" : history_["ctx"][[0]]
                    }
                ])

        # Convert X, y to torch
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        if torch.cuda.is_available():
            X = X.cuda()
            history = to_cuda(history)
            histories = to_cuda(histories)

        if self.update_gradient:
            self.add_gradient(X, history)

        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        length = len(histories)
        index = numpy.arange(length)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            numpy.random.shuffle(index)
            for idx in index:
                X, y, ctx, history = histories[idx]
                optimizer.zero_grad()
                delta = self.model(X, history) - y
                loss = delta * delta # * weights[idx]
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length

    def add_gradient(self, X, history):
        """
        Calculate the gradient on sample X and add it to the U matrix
        """
        y = self.model(X, history)
        fx = y[-1]
        self.model.zero_grad()
        fx.backward(retain_graph=True)
        g = torch.cat([p.grad.flatten().detach() for key, p in self.model.named_parameters() if "linear" in key])
        self.U += g * g

    def get_mean(self, X):
        """
        Predicts mean at the given points

        :param X: A `numpy.ndarray` of points with shape (N, features)

        :returns: A `numpy.ndarray` of the mean at X
        """
        self.model.eval()

        history = copy.deepcopy(history)
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
            history["X"] = rescale_X(numpy.concatenate(history["X"], axis=1).T, self.param_space_bounds)
        X = torch.from_numpy(X).float()
        if torch.cuda.is_available():
            X = X.cuda()
            for key, value in history.items():
                value = torch.from_numpy(numpy.array(value)).float()
                history[key] = value.cuda()
        y = self.model(X, history)

        return self.scaler.inverse_transform(y.cpu().data.numpy())

    def get_mean_std(self, X):
        """
        Predicts mean and standard deviation at the given points

        :param X: A `numpy.ndarray` of points with shape (N, features)

        :returns: A `numpy.ndarray` of the mean at X
                  A `numpy.ndarray` of the std at X
        """
        self.model.eval()

        history = copy.deepcopy(history)
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
            history["X"] = rescale_X(numpy.concatenate(history["X"], axis=1).T, self.param_space_bounds)
        X = torch.from_numpy(X).float()
        if torch.cuda.is_available():
            X = X.cuda()
            history = to_cuda(history)
        y = self.model(X, history)

        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        for fx in y:
            self.model.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for key, p in self.model.named_parameters() if "linear" in key])
            # sigma2 = self._lambda * self.nu * g * g / self.U
            sigma2 = self._lambda * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            sampled.append(sigma.item())

        # TODO: Verify how to update this value properly
        # self.U += g_list[arm] * g_list[arm]
        std = self.scaler.inverse_transform(numpy.array(sampled)[:, numpy.newaxis], std=True)
        return self.scaler.inverse_transform(y.cpu().data.numpy()), std

    def sample(self, X, history, seed=None, *args, **kwargs):
        """
        Samples the function at points X

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param seed: (optional) An `int` of the random seed

        :returns: A `numpy.ndarray` of the sampled function at the specified points
        """
        self.model.eval()

        history = copy.deepcopy(history)
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
            if len(history["X"]) > 0:
                history["X"] = rescale_X(numpy.concatenate(history["X"], axis=1).T, self.param_space_bounds)
                history["y"] = self.scaler.fit_transform(numpy.array(history["y"]))[:, [self.idx]]
                history["ctx"] = numpy.array(history["ctx"])

        X = torch.from_numpy(X).float()
        if torch.cuda.is_available():
            X = X.cuda()
            history = to_cuda(history)
        y = self.model(X, history)

        g_list = []
        sampled, sigmas = [], []
        ave_sigma = 0
        ave_rew = 0
        for i, fx in enumerate(y):
            g = self.cache(X[i], fx)
            # self.model.zero_grad()
            # fx.backward(retain_graph=True)
            # g = torch.cat([p.grad.flatten().detach() for key, p in self.model.named_parameters() if "linear" in key])
            g_list.append(g)

            # sigma2 = self._lambda * self.nu * g * g / self.U
            sigma2 = self._lambda * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            if self.style == 'TS':
                rng = numpy.random.default_rng(seed)
                sample_r = rng.normal(loc=fx.item(), scale=sigma.item())
            elif self.style == 'UCB':
                sample_r = fx.item() + sigma.item()
            else:
                raise RuntimeError('Exploration style not set')
            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r

        # TODO: Verify how to update this value
        # self.U += g_list[arm] * g_list[arm]
        sampled = numpy.array(sampled)[:, numpy.newaxis]
        return self.scaler.inverse_transform(sampled)

class ContextualImageLinearBanditDiag(ContextualLinearBanditDiag):
    def __init__(self, *args, **kwargs):

        self.datamap_opts = kwargs.get("datamap_opts", {"shape": 64})
        self.pretrained_opts = kwargs.get("pretrained_opts", {"use" : False})

        super(ContextualImageLinearBanditDiag, self).__init__(*args, **kwargs)

    def reset(self):
        self.histories = []
        self.model = ImageContextLinearModel(
            self.n_features, self.datamap_opts["shape"], self.n_hidden_dim,
            every_step_decision=self.every_step_decision, pretrained_opts=self.pretrained_opts
        )
        self.total_param = sum(p.numel() for key, p in self.model.named_parameters() if (p.requires_grad) and ("linear" in key))
        self.U = self._lambda * torch.ones((self.total_param,))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.U = self.U.cuda()

class LSTMLinearBanditDiag(LinearBanditDiag):
    """
    Implements a `LSTMLinearBanditDiag` solver. This solver automatically learns the
    features to extract using a NN model.

    This code comes from https://github.com/ZeroWeight/NeuralTS/blob/master/learner_diag_linear.py
    """
    def __init__(
            self, n_features, n_hidden_dim=32, param_space_bounds=None,
            _lambda=1, nu=1, style="TS", learning_rate=1e-2, update_exploration=False,
            *args, **kwargs
        ):
        super(LSTMLinearBanditDiag, self).__init__(
            n_features, n_hidden_dim, param_space_bounds, _lambda, nu, style,
            learning_rate, update_exploration, *args, **kwargs
        )

        self.histories = []
        self.idx = kwargs.get("idx")

    def reset(self):
        self.histories = []
        self.model = LSTMLinearModel(self.n_features, self.n_hidden_dim)
        self.total_param = sum(p.numel() for key, p in self.model.named_parameters() if (p.requires_grad) and ("linear" in key))
        self.U = self._lambda * torch.ones((self.total_param,))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.U = self.U.cuda()

    def update(self, X, y, history, weights=None, *args, **kwargs):
        """
        Updates the weights of the model

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param y: A `numpy.ndarray` of observed rewards
        """
        self.clear_cache()
        if self.update_exploration:
            self.nu = max(self.default_nu * 1 / numpy.sqrt(len(X)), 1e-4)
            self._lambda = max(self.default_lambda * 1 / numpy.sqrt(len(X)), 1e-4)

        # We will be training with all the acquired histories
        # To do so, we must recreate the sequence for each data point
        history = copy.deepcopy(history)
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
            if len(history["X"]) > 0:
                history["X"] = rescale_X(numpy.concatenate(history["X"], axis=1).T, self.param_space_bounds)
                history["y"] = self.scaler.fit_transform(numpy.array(history["y"]))[:, [self.idx]]
                history["ctx"] = numpy.array(history["ctx"])
        self.histories.append(history)

        histories = []
        for history_ in self.histories:
            for i in range(len(history["X"])):
                histories.append([
                    history_["X"][[i]], # Keeps a [1, N] shape
                    history_["y"][[i]], # Keeps a [1, 1] shape
                    history_["ctx"][i],
                    {
                        "X" : history_["X"][:i],
                        "y" : history_["y"][:i],
                        "ctx" : history_["ctx"][:i]
                    }
                ])

        # Convert X, y to torch
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        if torch.cuda.is_available():
            X = X.cuda()
            history = to_cuda(history)
            histories = to_cuda(histories)

        if self.update_gradient:
            self.add_gradient(X, history)

        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        length = len(histories)
        index = numpy.arange(length)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            numpy.random.shuffle(index)
            for idx in index:
                X, y, ctx, history = histories[idx]
                optimizer.zero_grad()
                delta = self.model(X, history) - y
                loss = delta * delta # * weights[idx]
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length

    def add_gradient(self, X, history):
        """
        Calculate the gradient on sample X and add it to the U matrix
        """
        y = self.model(X, history)
        fx = y[-1]
        self.model.zero_grad()
        fx.backward(retain_graph=True)
        g = torch.cat([p.grad.flatten().detach() for key, p in self.model.named_parameters() if "linear" in key])
        self.U += g * g

    def get_mean(self, X):
        """
        Predicts mean at the given points

        :param X: A `numpy.ndarray` of points with shape (N, features)

        :returns: A `numpy.ndarray` of the mean at X
        """
        history = copy.deepcopy(history)
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
            history["X"] = rescale_X(numpy.concatenate(history["X"], axis=1).T, self.param_space_bounds)
        X = torch.from_numpy(X).float()
        if torch.cuda.is_available():
            X = X.cuda()
            for key, value in history.items():
                value = torch.from_numpy(numpy.array(value)).float()
                history[key] = value.cuda()
        y = self.model(X, history)

        return self.scaler.inverse_transform(y.cpu().data.numpy())

    def get_mean_std(self, X):
        """
        Predicts mean and standard deviation at the given points

        :param X: A `numpy.ndarray` of points with shape (N, features)

        :returns: A `numpy.ndarray` of the mean at X
                  A `numpy.ndarray` of the std at X
        """
        history = copy.deepcopy(history)
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
            history["X"] = rescale_X(numpy.concatenate(history["X"], axis=1).T, self.param_space_bounds)
        X = torch.from_numpy(X).float()
        if torch.cuda.is_available():
            X = X.cuda()
            history = to_cuda(history)
        y = self.model(X, history)

        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        for fx in y:
            self.model.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for key, p in self.model.named_parameters() if "linear" in key])
            # sigma2 = self._lambda * self.nu * g * g / self.U
            sigma2 = self._lambda * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            sampled.append(sigma.item())

        # TODO: Verify how to update this value properly
        # self.U += g_list[arm] * g_list[arm]
        std = self.scaler.inverse_transform(numpy.array(sampled)[:, numpy.newaxis], std=True)
        return self.scaler.inverse_transform(y.cpu().data.numpy()), std

    def sample(self, X, history, seed=None, *args, **kwargs):
        """
        Samples the function at points X

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param seed: (optional) An `int` of the random seed

        :returns: A `numpy.ndarray` of the sampled function at the specified points
        """
        history = copy.deepcopy(history)
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
            if len(history["X"]) > 0:
                history["X"] = rescale_X(numpy.concatenate(history["X"], axis=1).T, self.param_space_bounds)
                history["y"] = self.scaler.fit_transform(numpy.array(history["y"]))[:, [self.idx]]
                history["ctx"] = numpy.array(history["ctx"])

        X = torch.from_numpy(X).float()
        if torch.cuda.is_available():
            X = X.cuda()
            history = to_cuda(history)
        y = self.model(X, history)

        g_list = []
        sampled, sigmas = [], []
        ave_sigma = 0
        ave_rew = 0
        for i, fx in enumerate(y):
            g = self.cache(X[i], fx)
            # self.model.zero_grad()
            # fx.backward(retain_graph=True)
            # g = torch.cat([p.grad.flatten().detach() for key, p in self.model.named_parameters() if "linear" in key])
            g_list.append(g)

            # sigma2 = self._lambda * self.nu * g * g / self.U
            sigma2 = self._lambda * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            if self.style == 'TS':
                rng = numpy.random.default_rng(seed)
                sample_r = rng.normal(loc=fx.item(), scale=sigma.item())
            elif self.style == 'UCB':
                sample_r = fx.item() + sigma.item()
            else:
                raise RuntimeError('Exploration style not set')
            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r

        # TODO: Verify how to update this value
        # self.U += g_list[arm] * g_list[arm]
        sampled = numpy.array(sampled)[:, numpy.newaxis]
        return self.scaler.inverse_transform(sampled)

class TS_sampler():
    """This class relies on regressor class to generate options to present to the user
    using a Thompson Sampling approach.

    :param regressor: class containing the methods update(self, X, y),
                      get_mean_std(self, X) and sample(self, X)
    """
    def __init__(self, regressor):
        self.regressor = regressor
        self.X = None
        self.y = None
        self.weights = None
        self.loaded = False

    def predict(self, X_pred, *args, **kwargs):
        """Predict mean and standard deviation at given points *X_pred*.

        :param X_pred: A 2d array of locations at which to predict.
        :returns: An array of means and an array of standard deviations.
        """
        if (self.X is not None) or self.loaded:
            return self.regressor.get_mean_std(X_pred, *args, **kwargs)
        else :
            mean = np.full(X_pred.shape[0], 0)
            std = np.full(X_pred.shape[0], 1)
            return mean, std

    def sample(self, X_sample, seed=None, *args, **kwargs):
        """Sample a function evaluated at points *X_sample*. When no points have
        been observed yet, the function values are sampled uniformly between 0 and 1.

        :param X_sample: A 2d array locations at which to evaluate the sampled function.
        :returns: A 1-D array of the pointwise evaluation of a sampled function.
        """
        if (self.X is not None) or (self.loaded):
            return self.regressor.sample(X_sample, seed=seed, *args, **kwargs)
        else:
#             mean= np.full(X_sample.shape[0], 0)
#             cov = np.identity(X_sample.shape[0])
#             rng = np.random.default_rng()
#             f_tilde = rng.multivariate_normal(mean, cov, method='eigh')
            f_tilde = np.random.uniform(0,1,X_sample.shape[0])[:,np.newaxis]
        return f_tilde

    def update(self, action, reward, weights=None, update_posterior=True, *args, **kwargs):
        """Update the regression model using the observations *reward* acquired at
        location *action*.

        :param action: A 2d array of locations.
        :param reward: A 1-D array of observations.
        """
        if self.X is not None:
            self.X = np.append(self.X, action, axis=0)
            self.y = np.append(self.y, reward, axis=0)
            if not isinstance(weights, type(None)):
                self.weights = np.append(self.weights, weights, axis=0)
        else:
            self.X = action
            self.y = reward
            self.weights = weights
        if update_posterior:
            self.regressor.update(self.X, self.y, weights=self.weights, *args, **kwargs)

    def update_params(self, **kwargs):
        """
        Update the regressor parameters
        """
        self.regressor.update_params(**kwargs)

    def save_ckpt(self, path, *args, **kwargs):
        """
        Saves a checkpoint of the model
        """
        self.regressor.save_ckpt(path=path, *args, **kwargs)

    def load_ckpt(self, path, *args, **kwargs):
        """
        Loads a checkpoint of the model
        """
        self.loaded = True
        self.regressor.load_ckpt(path=path, *args, **kwargs)
