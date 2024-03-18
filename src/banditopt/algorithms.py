"""This module contains classes to generate options presented to the user in the online
multi-objective bandits optimization problem.

The functions from this file are adapted from

.. [1] https://github.com/PDKlab/STED-Optimization/blob/master/src/algorithms.py
.. [2] https://github.com/ZeroWeight/NeuralTS/blob/master/learner_diag_linear.py
"""

import numpy
import torch
import os
import time
import pickle

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

def to_tensor(elements):
    """
    Recursively sends data to GPU
    """
    if isinstance(elements, (tuple)):
        elements = list(elements)
    if isinstance(elements, (list)):
        for i, element in enumerate(elements):
            elements[i] = to_tensor(element)
    elif isinstance(elements, dict):
        for key, value in elements.items():
            elements[key] = to_tensor(value)
    else:
        if isinstance(elements, numpy.ndarray):
            elements = torch.tensor(elements, dtype=torch.float32)
        if isinstance(elements, torch.Tensor):
            return elements
    return elements

class Scaler:
    """
    Min/Max scaler. Implementation is follows the implementation 
    of `StandardScaler` from scikit-learn library for ease of use.
    """
    def __init__(self, _min, _max):
        """
        Instantiates a `Scaler`

        :param _min: A `numpy.ndarray` of minimal bounds
        :param _max: A `numpy.ndarray` of maximal bounds
        """
        if isinstance(_min, type(None)):
            self._min = 0.
            self._max = 1.
        else:
            self._min = numpy.array(_min)[:, numpy.newaxis]
            self._max = numpy.array(_max)[:, numpy.newaxis]

    def fit_transform(self, X):
        """
        Transforms the data using the min/max values

        :param X: A `numpy.ndarray` of data
        """
        return self.transform(X)

    def transform(self, X):
        """
        Transforms the data using the min/max values

        :param X: A `numpy.ndarray` of data
        """
        return (X - self._min) / (self._max - self._min)

    def inverse_transform(self, X, std=False):
        """
        Inverse transform of the data using the min/max values

        :param X: A `numpy.ndarray` of data
        :param std: (optional) Complete rescale of the data
        """
        if std:
            return X * (self._max - self._min)
        else:
            return X * (self._max - self._min) + self._min

class MO_function_sample():
    """
    Creates a multi-objective function sample with randomly generated random seeds

    :param algos: A `list` of `TS_Sampler`
    :param with_time: A `bool` to optimize the dwell time
    :param param_names: A `list` of parameter names

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
        """
        Evaluates each individual using the regression models.

        :param individuals: A `list` of individual parameter value
        :param params_to_round: A `list` of param_names to round
        :param weights: A `list` of the weights when `time_limit` used when time
                        limit is defined.

        :returns: A `list` of evaluated parameters
        """        
        X = numpy.array(individuals)
        for param in params_to_round:
            X[:, self.param_names.index(param)] = numpy.round(X[:, self.param_names.index(param)])

        ys = numpy.array([self.algos[i].sample(X, seed=self.seeds[i], history=self.history, scale=0) for i in range(len(self.algos))]).squeeze(axis=-1)
        # ys = numpy.array([self.algos[i].predict(X)[0] for i in range(len(self.algos))]).squeeze(axis=-1)
        if self.time_limit is not None:
            pixeltimes = X[:, self.param_names.index("pdt")] * X[:, self.param_names.index("line_step")] * X[:, self.param_names.index("pixelsize")]**2/(20e-9)**2
            for i, bounds in enumerate(self.borders):
                if weights[i] < 0:
                    ys[i, :][pixeltimes > self.time_limit] = bounds[1] + (bounds[1]-bounds[0])*1.5
                elif weights[i] > 0:
                    ys[i, :][pixeltimes > self.time_limit] = bounds[0] - (bounds[1]-bounds[0])*1.5

        if self.with_time:
            ys = numpy.concatenate((ys, X[:, self.param_names.index("pdt")][numpy.newaxis]), axis=0)
            # return tuple(ys + X[:, self.param_names.index("pdt")])

        return list(map(tuple, ys.T))

def rescale_X(X, param_space_bounds):
    """
    Method to rescale the parameters using the half interval value

    :param X: A `numpy.ndarray` of parameters with shape (N, features)
    :param param_space_bounds: A `list` of tuple of (min, max) parameter bounds

    :returns: A `numpy.ndarray` of the rescaled parameters
    """    
    param_space_bounds = numpy.array(param_space_bounds).T
    xmin, xmax = param_space_bounds
    xmean = (xmax + xmin) / 2
    if X.ndim < 2:
        return []
    if X.ndim == 3:
        xmean, xmax, xmin = xmean[numpy.newaxis], xmax[numpy.newaxis], xmin[numpy.newaxis]
    if isinstance(X, torch.Tensor):
        xmean, xmax, xmin = torch.tensor(xmean, dtype=torch.float32), torch.tensor(xmax, dtype=torch.float32), torch.tensor(xmin, dtype=torch.float32)
        return (X - xmean.unsqueeze(0)) / (0.5 * (xmax - xmin).unsqueeze(0))
    X = (X - xmean[numpy.newaxis]) / (0.5 * (xmax - xmin)[numpy.newaxis])
    return X
    # X = copy.deepcopy(X)
    # for col in range(X.shape[1]):
    #     xmin, xmax = param_space_bounds[col]
    #     xmean = (xmax+xmin)/2
    #     X[:,col] = (X[:,col] - xmean)/(0.5 * (xmax - xmin))
    # return X

class sklearn_GP(GaussianProcessRegressor):
    """
    This class is meant to be used as a the regressor argument of the TS_sampler
    class. It uses a scikit-learn implementation of Gaussian process regression. The
    regularization is fixed (no adaptative regularization).
    """
    def __init__(self, bandwidth, s_lb, s_ub, param_space_bounds=None, **kwargs):
        """
        Instantiates the `sklearn_GP` model

        :param bandwith: A `float` of the bandwidth of the `RBF`
        :param s_lb: A `float` of the lower bound on noise
        :param s_ub: A `float` of the upper bound on noise
        :param param_space_bounds: A `list` of tuple of (min, max) parameter bounds
        :param normalize_y: A `bool` to normalize the values
        """
        self.bandwidth = bandwidth
        self.s_lb = s_lb
        self.s_ub = s_ub
        self.norm_bound = 5.0
        self.lambda_ = s_ub ** 2 / self.norm_bound ** 2

        super().__init__(RBF(length_scale=self.bandwidth), alpha=self.lambda_, optimizer=None, normalize_y=False)

        self.param_space_bounds = param_space_bounds
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.idx = kwargs.get("idx", None)

    def update(self, X, y, *args, **kwargs):
        """
        Updates the weights of the model

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param y: A `numpy.ndarray` of observed rewards
        """
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
        if y.ndim == 1:
            y = y[:, numpy.newaxis]
        y = self.scaler.fit_transform(y)

        self.fit(X,y)

    def get_mean_std(self, X):
        """
        Predicts mean and standard deviation at the given points

        :param X: A `numpy.ndarray` of points with shape (N, features)

        :returns: A `numpy.ndarray` of the mean at X
                  A `numpy.ndarray` of the std at X
        """        
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
        """
        Samples the function at points X

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param seed: (optional) An `int` of the random seed

        :returns: A `numpy.ndarray` of the sampled function at the specified points
        """        
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)

        mean, k = self.predict(X, return_cov=True)
        # Ensures covariance matrix is positivie-semidefinite
        min_eig = numpy.min(numpy.real(numpy.linalg.eigvals(k)))
        if min_eig < 0:
            k -= 10 * min_eig * numpy.eye(*k.shape)

        cov = self.s_ub ** 2 / self.lambda_ * k

        # Rescales sampled mean
        rng = numpy.random.default_rng(seed)
        f_tilde = rng.multivariate_normal(mean.flatten(), cov, method='eigh', check_valid="ignore")[:,numpy.newaxis]
        f_tilde = self.scaler.inverse_transform(f_tilde)
        return f_tilde

    def save_ckpt(self, path, prefix="", trial=""):
        """
        Saves a checkpoint of the model

        :param path: A `str` of the model path
        :param prefix: A `str` of the model name
        :param trial: A `str` of the trial number
        """
        path = os.path.join(path, "models")
        if trial:
            path = os.path.join(path, trial)

        os.makedirs(path, exist_ok=True)
        savename = f"{prefix}_model.ckpt" if prefix else "model.ckpt"
        pickle.dump(self, open(os.path.join(path, savename), "wb"))

    def load_ckpt(self, path, prefix="", trial=""):
        """
        Loads a checkpoint of the model

        :param path: A `str` of the model path
        :param prefix: A `str` of the model name
        :param trial: A `str` of the trial number
        """
        path = os.path.join(path, "models")
        if trial:
            path = os.path.join(path, trial)

        savename = f"{prefix}_model.ckpt" if prefix else "model.ckpt"
        params = pickle.load(open(os.path.join(path, savename), "rb"))
        self.update_params(**vars(params))

    def set_sampling_mode(self, is_sampling):
        pass

class sklearn_BayesRidge(BayesianRidge):
    """
    This class is meant to be used as a the regressor argument of the TS_sampler
    class. It uses a scikit-learn implementation of bayesian linear regression to fit a
    polynomial of a certain degree. 
    
    fit_intercept=False should be used.

    :param degree: An `int` of the degree of the polynomial
    :param param_space_bounds: A `list` of tuple of (min, max) parameter bounds
    
    Note.
    see `sklearn.linear_model.BayesianRidge` documentation for parameter description
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
        """
        Updates the weights of the model

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param y: A `numpy.ndarray` of observed rewards
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
        """
        Predicts mean and standard deviation at the given points

        :param X: A `numpy.ndarray` of points with shape (N, features)

        :returns: A `numpy.ndarray` of the mean at X
                  A `numpy.ndarray` of the std at X
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
        """
        Samples the function at points X

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param seed: (optional) An `int` of the random seed

        :returns: A `numpy.ndarray` of the sampled function at the specified points
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
        pickle.dump(self, open(os.path.join(path, savename), "wb"))

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
        params = pickle.load(open(os.path.join(path, savename), "rb"))
        self.update_params(**vars(params))

    def set_sampling_mode(self, is_sampling):
        pass

class LinearBanditDiag:
    """
    Implements a `LinearBanditDiag` solver. This solver automatically learns the
    features to extract using a NN model.

    This code is adapted from https://github.com/ZeroWeight/NeuralTS/blob/master/learner_diag_linear.py
    """
    def __init__(
            self, n_features, n_hidden_dim=32, param_space_bounds=None,
            _lambda=1, nu=1, style="TS", learning_rate=1e-2, update_exploration=False,
            *args, **kwargs
        ):
        """
        Instantiates `LinearBanditDiag`

        :param n_features: An `int` of the number of features in the parameter space
        :param n_hidden_dim: An `int` of the number of hidden nodes in the NN model
        :param param_space_bounds: A `list` of `tuple` (min, max) of the parameter space
        :param _lambda: A `float` of the regularization parameter
        :param nu: A `float` to control the variance
        :param style: A `str` of the style to use in the solver
        :param learning_rate: A `float` of the learning rate to use for optimization
        :param update_exploration: A `bool` to reduce the exploration during optimization 
        """
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

        self.idx = kwargs.get("idx")

        self.__cache = {}

        self.reset()

    def get_gradient(self):
        """
        Computes the gradient of the model
        """
        g = torch.cat([p.grad.flatten().detach() for key, p in self.model.named_parameters() if "linear" in key])
        return g

    def reset(self):
        """
        Resets the model
        """
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
        :param weights: (optional) A `numpy.ndarray` of the weights of each observations
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

        :param X: A `numpy.ndarray` of points with shape (N, features)
        """
        X = torch.from_numpy(X).float()
        if torch.cuda.is_available():
            X = X.cuda()
        y = self.model(X)

        fx = y[-1]
        self.model.zero_grad()
        fx.backward(retain_graph=True)
        g = self.get_gradient()
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
            g = self.get_gradient()
            sigma2 = self._lambda * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            sampled.append(sigma.item())

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
                sample_r = rng.normal(loc=fx.item(), scale=sigma.item() * kwargs.get("scale", 1))
            elif self.style == 'UCB':
                sample_r = fx.item() + sigma.item()
            else:
                raise RuntimeError('Exploration style not set')
            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r

        sampled = numpy.array(sampled)[:, numpy.newaxis]
        return self.scaler.inverse_transform(sampled)

    def train_batch(self, X, y, weights=None, *args, **kwargs):
        """
        Trains the model using a batch of observations

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param y: A `numpy.ndarray` of observed rewards
        :param weights: (optional) A `numpy.ndarray` of the weights for each observations
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
        y = y[:, [self.idx]]

        if isinstance(weights, type(None)):
            weights = numpy.ones_like(y)
        assert len(weights) == len(y), "Weights and y should have the same length"

        # if self.update_gradient:
        #     self.add_gradient(X)

        # Convert X, y to torch
        X = X.unsqueeze(1)
        y = y.to(torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        if torch.cuda.is_available():
            X = to_cuda(X)
            y = to_cuda(y)
            weights = to_cuda(weights)

        criterion = kwargs.get("criterion", nn.MSELoss())
        optimizer = kwargs.get("optimizer", optim.SGD(self.model.parameters(), lr=self.learning_rate))

        length = len(X)
        index = numpy.arange(length)

        numpy.random.shuffle(index)
        batch_loss = 0
        for idx in index:
            optimizer.zero_grad()
            pred = self.model(X[idx])
            loss = criterion(pred, y[[idx]])
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

        return batch_loss

    def predict_batch(self, X, y, weights=None, *args, **kwargs):
        """
        Predicts a batch of observations

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param y: A `numpy.ndarray` of observed rewards
        :param weights: (optional) A `numpy.ndarray` of the weights for each observations
        """
        self.model.eval()

        self.clear_cache()

        if self.update_exploration:
            self.nu = max(self.default_nu * 1 / numpy.sqrt(len(X)), 1e-4)
            self._lambda = max(self.default_lambda * 1 / numpy.sqrt(len(X)), 1e-4)

        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
        if y.ndim == 1:
            y = y[:, numpy.newaxis]
        y = self.scaler.fit_transform(y)
        y = y[:, [self.idx]]

        if isinstance(weights, type(None)):
            weights = numpy.ones_like(y)
        assert len(weights) == len(y), "Weights and y should have the same length"

        # if self.update_gradient:
        #     self.add_gradient(X)

        # Convert X, y to torch
        X = X.unsqueeze(1)
        y = y.to(torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        if torch.cuda.is_available():
            X = to_cuda(X)
            y = to_cuda(y)
            weights = to_cuda(weights)

        criterion = kwargs.get("criterion", nn.MSELoss())

        length = len(X)
        index = numpy.arange(length)

        numpy.random.shuffle(index)
        batch_loss = 0
        for idx in index:
            pred = self.model(X[idx])
            loss = criterion(pred, y[[idx]])
            batch_loss += loss.item()

        return batch_loss

    def add_gradient_batch(self, X, y, weights=None, *args, **kwargs):
        """
        Updates the gradient of the model using a batch of observations

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param y: A `numpy.ndarray` of observed rewards
        :param weights: (optional) A `numpy.ndarray` of the weights for each observations
        """
        self.model.eval()

        self.clear_cache()

        if self.update_exploration:
            self.nu = max(self.default_nu * 1 / numpy.sqrt(len(X)), 1e-4)
            self._lambda = max(self.default_lambda * 1 / numpy.sqrt(len(X)), 1e-4)

        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
        if y.ndim == 1:
            y = y[:, numpy.newaxis]
        y = self.scaler.fit_transform(y)
        y = y[:, [self.idx]]

        if isinstance(weights, type(None)):
            weights = numpy.ones_like(y)
        assert len(weights) == len(y), "Weights and y should have the same length"

        # if self.update_gradient:
        #     self.add_gradient(X)

        # Convert X, y to torch
        X = X.unsqueeze(1)
        y = y.to(torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        if torch.cuda.is_available():
            X = to_cuda(X)
            y = to_cuda(y)
            weights = to_cuda(weights)

        criterion = kwargs.get("criterion", nn.MSELoss())

        length = len(X)
        index = numpy.arange(length)

        numpy.random.shuffle(index)
        batch_loss = 0
        for idx in index:
            y = self.model(X[idx])

            fx = y[-1]
            self.model.zero_grad()
            fx.backward(retain_graph=True)
            g = self.get_gradient()
            self.U += g * g

        return batch_loss

    def cache(self, key, value):
        """
        Cache gradient value associated to observation

        :param key: key of the value
        :param value: value to cache

        :returns : The cached value 
        """
        if not self._isin_cache(key):
            self.model.zero_grad()
            value.backward(retain_graph=True)
            g = self.get_gradient()
            self.__cache[self._convert_to_key(key)] = g
        return self.__cache[self._convert_to_key(key)]

    def clear_cache(self):
        """
        Clears the current cache
        """
        self.__cache = {}

    def _isin_cache(self, key):
        """
        Verifies if `exists`

        :returns : A `bool` if `key` exists
        """
        return self._convert_to_key(key) in self.__cache

    def _convert_to_key(self, value):
        """
        Converts the values to a key. The values are rounded to 1 decimal

        :param value: Observations

        :returns : A `str` of the value
        """
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

        :param U: A `numpy.ndarray` of the exploration matrix
        """
        if isinstance(U, numpy.ndarray):
            self.U = torch.tensor(U)
        else:
            self.U = U
        if torch.cuda.is_available():
            self.U = self.U.cuda()

    def set_sampling_mode(self, is_sampling):
        """
        Activates/Deactivates the sampling mode of the model

        :param is_sampling: A `boolean` if the model is in sampling mode
        """
        self.model.sampling(is_sampling)

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
        :param trial: A `str` of the trial number
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
        :param trial: A `str` of the trial number
        """
        path = os.path.join(path, "models")
        if trial:
            path = os.path.join(path, trial)

        savename = f"{prefix}_model.ckpt" if prefix else "model.ckpt"
        if not os.path.isfile(os.path.join(path, savename)):
            print("[!!!!] Model {} does not exist...".format(os.path.join(path, savename)))
            return
        params = torch.load(os.path.join(path, savename), map_location="cpu")
        self.update_params(**vars(params))

class NeuralTS(LinearBanditDiag):
    """
    Implements a `NeuralTS` solver. This solver automatically learns the
    features to extract using a NN model.

    This code is adapted from https://github.com/ZeroWeight/NeuralTS/blob/master/learner_diag_linear.py
    """    
    def __init__(self, *args, **kwargs):
        super(NeuralTS, self).__init__(
            *args, **kwargs
        )
        
    def get_gradient(self):
        """
        Computes the gradient of the model
        """
        g = torch.cat([p.grad.flatten().detach() for key, p in self.model.named_parameters() if (p.requires_grad)])
        return g

    def reset(self):
        """
        Resets the model
        """
        self.model = LinearModel(self.n_features, self.n_hidden_dim)
        self.total_param = sum(p.numel() for key, p in self.model.named_parameters() if (p.requires_grad))
        self.U = self._lambda * torch.ones((self.total_param,))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.U = self.U.cuda()        

class ContextualLinearBanditDiag(LinearBanditDiag):
    """
    Implements a `ContextualLinearBanditDiag` solver. This solver automatically learns the
    features to extract using a NN model.

    This model relies on contextual information in the form of a vector.
    """
    def __init__(
            self, n_features, n_hidden_dim=32, param_space_bounds=None,
            _lambda=1, nu=1, style="TS", learning_rate=1e-2, update_exploration=False,
            ctx_features=1,
            *args, **kwargs
        ):
        """
        Instantiates `ContextualLinearBanditDiag`

        :param n_features: An `int` of the number of features in the parameter space
        :param n_hidden_dim: An `int` of the number of hidden nodes in the NN model
        :param param_space_bounds: A `list` of `tuple` (min, max) of the parameter space
        :param _lambda: A `float` of the regularization parameter
        :param nu: A `float` to control the variance
        :param style: A `str` of the style to use in the solver
        :param learning_rate: A `float` of the learning rate to use for optimization
        :param update_exploration: A `bool` to reduce the exploration during optimization 
        :param ctx_features: An `int` of the number of features in the context
        """        
        self.ctx_features = ctx_features
        self.every_step_decision = kwargs.get("every-step-decision", False)
        self.pretrained_opts = kwargs.get("pretrained_opts", {"use" : False})
        self.teacher_opts = kwargs.get("teacher_opts", {"use" : False})

        super(ContextualLinearBanditDiag, self).__init__(
            n_features, n_hidden_dim, param_space_bounds, _lambda, nu, style,
            learning_rate, update_exploration, *args, **kwargs
        )

        self.histories = []

    def reset(self):
        """
        Resets the model
        """
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
        :param history: A `dict` of the current history
        :param weights: (optional) A `numpy.ndarray` of the weights of each observations
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
        history["ctx"] = numpy.array(history["ctx"])[:, 0] # Keeps first channel image
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
        history = to_tensor(history)
        histories = to_tensor(histories)
        if torch.cuda.is_available():
            X = X.cuda()
            history = to_cuda(history)
            histories = to_cuda(histories)

        if self.update_gradient:
            self.add_gradient(X, history)

        # optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # length = len(histories)
        # index = numpy.arange(length)
        # cnt = 0
        # tot_loss = 0
        # while True:
        #     batch_loss = 0
        #     numpy.random.shuffle(index)
        #     for idx in index:
        #         X, y, ctx, history = histories[idx]
        #         optimizer.zero_grad()
        #         delta = self.model(X, history) - y
        #         loss = delta * delta # * weights[idx]
        #         loss.backward()
        #         optimizer.step()
        #         batch_loss += loss.item()
        #         tot_loss += loss.item()
        #         cnt += 1
        #         if cnt >= 1000:
        #             return tot_loss / 1000
        #     if batch_loss / length <= 1e-3:
        #         return batch_loss / length

        if self.teacher_opts["use"]:
            model = self.teacher_model
            model.train()
        else:
            model = self.model

        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        length = len(histories)
        index = numpy.arange(length)
        cnt = 0
        tot_loss = 0
        flag = False
        while not flag:
            batch_loss = 0
            numpy.random.shuffle(index)
            for idx in index:
                X, y, ctx, history = histories[idx]
                optimizer.zero_grad()
                delta = model(X, history) - y
                loss = delta * delta # * weights[idx]
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    flag = True
                    break
                    # return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                flag = True
                # return batch_loss / length

            if flag:
                break

        # Updates the model with the teacher model using a moving average
        def _soft_update(
            q_network_1: nn.Module,
            q_network_2: nn.Module,
            alpha: float
        ) -> None:
            """In-place, soft-update of q_network_1 parameters with parameters from q_network_2."""
            for p1, p2 in zip(q_network_1.parameters(), q_network_2.parameters()):
                p1.data.copy_(alpha * p2.data + (1 - alpha) * p1.data)

        if self.teacher_opts["use"]:
            _soft_update(self.model, self.teacher_model, alpha=self.teacher_opts["alpha"])

        return tot_loss / cnt

    def add_gradient(self, X, history):
        """
        Calculate the gradient on sample X and add it to the U matrix

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param history: A `dict` of the current history
        """
        y = self.model(X, history)
        fx = y[-1]
        self.model.zero_grad()
        fx.backward(retain_graph=True)
        g = self.get_gradient()
        self.U += g * g

    def train_batch(self, X, y, history, *args, **kwargs):
        """
        Trains the model using a batch of observations

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param y: A `numpy.ndarray` of observed rewards
        :param history: A `dict` of the current history
        """
        self.model.train()

        self.histories = []

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
                history["X"] = rescale_X(history["X"], self.param_space_bounds)
                history["y"] = self.scaler.fit_transform(history["y"])[..., [self.idx]]
                history["ctx"] = history["ctx"]
                for key, value in history.items():
                    history[key] = value.to(torch.float32)

        self.histories.append(history)

        histories = []
        for history_ in self.histories:
            if self.every_step_decision:
                for i in range(len(history["X"])):
                    histories.append([
                        history_["X"][:, [i]], # Keeps a [1, N] shape
                        history_["y"][:, [i]], # Keeps a [1, 1] shape
                        history_["ctx"][[i]],
                        {
                            "X" : history_["X"][:, :i],
                            "y" : history_["y"][:, :i],
                            "ctx" : history_["ctx"][:, :i + 1] # Uses the first context
                        }
                    ])
            else:
                histories.append([
                    history_["X"][:, [0]], # Keeps a [1, N] shape
                    history_["y"][:, [-1]], # Keeps a [1, 1] shape; Only last reward is kept
                    history_["ctx"][:, [0]],
                    {
                        "X" : history_["X"][:, [0]],
                        "y" : history_["y"][:, [-1]],
                        "ctx" : history_["ctx"][:, [0]]
                    }
                ])

        # Convert X, y to torch
        if torch.cuda.is_available():
            X = to_cuda(X)
            history = to_cuda(history)
            histories = to_cuda(histories)

        # if self.update_gradient:
        #     self.add_gradient(X, history)

        criterion = kwargs.get("criterion", nn.MSELoss())
        optimizer = kwargs.get("optimizer", optim.SGD(self.model.parameters(), lr=self.learning_rate))
        length = len(histories)
        index = numpy.arange(length)

        numpy.random.shuffle(index)
        batch_loss = 0
        for idx in index:
            X, y, ctx, history = histories[idx]
            optimizer.zero_grad()
            pred = self.model(X, history)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

        return batch_loss

    def predict_batch(self, X, y, history, *args, **kwargs):
        """
        Predicts a batch of observations

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param y: A `numpy.ndarray` of observed rewards
        :param history: A `dict` of the current history
        """        
        self.model.eval()

        self.histories = []

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
                history["X"] = rescale_X(history["X"], self.param_space_bounds)
                history["y"] = self.scaler.fit_transform(history["y"])[..., [self.idx]]
                history["ctx"] = history["ctx"]
                for key, value in history.items():
                    history[key] = value.to(torch.float32)

        self.histories.append(history)

        histories = []
        for history_ in self.histories:
            if self.every_step_decision:
                for i in range(len(history["X"])):
                    histories.append([
                        history_["X"][:, [i]], # Keeps a [1, N] shape
                        history_["y"][:, [i]], # Keeps a [1, 1] shape
                        history_["ctx"][[i]],
                        {
                            "X" : history_["X"][:, :i],
                            "y" : history_["y"][:, :i],
                            "ctx" : history_["ctx"][:, :i + 1] # Uses the first context
                        }
                    ])
            else:
                histories.append([
                    history_["X"][:, [0]], # Keeps a [1, N] shape
                    history_["y"][:, [-1]], # Keeps a [1, 1] shape; Only last reward is kept
                    history_["ctx"][:, [0]],
                    {
                        "X" : history_["X"][:, [0]],
                        "y" : history_["y"][:, [-1]],
                        "ctx" : history_["ctx"][:, [0]]
                    }
                ])

        # Convert X, y to torch
        if torch.cuda.is_available():
            X = to_cuda(X)
            history = to_cuda(history)
            histories = to_cuda(histories)

        # if self.update_gradient:
        #     self.add_gradient(X, history)

        criterion = kwargs.get("criterion", nn.MSELoss())
        length = len(histories)
        index = numpy.arange(length)

        numpy.random.shuffle(index)
        batch_loss = 0
        for idx in index:
            X, y, ctx, history = histories[idx]
            pred = self.model(X, history)
            loss = criterion(pred, y)

            batch_loss += loss.item()

        return batch_loss

    def add_gradient_batch(self, X, y, history, *args, **kwargs):
        """
        Updates the gradient of the model using a batch of observations

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param y: A `numpy.ndarray` of observed rewards
        :param history: A `dict` of the current history
        """

        self.model.eval()

        self.histories = []

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
                history["X"] = rescale_X(history["X"], self.param_space_bounds)
                history["y"] = self.scaler.fit_transform(history["y"])[..., [self.idx]]
                history["ctx"] = history["ctx"]
                for key, value in history.items():
                    history[key] = value.to(torch.float32)

        self.histories.append(history)

        histories = []
        for history_ in self.histories:
            if self.every_step_decision:
                for i in range(len(history["X"])):
                    histories.append([
                        history_["X"][:, [i]], # Keeps a [1, N] shape
                        history_["y"][:, [i]], # Keeps a [1, 1] shape
                        history_["ctx"][[i]],
                        {
                            "X" : history_["X"][:, :i],
                            "y" : history_["y"][:, :i],
                            "ctx" : history_["ctx"][:, :i + 1] # Uses the first context
                        }
                    ])
            else:
                histories.append([
                    history_["X"][:, [0]], # Keeps a [1, N] shape
                    history_["y"][:, [-1]], # Keeps a [1, 1] shape; Only last reward is kept
                    history_["ctx"][:, [0]],
                    {
                        "X" : history_["X"][:, [0]],
                        "y" : history_["y"][:, [-1]],
                        "ctx" : history_["ctx"][:, [0]]
                    }
                ])

        # Convert X, y to torch
        if torch.cuda.is_available():
            X = to_cuda(X)
            history = to_cuda(history)
            histories = to_cuda(histories)

        length = len(histories)
        index = numpy.arange(length)

        numpy.random.shuffle(index)
        batch_loss = 0
        for idx in index:
            X, y, ctx, history = histories[idx]
            y = self.model(X, history)

            fx = y[-1]
            self.model.zero_grad()
            fx.backward(retain_graph=True)
            g = self.get_gradient()
            self.U += g * g

    def get_mean(self, X, history):
        """
        Predicts mean at the given points

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param history: A `dict` of the current history

        :returns: A `numpy.ndarray` of the mean at X
        """
        self.model.eval()

        history = copy.deepcopy(history)
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
            history["X"] = rescale_X(numpy.concatenate(history["X"], axis=1).T, self.param_space_bounds)
        X = torch.from_numpy(X).float()
        history = to_tensor(history)
        if torch.cuda.is_available():
            X = X.cuda()
            for key, value in history.items():
                value = torch.from_numpy(numpy.array(value)).float()
                history[key] = value.cuda()
        y = self.model(X, history)

        return self.scaler.inverse_transform(y.cpu().data.numpy())

    def get_mean_std(self, X, history):
        """
        Predicts mean and standard deviation at the given points

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param history: A `dict` of the current history

        :returns: A `numpy.ndarray` of the mean at X
                  A `numpy.ndarray` of the std at X
        """
        self.model.eval()

        history = copy.deepcopy(history)
        if self.param_space_bounds is not None:
            X = rescale_X(X, self.param_space_bounds)
            history["X"] = rescale_X(numpy.concatenate(history["X"], axis=1).T, self.param_space_bounds)
        X = torch.from_numpy(X).float()
        history = to_tensor(history)
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
            g = self.get_gradient()
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
        :param history: A `dict` of the current history
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
        history["ctx"] = numpy.array(history["ctx"])[:, 0] # Keeps first channel image

        X = torch.from_numpy(X).float()
        history = to_tensor(history)
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
                sample_r = rng.normal(loc=fx.item(), scale=sigma.item() * kwargs.get("scale", 1))
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

class ContextualNeuralTS(ContextualLinearBanditDiag):
    """
    Implements a `ContextualNeuralTS` solver. This solver automatically learns the
    features to extract using a NN model.

    This model relies on contextual information in the form of a vector.    
    """
    def __init__(self, *args, **kwargs):
        super(ContextualNeuralTS, self).__init__(*args, **kwargs)

    def get_gradient(self):
        """
        Computes the gradient of the model
        """
        g = torch.cat([p.grad.flatten().detach() for key, p in self.model.named_parameters() if (p.requires_grad)])
        return g

    def reset(self):
        """
        Resets the model
        """
        self.histories = []
        self.model = ContextLinearModel(self.n_features + self.ctx_features, self.n_hidden_dim, every_step_decision=self.every_step_decision)
        self.total_param = sum(p.numel() for key, p in self.model.named_parameters() if (p.requires_grad))
        self.U = self._lambda * torch.ones((self.total_param,))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.U = self.U.cuda()    

class ContextualImageLinearBanditDiag(ContextualLinearBanditDiag):
    """
    Implements a `ContextualImageLinearBanditDiag` solver. This solver automatically learns the
    features to extract using a NN model.

    This model relies on contextual information in the form of an image.
    """    
    def __init__(self, *args, **kwargs):

        self.datamap_opts = kwargs.get("datamap_opts", {"shape": 64})
        self.pretrained_opts = kwargs.get("pretrained_opts", {"use" : False})
        self.teacher_opts = kwargs.get("teacher_opts", {"use" : False})

        super(ContextualImageLinearBanditDiag, self).__init__(*args, **kwargs)

    def reset(self):
        """
        Resets the model
        """
        self.histories = []
        self.model = ImageContextLinearModel(
            self.n_features, self.datamap_opts["shape"], self.n_hidden_dim,
            every_step_decision=self.every_step_decision, pretrained_opts=self.pretrained_opts
        )
        if self.teacher_opts["use"]:
            self.teacher_model = ImageContextLinearModel(
                self.n_features, self.datamap_opts["shape"], self.n_hidden_dim,
                every_step_decision=self.every_step_decision, pretrained_opts=self.pretrained_opts
            )
            if torch.cuda.is_available():
                self.teacher_model = self.teacher_model.cuda()

        self.total_param = sum(p.numel() for key, p in self.model.named_parameters() if (p.requires_grad) and ("linear" in key))
        self.U = self._lambda * torch.ones((self.total_param,))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.U = self.U.cuda()


class ContextualImageNeuralTS(ContextualNeuralTS):
    """
    Implements a `ContextualImageNeuralTS` solver. This solver automatically learns the
    features to extract using a NN model.

    This model relies on contextual information in the form of an image.
    """      
    def __init__(self, *args, **kwargs):

        self.datamap_opts = kwargs.get("datamap_opts", {"shape": 64})
        self.pretrained_opts = kwargs.get("pretrained_opts", {"use" : False})
        self.teacher_opts = kwargs.get("teacher_opts", {"use" : False})

        super(ContextualImageNeuralTS, self).__init__(*args, **kwargs)

    def reset(self):
        """
        Resets the model
        """
        self.histories = []
        self.model = ImageContextLinearModel(
            self.n_features, self.datamap_opts["shape"], self.n_hidden_dim,
            every_step_decision=self.every_step_decision, pretrained_opts=self.pretrained_opts,
            full_gradient=True
        )
        if self.teacher_opts["use"]:
            self.teacher_model = ImageContextLinearModel(
                self.n_features, self.datamap_opts["shape"], self.n_hidden_dim,
                every_step_decision=self.every_step_decision, pretrained_opts=self.pretrained_opts,
                full_gradient=True                
            )
            if torch.cuda.is_available():
                self.teacher_model = self.teacher_model.cuda()

        self.total_param = sum(p.numel() for key, p in self.model.named_parameters() if (p.requires_grad))
        self.U = self._lambda * torch.ones((self.total_param,))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.U = self.U.cuda()


class LSTMLinearBanditDiag(LinearBanditDiag):
    """
    Implements a `LSTMLinearBanditDiag` solver. This solver automatically learns the
    features to extract using a NN model.

    This model relies on a model that implements an LSTM backbone
    """
    def __init__(
            self, n_features, n_hidden_dim=32, param_space_bounds=None,
            _lambda=1, nu=1, style="TS", learning_rate=1e-2, update_exploration=False,
            *args, **kwargs
        ):
        """
        Instantiates `ContextualLinearBanditDiag`

        :param n_features: An `int` of the number of features in the parameter space
        :param n_hidden_dim: An `int` of the number of hidden nodes in the NN model
        :param param_space_bounds: A `list` of `tuple` (min, max) of the parameter space
        :param _lambda: A `float` of the regularization parameter
        :param nu: A `float` to control the variance
        :param style: A `str` of the style to use in the solver
        :param learning_rate: A `float` of the learning rate to use for optimization
        :param update_exploration: A `bool` to reduce the exploration during optimization 
        """                
        super(LSTMLinearBanditDiag, self).__init__(
            n_features, n_hidden_dim, param_space_bounds, _lambda, nu, style,
            learning_rate, update_exploration, *args, **kwargs
        )

        self.histories = []
        self.idx = kwargs.get("idx")

    def reset(self):
        """
        Resets the model
        """
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
        :param history: A `dict` of the current history
        :param weights: (optional) A `numpy.ndarray` of the weights of each observations
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

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param history: A `dict` of the current history
        """
        y = self.model(X, history)
        fx = y[-1]
        self.model.zero_grad()
        fx.backward(retain_graph=True)
        g = self.get_gradient()
        self.U += g * g

    def get_mean(self, X, history):
        """
        Predicts mean at the given points

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param history: A `dict` of the current history

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

    def get_mean_std(self, X, history):
        """
        Predicts mean and standard deviation at the given points

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param history: A `dict` of the current history

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
            g = self.get_gradient()
            # sigma2 = self._lambda * self.nu * g * g / self.U
            sigma2 = self._lambda * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            sampled.append(sigma.item())

        std = self.scaler.inverse_transform(numpy.array(sampled)[:, numpy.newaxis], std=True)
        return self.scaler.inverse_transform(y.cpu().data.numpy()), std

    def sample(self, X, history, seed=None, *args, **kwargs):
        """
        Samples the function at points X

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param history: A `dict` of the current history
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
                sample_r = rng.normal(loc=fx.item(), scale=sigma.item() * kwargs.get("scale", 1.))
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
    """
    This class relies on regressor class to generate options to present to the user
    using a Thompson Sampling approach.

    :param regressor: A regression model
    """
    def __init__(self, regressor):
        self.regressor = regressor
        self.X = None
        self.y = None
        self.weights = None
        self.loaded = False

    def predict(self, X_pred, *args, **kwargs):
        """
        Predict mean and standard deviation at given points *X_pred*.

        :param X_pred: A `numpy.ndarray` of points with shape (N, features)

        :returns: An array of means and an array of standard deviations.
        """
        if (self.X is not None) or self.loaded:
            return self.regressor.get_mean_std(X_pred, *args, **kwargs)
        else :
            mean = np.full(X_pred.shape[0], 0)
            std = np.full(X_pred.shape[0], 1)
            return mean, std

    def sample(self, X_sample, seed=None, *args, **kwargs):
        """
        Sample a function evaluated at points *X_sample*. When no points have
        been observed yet, the function values are sampled uniformly between 0 and 1.

        :param X_sample: A `numpy.ndarray` of points with shape (N, features)
        :param seed: (optional) An `int` of the random seed

        :returns: A `numpy.ndarray` of the pointwise evaluation of a sampled function.
        """
        if (self.X is not None) or (self.loaded):
            return self.regressor.sample(X_sample, seed=seed, *args, **kwargs)
        else:
            f_tilde = np.random.uniform(0,1,X_sample.shape[0])[:,np.newaxis]
        return f_tilde

    def update(self, action, reward, weights=None, update_posterior=True, *args, **kwargs):
        """
        Update the regression model using the observations *reward* acquired at
        location *action*.

        :param X: A `numpy.ndarray` of points with shape (N, features)
        :param y: A `numpy.ndarray` of observed rewards
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

    def train_batch(self, batch_action, batch_reward, *args, **kwargs):
        """
        Update the regression model using the observations *reward* acquired at
        location *action*.

        :param batch_action: A `numpy.ndarray` of points with shape (B, N, features)
        :param batch_reward: A `numpy.ndarray` of observed rewards        
        """
        return self.regressor.train_batch(batch_action, batch_reward, *args, **kwargs)

    def predict_batch(self, batch_action, batch_reward, *args, **kwargs):
        """
        Predicts the observations at location *action*.

        :param batch_action: A `numpy.ndarray` of points with shape (B, N, features)
        :param batch_reward: A `numpy.ndarray` of observed rewards        
        """
        return self.regressor.predict_batch(batch_action, batch_reward, *args, **kwargs)

    def add_gradient_batch(self, batch_action, batch_reward, *args, **kwargs):
        """
        Calculate the gradient on sample X and add it to the U matrix
        
        :param batch_action: A `numpy.ndarray` of points with shape (B, N, features)
        :param batch_reward: A `numpy.ndarray` of observed rewards         
        """
        return self.regressor.add_gradient_batch(batch_action, batch_reward, *args, **kwargs)

    def update_params(self, **kwargs):
        """
        Update the regressor parameters
        """
        self.regressor.update_params(**kwargs)

    def set_sampling_mode(self, is_sampling):
        """
        Activates/Deactivates the sampling mode of the model

        :param is_sampling: A `boolean` if the sampling mode is activated
        """
        self.regressor.set_sampling_mode(is_sampling)

    def save_ckpt(self, path, *args, **kwargs):
        """
        Saves a checkpoint of the model

        :param path: A `str` of the model path
        """
        self.regressor.save_ckpt(path=path, *args, **kwargs)

    def load_ckpt(self, path, *args, **kwargs):
        """
        Loads a checkpoint of the model

        :param path: A `str` of the model path
        """
        self.loaded = True
        self.regressor.load_ckpt(path=path, *args, **kwargs)
