#This file is largely based on:
#https://github.com/PDKlab/STED-Optimization/blob/master/src/algorithms.py
"""This module contains classes to generate options presented to the user in the online
multi-objective bandits optimization problem.
"""

import numpy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures
import sklearn
from sklearn.linear_model import BayesianRidge
import numpy as np




class sklearn_GP(GaussianProcessRegressor):
    """This class is meant to be used as a the regressor argument of the TS_sampler
    class. It uses a scikit-learn implementation of Gaussian process regression. The
    regularization is fixed (no adaptative regularization).
    """

    def __init__(self, cte, length_scale, noise_level, alpha, normalize_y=True ):
#        kernel=cte * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
#        super().__init__(kernel=kernel, alpha=alpha, normalize_y=normalize_y, optimizer=optimizer, )
#        lambda_ =
#        self.lambda = s_ub**2/norm_bound**2
        norm_bound = np.sqrt(cte)
        super().__init__(RBF(length_scale=length_scale), alpha=noise_level/norm_bound**2, optimizer=None, normalize_y=normalize_y)
        self.noise_var = noise_level
        self.norm_bound = norm_bound
        
    def update(self, X, y):
        self.fit(X,y)
    
    def get_mean_std(self, X):
        mean, sqrt_k = self.predict(X, return_std=True)
#        std = np.sqrt(std**2 - self.noise_var)
#        mean, sqrt_k = gp.predict(X_pred, return_std=True)
#        std = self.s_ub / numpy.sqrt(self.lambda_) * sqrt_k
        std = self.norm_bound * sqrt_k
        return mean, std
    
    def sample(self, X):
        mean, k = self.predict(X, return_cov=True)
        cov = self.norm_bound**2  * k
        rng = numpy.random.default_rng()
        f_tilde = rng.multivariate_normal(mean.flatten(), cov, method='eigh')[:,np.newaxis]
        return f_tilde


class sklearn_BayesRidge(BayesianRidge):
    """This class is meant to be used as a the regressor argument of the TS_sampler
    class. It uses a scikit-learn implementation of bayesian linear regression to fit a
    polynomial of a certain degree. fit_intercept=True should be used.

    :param degree: degree of the polynomial
    :param other parameters: see sklearn.linear_model.BayesianRidge documentation
    """
    def __init__(self, degree, param_space_bounds=None,
                 tol=1e-6, fit_intercept=True,
                 compute_score=True,alpha_init=None,
                 lambda_init=None, 
                 alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06):
        super().__init__(tol=tol, fit_intercept=fit_intercept,
                         compute_score=compute_score, alpha_init=alpha_init,
                         lambda_init=lambda_init,
                        alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2)
        self.degree=degree
        self.param_space_bounds=param_space_bounds

        
    def update(self, X, y):
        """Update the regression model using the observations *y* acquired at
        locations *X*.
        
        :param action: A 2d array of locations.
        :param reward: A 1-D array of observations.
        """
        if self.param_space_bounds is not None:
            for col in range(X.shape[1]):
                xmin, xmax = self.param_space_bounds[col]
                xmean = (xmax+xmin)/2
                X[:,col] = (X[:,col] - xmean)/(xmax - xmin)
        
        
        X = PolynomialFeatures(self.degree).fit_transform(X)[:,1:]
        self.fit(X,y.flatten())
    
    def get_mean_std(self, X):
        """Predict mean and standard deviation at given points *X*.

        :param : A 2d array of locations at which to predict.
        :returns: An array of means and an array of standard deviations.
        """
        if self.param_space_bounds is not None:
            for col in range(X.shape[1]):
                xmin, xmax = self.param_space_bounds[col]
                xmean = (xmax+xmin)/2
                X[:,col] = (X[:,col] - xmean)/(xmax - xmin)

        
        
        X = PolynomialFeatures(self.degree).fit_transform(X)[:,1:]
        mean, std = self.predict(X, return_std=True)
        std = np.sqrt(std**2 - (1/self.alpha_))
        return mean, std
    
    def sample(self, X):
        """Sample a function evaluated at points *X*.

        :param X: A 2d array locations at which to evaluate the sampled function.
        :returns: A 1-D array of the pointwise evaluation of a sampled function.
        """
        print(np.min(X, axis=0), np.max(X, axis=0), np.mean(X, axis=0))
        if self.param_space_bounds is not None:
            for col in range(X.shape[1]):
                xmin, xmax = self.param_space_bounds[col]
                xmean = (xmax+xmin)/2
                X[:,col] = (X[:,col] - xmean)/(xmax - xmean)
        print(np.min(X, axis=0), np.max(X, axis=0), np.mean(X, axis=0))
        
        
        rng = np.random.default_rng()
#        weigths = self.coef_
#        weigths[0] = self.intercept_
        w_sample = np.random.multivariate_normal(self.coef_, self.sigma_)
        X = PolynomialFeatures(self.degree).fit_transform(X)[:,1:]
        return X@w_sample[:,np.newaxis] + self.intercept_


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
        
    def predict(self, X_pred):
        """Predict mean and standard deviation at given points *X_pred*.

        :param X_pred: A 2d array of locations at which to predict.
        :returns: An array of means and an array of standard deviations.
        """
        if self.X is not None:
            return self.regressor.get_mean_std(X_pred)
        else :
            mean = np.full(X_pred.shape[0], 0)
            std = np.full(X_pred.shape[0], 1)
            return mean, std
        
    def sample(self, X_sample):
        """Sample a function evaluated at points *X_sample*. When no points have
        been observed yet, the function values are sampled uniformly between 0 and 1.

        :param X_sample: A 2d array locations at which to evaluate the sampled function.
        :returns: A 1-D array of the pointwise evaluation of a sampled function.
        """
        if self.X is not None:
            return self.regressor.sample(X_sample)
        else:
#             mean= np.full(X_sample.shape[0], 0)
#             cov = np.identity(X_sample.shape[0])
#             rng = np.random.default_rng()
#             f_tilde = rng.multivariate_normal(mean, cov, method='eigh')
              f_tilde = np.random.uniform(0,1,X_sample.shape[0])[:,np.newaxis]
        return f_tilde


    def update(self, action, reward):
        """Update the regression model using the observations *reward* acquired at
        location *action*.
        
        :param action: A 2d array of locations.
        :param reward: A 1-D array of observations.
        """
        if self.X is not None:
            self.X = np.append(self.X, action, axis=0)
            self.y = np.append(self.y, reward, axis=0)
        else:
            self.X = action
            self.y = reward
        self.regressor.update(self.X, self.y)
