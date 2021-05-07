#This file is from:
#https://github.com/PDKlab/STED-Optimization/blob/master/src/algorithms.py
"""This module contains classes to generate options presented to the user in the online
multi-objective kernelized bandits optimization problem.

.. [Durand2018] Durand, Maillard and Pineau (2018). Streaming kernel regression with
   provably adaptive mean, variance, and regularization. *JMLR*
.. [Williams2006] Williams and Rasmussen (2006). Gaussian processes for machine learning.
   *The MIT Press*
"""

import numpy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures
import sklearn
from sklearn.linear_model import BayesianRidge
import numpy as np


class sklearn_GP(GaussianProcessRegressor):
    def __init__(self, cte, length_scale, noise_levels, alpha, optimizer=None, ):
        kernel=cte * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_levels)
        super().__init__(kernel=kernel, alpha=alpha, optimizer=optimizer,)
        
    def update(self, X, y):
        self.fit(X,y)
    
    def get_mean_std(self, X):
        return self.predict(X, return_std=True)
    
    def sample(self, X):
        mean, cov = self.predict(X, return_cov=True)
        rng = numpy.random.default_rng()
        f_tilde = rng.multivariate_normal(mean, cov, method='eigh')[:,np.newaxis]
        return f_tilde


class sklearn_BayesRidge(BayesianRidge):
    def __init__(self, degree,
                 tol=1e-6, fit_intercept=False,
                 compute_score=True,alpha_init=None,
                 lambda_init=None, 
                 alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06):
        super().__init__(tol=tol, fit_intercept=fit_intercept,
                         compute_score=compute_score, alpha_init=alpha_init,
                         lambda_init=lambda_init,
                        alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2)
        self.degree=degree

        
    def update(self, X, y):
        X = PolynomialFeatures(self.degree).fit_transform(X)
        self.fit(X,y.flatten())
    
    def get_mean_std(self, X):
        X = PolynomialFeatures(self.degree).fit_transform(X)
        
        return self.predict(X, return_std=True)
    
    def sample(self, X):
        rng = np.random.default_rng()
        w_sample = np.random.multivariate_normal(self.coef_, self.sigma_)
        X = PolynomialFeatures(self.degree).fit_transform(X)
        return X@w_sample[:,np.newaxis]


class TS_sampler():
    def __init__(self, regressor):
        self.regressor = regressor
        self.X = None
        self.y = None
        
    def predict(self, X_pred):
        if self.X is not None:
            return self.regressor.get_mean_std(X_pred)
        else :
            mean = np.full(X_pred.shape[0], 0)
            std = np.full(X_pred.shape[0], 1)
            return mean, std
        
    def sample(self, X_sample):
        if self.X is not None:
            return self.regressor.sample(X_sample)
        else:
#             mean= np.full(X_sample.shape[0], 0)
#             cov = np.identity(X_sample.shape[0])
#             rng = np.random.default_rng()
#             f_tilde = rng.multivariate_normal(mean, cov, method='eigh')
              f_tilde = np.random.uniform(0,1,X_sample.shape[0])  
        return f_tilde
        
    def update(self, action, reward):
        if self.X is not None:
            self.X = np.append(self.X, action, axis=0)
            self.y = np.append(self.y, reward, axis=0)
        else:
            self.X = action
            self.y = reward
        self.regressor.update(self.X, self.y)



class custom_GP_TS:
    """This class relies on kernel regression to generate options to present to the user
    using a Thompson Sampling approach. It relies on the Gaussian process implementation
    from :mod:`sklearn` with a fixed RBF kernel [Williams2006]_ and maintain empirical
    confidence interval on the noise standard deviation :math:`\sigma` [Durand2018]_.

    :param bandwidth: The bandwidth of the RBF kernel.
    :param s_lb: An initial lower bound on :math:`\sigma`.
    :param s_ub: An initial upper bound on :math:`\sigma`.
    """
    def __init__(self, kernel_str ,alpha):
    
        kernel = eval(kernel_str)
        gp = GaussianProcessRegressor(kernel=kernel,
                                      optimizer=None,
                                      alpha=alpha)
        self.gp = gp
        self.X=None
        self.y=None

    def predict(self, X_pred, return_cov=False):
        """Predict mean and standard deviation at given points *X_pred*.

        :param X_pred: A 2d array of locations at which to predict.
        :returns: An array of means and an array of standard deviations.
        """
        if self.X is not None:
            self.gp = self.gp.fit(self.X, self.y)
            mean, std = self.gp.predict(X_pred, return_std=True)
        else:
            mean = numpy.full(X_pred.shape[0], 0)
            std = numpy.full(X_pred.shape[0], 0.3) #The std does not matter since the mean is zero everywhere
        return mean, std

    def sample(self, X_sample):
        """Sample a function evaluated at points *X_sample*.

        :param X_sample: A 2d array locations at which to evaluate the sampled function.
        :returns: A 1-D of the pointwise evaluation of a sampled function.
        """
        if self.X is not None:
            self.gp = self.gp.fit(self.X, self.y)
            mean, cov = self.gp.predict(X_sample, return_cov=True)
        else:
            mean= numpy.full(X_sample.shape[0], 0)
            cov = 0.5 * numpy.identity(X_sample.shape[0]) #The std does not matter since the mean is zero everywhere
        rng = numpy.random.default_rng()
        f_tilde = rng.multivariate_normal(mean, cov, 1, method='eigh')[0]
#        f_tilde = numpy.random.multivariate_normal(mean, cov, 1, method='eigh')[0]
        return f_tilde

    def update(self, action, reward, *args):
        """Update the kernel regression model using the observations *reward* acquired at
        location *action*. Estimate upper and lower bounds on the noise variance using
        :func:`estimate_noise` with confidence :math:`\delta=0.1`.
        
        :param action: A 2d array of locations.
        :param reward: A 1-D array of observations.
        :param `*args`: Dummy parameter to handle functions of inheritated classes.
        """
        if self.X is None:
            self.X = numpy.asarray(action)
            self.y = numpy.asarray(reward)
        else:
            self.X = numpy.r_[self.X, action]
            self.y = numpy.r_[self.y, reward]

class linear_regression():
    #TODO: Test validity of the regression method
    #TODO: Add a bayesian linear regression option?
    def __init__(self,  noise, alpha=None):
        self.alpha_ = alpha
        self.noise_ = noise
    def fit(self, X, y):
        X = np.concatenate([X, np.ones(X.shape[0])[:, np.newaxis]], axis=1)
        if self.alpha_:
            A = 1/self.noise_**2 * (X.T@X + self.alpha_*np.ones((X.shape[1], X.shape[1])) )
        else:
            A = 1/self.noise_**2*X.T@X
        self.weights_cov_ = np.linalg.inv(A)
        self.weights_ = 1/self.noise_**2*self.weights_cov_@X.T@y
        print(np.linalg.inv(X.T@X+self.alpha_*np.ones((X.shape[1], X.shape[1])))@X.T@y)
    def predict(self, X):
        X = np.concatenate([X, np.ones(X.shape[0])], axis=1)
        mean = X@self.weights_
        cov = X@self.weights_cov_@X.T
        return mean, cov
    def sample(X):
        X = np.concatenate([X, np.ones(X.shape[0])], axis=1)
        sampled_weigths = np.random.multivariate_normal(self.weights, self.weights_cov_)
        return X@sampled_weigths


class poly_reg_TS():
    def __init__(self, linear_model, degree):
        self.linear_model = linear_model
        self.degree  = degree
        self.poly_features_maker = PolynomialFeatures(degree).fit_transform
        self.X = None
        self.y = None
        
    def predict(self, X_pred):
        X_pred = self.poly_features_maker(X_pred)
        mean, cov = linear_model.predict(X_pred)
        std = np.sqrt(cov.diagonal())
        return mean, std
        
    def sample(self, X_sample):
        X_sample = self.poly_features_maker(X_sample)
        return self.linear_model.sample(X_sample)
        
    def update(self, action, reward):
        if self.X:
            self.X = np.append(self.X, action, axis=0)
            self.y = np.append(self.y, reward, axis=0)
        else:
            self.X = action
            self.y = reward
        self.linear_model.fit(self.poly_features_maker(self.X), self.y)


def estimate_noise(X, y, bandwidth, s_minus, s_plus, norm_bound, delta):
    """Given initial lower and upper bounds on the noise standard deviation :math:`\sigma`, this function
    estimates lower and upper bounds on :math:`\sigma` from previous observations
    obtained using streaming kernel regression [Durand2018]_. The estimated bounds define a
    confidence interval that holds with probability :math:`1-3\delta`. This function relies on the
    Gaussian process implementation from :mod:`sklearn` with a fixed RBF kernel [Williams2006]_.

    :param X: Input points (2d array).
    :param y: Observations (1d array)
    :param bandwidth: The bandwidth of the RBF kernel.
    :param s_minus: Initial lower bound on :math:`\sigma`.
    :param s_plus: Initial upper bound on :math:`\sigma`.
    :param norm_bound: A bound on the norm of the function in the RKHS induced by the
                       RBF kernel of given `bandwidth`.
    :param delta: The confidence :math:`\delta`.
    :returns: Lower and upper bound estimates on :math:`\sigma`.
    """
    lambda_ = s_plus**2 / norm_bound**2
    model = GaussianProcessRegressor(RBF(length_scale=bandwidth), alpha=lambda_, optimizer=None, normalize_y=True)
#    import pdb; pdb.set_trace()
    model.fit(X, y)
    y_hat, sqrt_k = model.predict(X, return_std=True)
    ks = sqrt_k**2
    s_hat = numpy.sqrt(numpy.mean((y - y_hat)**2))

    t = X.shape[0]
    c = numpy.log(numpy.e/delta) * (1 + numpy.log(numpy.pi**2*numpy.log(t)/6) / numpy.log(1/delta))

    e = 1 - 1 / numpy.max(1 + ks / lambda_)
    s_lb = (s_hat - norm_bound * numpy.sqrt(lambda_*e/t)) / (1 + numpy.sqrt(2*c/t))
    if numpy.isnan(s_lb):
        s_lb = s_minus
    else:
        s_lb = max(s_lb, s_minus)

    lambda_star = s_lb**2 / norm_bound**2
    model = GaussianProcessRegressor(RBF(length_scale=bandwidth), alpha=lambda_star, optimizer=None, normalize_y=True)
    model.fit(X, y)
    _, sqrt_k = model.predict(X, return_std=True)
    ks = sqrt_k**2

    d = 2 * numpy.log(1/delta) + numpy.sum(numpy.log(1+ks/lambda_star))
    a = max(1 - numpy.sqrt(c/t) - numpy.sqrt((c+2*d)/t), 1e-10)
    b = norm_bound * numpy.sqrt(lambda_*d) / (2 * t)
    s_ub = (numpy.sqrt(b) + numpy.sqrt(b + s_hat * a))**2 / a**2
    if numpy.isnan(s_ub):
        s_ub = s_plus
    else:
        s_ub = min(s_ub, s_plus)

    return s_lb, s_ub


class Kernel_TS:
    """This class relies on kernel regression to generate options to present to the user
    using a Thompson Sampling approach. It relies on the Gaussian process implementation
    from :mod:`sklearn` with a fixed RBF kernel [Williams2006]_ and maintain empirical
    confidence interval on the noise standard deviation :math:`\sigma` [Durand2018]_.

    :param bandwidth: The bandwidth of the RBF kernel.
    :param s_lb: An initial lower bound on :math:`\sigma`.
    :param s_ub: An initial upper bound on :math:`\sigma`.
    """
    def __init__(self, bandwidth, s_lb, s_ub, C=5):
        self.bandwidth = bandwidth
        self.s_lb = s_lb
        self.s_ub = s_ub
        self.X = None
        self.y = None

#        norm_bound = 5
        norm_bound = C
        self.lambda_ = s_ub**2/norm_bound**2

    def predict(self, X_pred, return_cov=False):
        """Predict mean and standard deviation at given points *X_pred*.

        :param X_pred: A 2d array of locations at which to predict.
        :returns: An array of means and an array of standard deviations.
        """
        if self.X is not None:
            gp = GaussianProcessRegressor(RBF(length_scale=self.bandwidth), alpha=self.lambda_, optimizer=None, normalize_y=True)
            gp.fit(self.X, self.y)
            if return_cov:
                mean, cov = gp.predict(X_pred, return_cov=True)
                cov = self.s_ub**2 / self.lambda_ * cov
                std = numpy.sqrt(numpy.diagonal(cov))
            else:
                mean, sqrt_k = gp.predict(X_pred, return_std=True)
                std = self.s_ub / numpy.sqrt(self.lambda_) * sqrt_k
        else:
            mean = numpy.full(X_pred.shape[0], 0)
            std = numpy.full(X_pred.shape[0], self.s_ub / numpy.sqrt(self.lambda_))
        if return_cov:
            return mean, std, cov
        else:
            return mean, std

    def sample(self, X_sample):
        """Sample a function evaluated at points *X_sample*.

        :param X_sample: A 2d array locations at which to evaluate the sampled function.
        :returns: A 1-D of the pointwise evaluation of a sampled function.
        """
        if self.X is not None:
            gp = GaussianProcessRegressor(RBF(length_scale=self.bandwidth), alpha=self.lambda_,
                                              optimizer=None, normalize_y=True)
            gp.fit(self.X, self.y)
            mean, k = gp.predict(X_sample, return_cov=True)
            cov = self.s_ub**2 / self.lambda_ * k
        else:
            mean= numpy.full(X_sample.shape[0], 0)
            cov = self.s_ub**2 / self.lambda_ * numpy.identity(X_sample.shape[0])
        f_tilde = numpy.random.multivariate_normal(mean, cov, 1)[0]
        return f_tilde

    def update(self, action, reward, *args):
        """Update the kernel regression model using the observations *reward* acquired at
        location *action*. Estimate upper and lower bounds on the noise variance using
        :func:`estimate_noise` with confidence :math:`\delta=0.1`.
        
        :param action: A 2d array of locations.
        :param reward: A 1-D array of observations.
        :param `*args`: Dummy parameter to handle functions of inheritated classes.
        """
        if self.X is None:
            self.X = numpy.asarray(action)
            self.y = numpy.asarray(reward)
        else:
            self.X = numpy.r_[self.X, action]
            self.y = numpy.r_[self.y, reward]

#        norm_bound = 5
#        delta = 0.1
#        s_lb, s_ub = estimate_noise(self.X, self.y, self.bandwidth, self.s_lb, self.s_ub,
#                                    norm_bound, delta)
#        lambda_, lambda_star = s_ub**2/norm_bound**2, s_lb**2/norm_bound**2
#        self.s_lb, self.s_ub, self.lambda_, self.lambda_star = s_lb, s_ub, lambda_, lambda_star


class Kernel_TS_cholesky:
    # cholesky decomposition is used for the sampling process instead of the default svd
    """This class relies on kernel regression to generate options to present to the user
    using a Thompson Sampling approach. It relies on the Gaussian process implementation
    from :mod:`sklearn` with a fixed RBF kernel [Williams2006]_ and maintain empirical
    confidence interval on the noise standard deviation :math:`\sigma` [Durand2018]_.

    :param bandwidth: The bandwidth of the RBF kernel.
    :param s_lb: An initial lower bound on :math:`\sigma`.
    :param s_ub: An initial upper bound on :math:`\sigma`.
    """
    def __init__(self, bandwidth, s_lb, s_ub, C=5):
        self.bandwidth = bandwidth
        self.s_lb = s_lb
        self.s_ub = s_ub
        self.X = None
        self.y = None

#        norm_bound = 5
        norm_bound = C
        self.lambda_ = s_ub**2/norm_bound**2

    def predict(self, X_pred):
        """Predict mean and standard deviation at given points *X_pred*.

        :param X_pred: A 2d array of locations at which to predict.
        :returns: An array of means and an array of standard deviations.
        """
        if self.X is not None:
            gp = GaussianProcessRegressor(RBF(length_scale=self.bandwidth), alpha=self.lambda_, optimizer=None, normalize_y=True)
            gp.fit(self.X, self.y)
            mean, sqrt_k = gp.predict(X_pred, return_std=True)
            std = self.s_ub / numpy.sqrt(self.lambda_) * sqrt_k
        else:
            mean = numpy.full(X_pred.shape[0], 0)
            std = numpy.full(X_pred.shape[0], self.s_ub / numpy.sqrt(self.lambda_))
        return mean, std

    def sample(self, X_sample):
        """Sample a function evaluated at points *X_sample*.

        :param X_sample: A 2d array locations at which to evaluate the sampled function.
        :returns: A 1-D of the pointwise evaluation of a sampled function.
        """
        if self.X is not None:
            gp = GaussianProcessRegressor(RBF(length_scale=self.bandwidth), alpha=self.lambda_,
                                              optimizer=None, normalize_y=True)
            gp.fit(self.X, self.y)
            mean, k = gp.predict(X_sample, return_cov=True)
            cov = self.s_ub**2 / self.lambda_ * k
        else:
            mean= numpy.full(X_sample.shape[0], 0)
            cov = self.s_ub**2 / self.lambda_ * numpy.identity(X_sample.shape[0])
        rng = numpy.random.default_rng()
#         f_tilde = numpy.random.multivariate_normal(mean, cov, 1)[0]
        f_tilde = rng.multivariate_normal(mean, cov, 1, method='cholesky')[0]
        return f_tilde

    def update(self, action, reward, *args):
        """Update the kernel regression model using the observations *reward* acquired at
        location *action*. Estimate upper and lower bounds on the noise variance using
        :func:`estimate_noise` with confidence :math:`\delta=0.1`.
        
        :param action: A 2d array of locations.
        :param reward: A 1-D array of observations.
        :param `*args`: Dummy parameter to handle functions of inheritated classes.
        """
        if self.X is None:
            self.X = numpy.asarray(action)
            self.y = numpy.asarray(reward)
        else:
            self.X = numpy.r_[self.X, action]
            self.y = numpy.r_[self.y, reward]

        norm_bound = 5
        delta = 0.1
        s_lb, s_ub = estimate_noise(self.X, self.y, self.bandwidth, self.s_lb, self.s_ub,
                                    norm_bound, delta)
        lambda_, lambda_star = s_ub**2/norm_bound**2, s_lb**2/norm_bound**2
        self.s_lb, self.s_ub, self.lambda_, self.lambda_star = s_lb, s_ub, lambda_, lambda_star
        
class Kernel_TS_eigh:
    # cholesky decomposition is used for the sampling process instead of the default svd
    """This class relies on kernel regression to generate options to present to the user
    using a Thompson Sampling approach. It relies on the Gaussian process implementation
    from :mod:`sklearn` with a fixed RBF kernel [Williams2006]_ and maintain empirical
    confidence interval on the noise standard deviation :math:`\sigma` [Durand2018]_.

    :param bandwidth: The bandwidth of the RBF kernel.
    :param s_lb: An initial lower bound on :math:`\sigma`.
    :param s_ub: An initial upper bound on :math:`\sigma`.
    """
    def __init__(self, bandwidth, s_lb, s_ub, C=5):
        self.bandwidth = bandwidth
        self.s_lb = s_lb
        self.s_ub = s_ub
        self.X = None
        self.y = None

#        norm_bound = 5
        norm_bound = C
        self.lambda_ = s_ub**2/norm_bound**2

    def predict(self, X_pred):
        """Predict mean and standard deviation at given points *X_pred*.

        :param X_pred: A 2d array of locations at which to predict.
        :returns: An array of means and an array of standard deviations.
        """
        if self.X is not None:
            gp = GaussianProcessRegressor(RBF(length_scale=self.bandwidth), alpha=self.lambda_, optimizer=None, normalize_y=True)
            gp.fit(self.X, self.y)
            mean, sqrt_k = gp.predict(X_pred, return_std=True)
            std = self.s_ub / numpy.sqrt(self.lambda_) * sqrt_k
        else:
            mean = numpy.full(X_pred.shape[0], 0)
            std = numpy.full(X_pred.shape[0], self.s_ub / numpy.sqrt(self.lambda_))
        return mean, std

    def sample(self, X_sample):
        """Sample a function evaluated at points *X_sample*.

        :param X_sample: A 2d array locations at which to evaluate the sampled function.
        :returns: A 1-D of the pointwise evaluation of a sampled function.
        """
        if self.X is not None:
            gp = GaussianProcessRegressor(RBF(length_scale=self.bandwidth), alpha=self.lambda_,
                                              optimizer=None, normalize_y=True)
            gp.fit(self.X, self.y)
            mean, k = gp.predict(X_sample, return_cov=True)
            cov = self.s_ub**2 / self.lambda_ * k
        else:
            mean= numpy.full(X_sample.shape[0], 0)
            cov = self.s_ub**2 / self.lambda_ * numpy.identity(X_sample.shape[0])
        rng = numpy.random.default_rng()
#         f_tilde = numpy.random.multivariate_normal(mean, cov, 1)[0]
        f_tilde = rng.multivariate_normal(mean, cov, 1, method='eigh')[0]
        return f_tilde

    def update(self, action, reward, *args):
        """Update the kernel regression model using the observations *reward* acquired at
        location *action*. Estimate upper and lower bounds on the noise variance using
        :func:`estimate_noise` with confidence :math:`\delta=0.1`.
        
        :param action: A 2d array of locations.
        :param reward: A 1-D array of observations.
        :param `*args`: Dummy parameter to handle functions of inheritated classes.
        """
        if self.X is None:
            self.X = numpy.asarray(action)
            self.y = numpy.asarray(reward)
        else:
            self.X = numpy.r_[self.X, action]
            self.y = numpy.r_[self.y, reward]

        norm_bound = 5
        delta = 0.1
        s_lb, s_ub = estimate_noise(self.X, self.y, self.bandwidth, self.s_lb, self.s_ub,
                                    norm_bound, delta)
        lambda_, lambda_star = s_ub**2/norm_bound**2, s_lb**2/norm_bound**2
        self.s_lb, self.s_ub, self.lambda_, self.lambda_star = s_lb, s_ub, lambda_, lambda_star


class Kernel_TS_PseudoActions(Kernel_TS):
    """This class relies on kernel regression to generate options to present to the user
    using a Thompson Sampling approach. It relies on the Gaussian process implementation
    from :mod:`sklearn` with a fixed RBF kernel [Williams2006]_ and maintain empirical
    confidence interval on the noise standard deviation :math:`\sigma` [Durand2018]_.
    It extends the class :class:`Kernel_TS` to hallucinate pseudo-actions (and associated
    pseudo-rewards). Pseudo-actions are reflected at over the boundaries of the space and
    

    :param bandwidth: The bandwidth of the RBF kernel.
    :param s_lb: An initial lower bound on :math:`\sigma`.
    :param s_ub: An initial upper bound on :math:`\sigma`.
    :param space_bounds: A list of tuple (lower, upper) bounds, bounding the input space in
                         for each dimension.
    """
    def __init__(self, bandwidth, s_lb, s_ub, space_bounds):
        super().__init__(bandwidth, s_lb, s_ub)

        self.space_bounds = space_bounds
        self.pseudo_X = None
        self.pseudo_y = None

    def predict(self, X_pred):
        """Predict mean and standard deviation at given points *X_pred*.

        :param X_pred: A 2d array of locations at which to predict.
        :returns: An array of means and an array of standard deviations.
        """
        if self.pseudo_X is not None:
            gp = GaussianProcessRegressor(RBF(length_scale=self.bandwidth), alpha=self.lambda_,
                                              optimizer=None, normalize_y=True)
            gp.fit(self.pseudo_X, self.pseudo_y)
            mean, sqrt_k = gp.predict(X_pred, return_std=True)
            std = self.s_ub / numpy.sqrt(self.lambda_) * sqrt_k
        else:
            mean = numpy.full(X_pred.shape[0], 0)
            std = numpy.full(X_pred.shape[0], self.s_ub / numpy.sqrt(self.lambda_))
        return mean, std

    def sample(self, X_sample):
        """Sample a function evaluated at points *X_sample*.

        :param X_sample: A 2d array locations at which to evaluate the sampled function.
        :returns: A 1-D of the pointwise evaluation of a sampled function.
        """
        if self.pseudo_X is not None:
            gp = GaussianProcessRegressor(RBF(length_scale=self.bandwidth), alpha=self.lambda_,
                                              optimizer=None, normalize_y=True)
            gp.fit(self.pseudo_X, self.pseudo_y)
            mean, k = gp.predict(X_sample, return_cov=True)
            cov = self.s_ub**2 / self.lambda_ * k
        else:
            mean= numpy.full(X_sample.shape[0], 0)
            cov = self.s_ub**2 / self.lambda_ * numpy.identity(X_sample.shape[0])
        f_tilde = numpy.random.multivariate_normal(mean, cov, 1)[0]
        return f_tilde

    def update(self, actions, rewards, space_bounds=None):
        """Update the kernel regression model using the observations *reward* acquired at
        location *action*. Estimate upper and lower bounds on the noise variance using
        :func:`estimate_noise` with confidence :math:`\delta=0.1`.
        
        :param action: A 2d array of locations.
        :param reward: A 1-D array of observations.
        :param space_bounds: A list of tuple (lower, upper) bounds, bounding the input space in
                             for each dimension (default: None). If None, uses the object attribute
                             :attr:`space_bounds`.
        """
        if self.X is None:
            self.X = numpy.asarray(actions)
            self.y = numpy.asarray(rewards)
            self.pseudo_X = numpy.asarray(actions)
            self.pseudo_y = numpy.asarray(rewards)
        else:
            self.X = numpy.r_[self.X, actions]
            self.y = numpy.r_[self.y, rewards]
            self.pseudo_X = numpy.r_[self.pseudo_X, actions]
            self.pseudo_y = numpy.r_[self.pseudo_y, rewards]

        # add pseudo rewards
        if space_bounds is None: space_bounds = self.space_bounds
        for a, r in zip(actions, rewards):
            for i, (l, u) in enumerate(space_bounds):
                if a[i] == l:
                    pseudo_a = numpy.copy(a)
                    pseudo_a[i] = l - (u - l)
                    self.pseudo_X = numpy.r_[self.pseudo_X, [pseudo_a]]
                    self.pseudo_y = numpy.r_[self.pseudo_y, r]
                elif a[i] == u:
                    pseudo_a = numpy.copy(a)
                    pseudo_a[i] = u + (u - l)
                    self.pseudo_X = numpy.r_[self.pseudo_X, [pseudo_a]]
                    self.pseudo_y = numpy.r_[self.pseudo_y, r]

        norm_bound = 5
        delta = 0.1
        s_lb, s_ub = estimate_noise(self.X, self.y, self.bandwidth, self.s_lb, self.s_ub,
                                    norm_bound, delta)
        lambda_, lambda_star = s_ub**2/norm_bound**2, s_lb**2/norm_bound**2
        self.s_lb, self.s_ub, self.lambda_, self.lambda_star = s_lb, s_ub, lambda_, lambda_star


class linear_regression():
    #TODO: Test validity of the regression method
    #TODO: Add a bayesian linear regression option?
    def __init__(self,  noise, alpha=None):
        self.alpha_ = alpha
        self.noise_ = noise
        self.weights_ = None
    def fit(self, X, y):
        X = np.concatenate([X, np.ones(X.shape[0])[:, np.newaxis]], axis=1)
        if self.alpha_:
            A = 1/self.noise_**2 * (X.T@X + self.alpha_*np.eye(X.shape[1]) )
        else:
            A = 1/self.noise_**2*X.T@X
        self.weights_cov_ = np.linalg.inv(A)
        self.weights_ = 1/self.noise_**2*self.weights_cov_@X.T@y
#        print(np.linalg.inv(X.T@X+self.alpha_*np.ones((X.shape[1], X.shape[1])))@X.T@y)
    def predict(self, X):
        X = np.concatenate([X, np.ones(X.shape[0])[:,np.newaxis]], axis=1)
        mean = X@self.weights_
        cov = X@self.weights_cov_@X.T
        return mean, cov
    def sample(self, X):
            X = np.concatenate([X, np.ones(X.shape[0])[:,np.newaxis]], axis=1)
            if self.weights_ is not None:
                sampled_weigths = np.random.multivariate_normal(self.weights_, self.weights_cov_)
            else:
                sampled_weigths = np.random.multivariate_normal(np.ones(X.shape[1]), np.eye(X.shape[1]))
            return X@sampled_weigths


class poly_reg_TS():
    def __init__(self, linear_model, degree):
        self.linear_model = linear_model
        self.degree  = degree
        self.poly_features_maker = PolynomialFeatures(degree).fit_transform
        self.X = None
        self.y = None
        
    def predict(self, X_pred):
        X_pred = self.poly_features_maker(X_pred)
        mean, cov = self.linear_model.predict(X_pred)
        std = np.sqrt(cov.diagonal())
        return mean, std
        
    def sample(self, X_sample):
        X_sample = self.poly_features_maker(X_sample)
        return self.linear_model.sample(X_sample)
        
    def update(self, action, reward):
        if self.X is not None:
            self.X = np.append(self.X, action, axis=0)
            self.y = np.append(self.y, reward, axis=0)
        else:
            self.X = action
            self.y = reward
        self.linear_model.fit(self.poly_features_maker(self.X), self.y)
        
    
    

        


class poly_reg_TS_discrete():
    def __init__(self, poly_degree, regul_cte, s):
        self.poly_degree = poly_degree
        self.regul_cte = regul_cte
        self.s = s
        self.X = None
        self.X_original = None
        self.y = None
        self.get_poly_features = sklearn.preprocessing.PolynomialFeatures(degree=poly_degree).fit_transform
        self.A = None
    
    def X_for_poly_reg(self, X):
        if type(X) == np.ndarray:
            if len(X.shape) > 1:
                return self.get_poly_features(X)
            else:
                return self.get_poly_features(X[np.newaxis, :])
        else:
            return self.get_poly_features(np.asarray(X))
            
        
        
    def predict(self, X_pred):
        X_pred = self.X_for_poly_reg(X_pred)
        if self.X is not None:
            mean = self.s**-2*X_pred@np.linalg.inv(self.A)@self.X.T@self.y
        else:
            mean = np.zeros(X_pred.shape[0])
            self.A = self.regul_cte*self.s**-2*np.identity(X_pred.shape[1])
        cov = X_pred@np.linalg.inv(self.A)@X_pred.T
        return mean.flatten(), np.sqrt(np.diag(cov)).flatten()
    
    def sample(self, X_sample):
        X_sample = self.X_for_poly_reg(X_sample)
        if self.X is not None:
            sampled_w = np.random.multivariate_normal((self.s**-2*np.linalg.inv(self.A)@self.X.T@self.y).flatten(),\
                                                      np.linalg.inv(self.A))[:,np.newaxis]
        else:
            self.A = self.regul_cte*self.s**-2*np.identity(X_sample.shape[1])
            sampled_w = np.random.multivariate_normal(np.zeros(X_sample.shape[1]), np.linalg.inv(self.A))[:,np.newaxis]
        f_tilde = X_sample@sampled_w
        return f_tilde.flatten()
    
    def update(self, action, reward):
        if self.X is None:
            self.X = self.X_for_poly_reg(action)
            self.X_original = np.asarray(action)
            self.y = np.asarray(reward)[:,np.newaxis]
        else:
            self.X = np.append(self.X, self.X_for_poly_reg(action), axis=0)
            self.X_original = np.concatenate([self.X_original, np.asarray(action)], axis=0)
            self.y = np.append(self.y, np.asarray(reward)[:,np.newaxis], axis=0)
        self.A = (self.s**-2*self.X.T@self.X + self.regul_cte*self.s**-2*np.identity(self.X.shape[1]))




##TODO: in TS_fitted_poly.py, load directly from here
regressor_name_dict = {
    "sklearn_BayesRidge": sklearn_BayesRidge,
    "sklearn_GP": sklearn_GP,
}
