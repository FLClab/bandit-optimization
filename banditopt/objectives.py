
"""This module contains classes that implement several objectives to optimize.
One can define a new objective by inheriting abstract class :class:`Objective`.
"""

from abc import ABC, abstractmethod

import numpy
import itertools
import warnings

from statsmodels.tsa.stattools import acf
from scipy.ndimage import gaussian_filter
from scipy import optimize
from skimage.transform import resize
from skimage.feature import peak_local_max
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error

import scipy.fft
from scipy.optimize import curve_fit

#import fsc
# import src.utils as utils
# import src.user as user

from . import decorr
from . import utils
from . import user
from . import fsc

class Objective(ABC):
    """Abstract class to implement an objective to optimize. When inheriting this class,
    one needs to define an attribute `label` to be used for figure labels, and a
    function :func:`evaluate` to be called during optimization.
    """
    @abstractmethod
    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        """Compute the value of the objective given the result of an acquisition.

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param concofal_end: A confocal image acquired after the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).
        """
        raise NotImplementedError

    def mirror_ticks(self, ticks):
        """Tick values to override the true *tick* values for easier plot understanding.

        :param ticks: Ticks to replace.

        :returns: New ticks or None to keep the same.
        """
        return None


class Signal_Ratio(Objective):
    """Objective corresponding to the signal to noise ratio (SNR) defined by

    .. math::
        \\text{SNR} = \\frac{\\text{STED}_{\\text{fg}}^{75} - \overline{\\text{STED}_{\\text{fg}}}}{\\text{Confocal1}_{\\text{fg}}^{75}}

    where :math:`\\text{image}^q` and :math:`\\overline{\\text{image}}` respectively
    denote the :math:`q`-th percentile signal on an image and the average signal
    on an image, and :math:`\\text{STED}_{\\text{fg}}`, :math:`\\text{Confocal1}_{\\text{fg}}`, and
    :math:`\\text{Confocal2}_{\\text{fg}}` respectively refer to the foreground of the STED image
    and confocal images acquired before and after.

    :param float percentile: :math:`q`-th percentile in :math:`[0,100]`.
    """
    def __init__(self, percentile):
        self.label = "Signal Ratio"
        self.select_optimal = numpy.argmax
        self.percentile = percentile

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        """Compute the signal to noise ratio (SNR) given the result of an acquisition.

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param concofal_end: A confocal image acquired after the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).

        :returns: :math:`0` if no STED foreground, None if :math:`\\text{SNR} < 0` (error), or
                  SNR value otherwise.

        """
        if numpy.any(sted_fg):
            # foreground = numpy.percentile(sted_stack[0][confocal_fg], self.percentile)
            foreground = numpy.mean(sted_stack[0][confocal_fg])
            # background = numpy.mean(sted_stack[0][numpy.invert(sted_fg)])
            # ratio = (foreground - background) / numpy.percentile(confocal_init[confocal_fg], self.percentile)
            ratio = foreground / numpy.percentile(confocal_init[confocal_fg], self.percentile)
            if ratio < 0:
                return None
            else:
                return ratio
        else:
            return 0


class FWHM(Objective):
    """Objective corresponding to the full width at half maximum (FWHM) defined by

    .. math::
        \\text{average}(|2.3558 \\cdot \\sigma |) \\cdot p \\cdot 10^9

    where :math:`p` is the size of a pixel (in nm) in the STED image, and :math:`\\sigma`
    is the standard deviation estimated from a Gaussian fit (see function
    :func:`utils.gaussian_fit`) on at least three line profiles using function
    :func:`user.get_lines`.

    :param pixelsize: Size of a pixel in a STED image (in nm).
    :param `**kwargs`: This method also takes the keyword arguments for :func:`user.get_lines`.
    """
    def __init__(self, pixelsize, **kwargs):
        self.label = "FWHM (nm)"
        self.select_optimal = numpy.argmin
        self.pixelsize = pixelsize
        self.kwargs = kwargs

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        """Compute the full width at half maximum (FWHM) given the result of an acquisition.
        It relies on the function :func:`user.get_lines` to request the user to select line
        profiles in the first STED image of the stack. If the user does not select any lines
        in the STED image, ask the user to select line profiles in the initial confocal.

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param concofal_end: A confocal image acquired after the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).

        :returns: The averaged FWHM (in nm) if success, else None.
        """
        lines = user.get_lines(sted_stack[0], 3, minlen=4, deltas=[-1, 0, 1], **self.kwargs)
        if not lines:
            lines = user.get_lines(confocal_init, 3, minlen=4, deltas=[-1, 0, 1], **self.kwargs)
        fwhms = []
        for positions, profile in lines:
            popt = utils.gaussian_fit(positions, profile)
            if popt is not None:
                fwhms.append(numpy.abs(2.3548 * popt[-1]))
        if fwhms: return numpy.mean(fwhms)*self.pixelsize*1e9
        else: return None



class Autocorrelation(Objective):
    """Objective corresponding to the autocorrelation defined as the difference between
    the value at the first maximum and the value at the first minimum following the first
    maximum, for

    :param `**kwargs`: This method also takes the keyword arguments for :func:`user.get_lines`.
    """
    def __init__(self, **kwargs):
        self.label = "Autocorrelation"
        self.select_optimal = numpy.argmax
        self.kwargs = kwargs

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        lines = user.get_lines(sted_stack[0], minlen=40, deltas=[-1, 0, 1], **self.kwargs)
        profiles = [l[1] for l in lines]
        autocorr = acf(profiles)
        min_val, min_idx = utils.find_first_min(autocorr)
        max_val, max_idx = utils.find_first_max(autocorr, min_idx)
        assert max_val >= min_val
        if max_idx < min_idx:
            return None
        else:
            return max_val - min_val


class Score(Objective):
    """Objective corresponding to the autocorrelation defined as the difference between
    the value at the first maximum and the value at the first minimum following the first
    maximum, for

    :param `**kwargs`: This method also takes the keyword arguments for :func:`user.give_score`.
    """
    def __init__(self, label, select_optimal=numpy.argmax, idx=0, **kwargs):
        self.label = label
        self.select_optimal = select_optimal
        self.idx = idx
        self.kwargs = kwargs

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        return user.give_score(confocal_init, sted_stack[self.idx], self.label, **self.kwargs)


class Bleach(Objective):
    def __init__(self):
        self.label = "Bleach"
        self.select_optimal = numpy.argmin

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        signal_i = numpy.mean(confocal_init[confocal_fg])
        signal_e = numpy.mean(confocal_end[confocal_fg])
        bleach = (signal_i - signal_e) / signal_i
        return bleach


class ScoreNet(Objective):
    def __init__(self, label, net, select_optimal=numpy.argmax, idx=0):
        self.label = label
        self.net = net
        self.select_optimal = select_optimal
        self.idx = idx

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        score = self.net.predict(utils.img2float(sted_stack[self.idx]))
        print("Net", self.label, "score", score)
        return score


class FRC(Objective):
    def __init__(self, pixelsize):
        self.label = "FRC"
        self.select_optimal = numpy.argmax
        self.pixelsize = pixelsize # µm
        self.max_spatialfreq = 1 / (2 * pixelsize) # 1/µm

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        sted = numpy.array(sted_stack[0])
        # verify that the STED image is of squared shape
        assert sted.shape[0] == sted.shape[1],\
            "The STED image is not a square, you cannot evaluate the Fourier Ring Correlation!"
        imgs = fsc.split_image_array(sted, 2)
        fourierringcorr, sigmacurves, freqs = [], [], []
        for im1, im2 in itertools.combinations(imgs, 2):
            frc, nPx = fsc.fourier_shell_corr(im1, im2)
            sigma = fsc.sigma_curve(nPx)
            freq = numpy.arange(frc.shape[0]) / (im1.shape[0] * self.pixelsize)
            fourierringcorr.append(frc)
            sigmacurves.append(sigma)
            freqs.append(freq)

        frc = numpy.mean(numpy.array(fourierringcorr), axis = 0)
        sigma = numpy.mean(numpy.array(sigmacurves), axis = 0)
        freq = numpy.mean(numpy.array(freqs), axis = 0)

        spatialfreq = fsc.meeting_point(fsc.moving_average(frc, 3), freq, fsc.moving_average(sigma, 3))

        return spatialfreq / self.max_spatialfreq

    def mirror_ticks(self, ticks):
        return ["{:0.0f}".format(1e+3 / (self.max_spatialfreq * x)) if x > 0 else "" for x in ticks]


class Resolution(Objective):
    def __init__(self, pixelsize, res_cap=350):
        self.label = "Resolution (nm)"
        self.select_optimal = numpy.argmin
        if isinstance(pixelsize, (tuple, list)):
            pixelsize = pixelsize[0]
        self.pixelsize = pixelsize
#            self.kwargs = kwargs
        self.res_cap=res_cap

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = decorr.calculate(image=sted_stack[0])*self.pixelsize/1e-9
        if res > self.res_cap:
            res = self.res_cap
        return res

class FWHMResolution(Objective):
    def __init__(self, pixelsize):
        self.label = "FWHM (nm)"
        self.select_optimal = numpy.argmin
        self.pixelsize = pixelsize

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        positions = kwargs.get("positions", None)
        if not isinstance(positions, type(None)) and len(positions) < 1:
            positions = peak_local_max(confocal_init, min_distance=10, threshold_rel=0.5)
        return self.get_resolution(sted_stack[0], positions) * 1e+9

    def get_resolution(self, emitter, positions, delta=4, avg=2):
        """
        Computes the resolution from center point in the image. A resolution of 0 is
        returned if no signal is available.

        :param emitter: A `numpy.ndarray` of emitter
        :param delta: The center crop size
        :param avg: The crop height

        :returns : A `float` of the resolution of the image (FWHM)
        """
        def gaussian(x,a,x0,sigma):
            return a*numpy.exp(-(x-x0)**2/(2*sigma**2))
        def fit(func, x, y):
            try:
                popt, pcov = optimize.curve_fit(func, x, y, bounds=((-numpy.inf, -numpy.inf, 0), numpy.inf))
                return min(popt[-1], delta)
            except RuntimeError:
                return None

        # Returns the center resolution
        if isinstance(positions, type(None)):
            positions = [(emitter.shape[0] // 2, emitter.shape[1] // 2)]

        # Calculates the resolutions
        resolutions = []
        for ypos, xpos in positions:
            y = emitter[
                max(0, ypos - avg) : min(ypos + avg, emitter.shape[0]),
                max(0, xpos - delta) : min(xpos + delta, emitter.shape[1])
            ].max(axis=0)
            x = numpy.arange(len(y)) - numpy.argmax(y)

            # We skip if there is no emitters
            if numpy.all(numpy.diff(y) == 0):
                continue
            sigma = fit(gaussian, x, y)
            # We skip if the fit did not converge
            if isinstance(sigma, type(None)):
                continue
            resolution = 2 * numpy.sqrt(2 * numpy.log(2)) * sigma * self.pixelsize
            resolutions.append(
                resolution if resolution >= 2 * self.pixelsize # Nyquist criterion
                else 2 * numpy.sqrt(2 * numpy.log(2)) * delta * self.pixelsize # invalid resolution
            )
        if resolutions:
            # We return the average calculated resolution
            return numpy.mean(resolutions)
        else:
            # We return the maximal resolution
            return 2 * numpy.sqrt(2 * numpy.log(2)) * delta * self.pixelsize

# class Squirrel(Objective):
#     """
#     Implements the `Squirrel` objective
#
#     :param method: A `str` of the method used to optimize
#     :param normalize: A `bool` wheter to normalize the images
#     """
#     def __init__(self, method="L-BFGS-B", normalize=False, use_foreground=False):
#
#         self.method = method
#         self.bounds = (-numpy.inf, numpy.inf), (-numpy.inf, numpy.inf), (0, numpy.inf)
#         self.x0 = (1, 0, 1)
#         self.normalize = normalize
#         self.select_optimal = numpy.argmin
#         self.use_foreground = use_foreground
#
#     def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
#         """
#         Evaluates the objective
#
#         :param sted_stack: A list of STED images.
#         :param confocal_init: A confocal image acquired before the STED stack.
#         :param concofal_end: A confocal image acquired after the STED stack.
#         :param sted_fg: A background mask of the first STED image in the stack
#                         (2d array of bool: True on foreground, False on background).
#         :param confocal_fg: A background mask of the initial confocal image
#                             (2d array of bool: True on foreground, False on background).
#         """
#         # Optimize
#         if not numpy.any(sted_stack[0]):
#             return mean_squared_error(confocal_init[confocal_fg], sted_stack[0][confocal_fg], squared=False)
#         # Optimize
#         result = self.optimize(sted_stack[0], confocal_init)
#         if self.use_foreground:
#             return self.squirrel(result.x, sted_stack[0], confocal_init, confocal_fg=confocal_fg)
#         else:
#             return self.squirrel(result.x, sted_stack[0], confocal_init)
#
#     def squirrel(self, x, *args, **kwargs):
#         """
#         Computes the reconstruction error between
#         """
#         alpha, beta, sigma = x
#         super_resolution, reference = args
#         confocal_fg = kwargs.get("confocal_fg", numpy.ones_like(super_resolution, dtype=bool))
#         convolved = self.convolve(super_resolution, alpha, beta, sigma)
#         if self.normalize:
#             reference = (reference - reference.min()) / (reference.max() - reference.min() + 1e-9)
#             convolved = (convolved - convolved.min()) / (convolved.max() - convolved.min() + 1e-9)
#         error = mean_squared_error(reference[confocal_fg], convolved[confocal_fg], squared=False)
#         return error
#
#     def optimize(self, super_resolution, reference):
#         """
#         Optimizes the SQUIRREL parameters
#
#         :param super_resolution: A `numpy.ndarray` of the super-resolution image
#         :param reference: A `numpy.ndarray` of the reference image
#
#         :returns : An `OptimizedResult`
#         """
#         result = optimize.minimize(
#             self.squirrel, self.x0, args=(super_resolution, reference),
#             method="L-BFGS-B", bounds=((-numpy.inf, numpy.inf), (-numpy.inf, numpy.inf), (0, numpy.inf))
#         )
#         return result
#
#     def convolve(self, img, alpha, beta, sigma):
#         """
#         Convolves an image with the given parameters
#         """
#         return gaussian_filter(img * alpha + beta, sigma=sigma)

class Squirrel(Objective):
    """
    Implements the `Squirrel` objective

    :param method: A `str` of the method used to optimize
    :param normalize: A `bool` wheter to normalize the images
    """
    def __init__(self, method="L-BFGS-B", normalize=False, use_foreground=False):

        self.label = "Squirrel"
        self.method = method
        self.bounds = (-1e+6, 1e+6), (-1e+6, 1e+6), (0, 1e+2)
        self.x0 = (1, 0, 1)
        self.normalize = normalize
        self.select_optimal = numpy.argmin
        self.use_foreground = use_foreground

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        """
        Evaluates the objective

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param concofal_end: A confocal image acquired after the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).
        """
        # Optimize
        if not numpy.any(sted_stack[0]):
            return 1.
#             return mean_squared_error(confocal_init[confocal_fg], sted_stack[0][confocal_fg], squared=False)
        # Optimize
        result = self.optimize(sted_stack[0], confocal_init)
        if self.use_foreground:
            return 1.0 - self.out_squirrel(result.x, sted_stack[0], confocal_init, confocal_fg=confocal_fg)
        else:
            return 1.0 - self.out_squirrel(result.x, sted_stack[0], confocal_init)

    def squirrel(self, x, *args, **kwargs):
        """
        Computes the reconstruction error between
        """
        alpha, beta, sigma = x
        super_resolution, reference = args
        confocal_fg = kwargs.get("confocal_fg", numpy.ones_like(super_resolution, dtype=bool))
        convolved = self.convolve(super_resolution, alpha, beta, sigma)
        if self.normalize:
            reference = (reference - reference.min()) / (reference.max() - reference.min() + 1e-9)
            convolved = (convolved - convolved.min()) / (convolved.max() - convolved.min() + 1e-9)
        error = mean_squared_error(reference[confocal_fg], convolved[confocal_fg], squared=True)
#         error = numpy.quantile(numpy.abs(reference[confocal_fg] - convolved[confocal_fg]), [0.95]).item()
#         error = numpy.mean((reference[confocal_fg] - convolved[confocal_fg]))
        return error

    def out_squirrel(self, x, *args, **kwargs):
        """
        Computes the reconstruction error between
        """
        alpha, beta, sigma = x
        super_resolution, reference = args
        confocal_fg = kwargs.get("confocal_fg", numpy.ones_like(super_resolution, dtype=bool))
        convolved = self.convolve(super_resolution, alpha, beta, sigma)

        reference = (reference - reference.min()) / (reference.max() - reference.min() + 1e-9)
        convolved = (convolved - convolved.min()) / (convolved.max() - convolved.min() + 1e-9)

        _, S = structural_similarity(reference, convolved, full=True, data_range=1.0)
        error = numpy.mean(S[confocal_fg])
        return error

    def optimize(self, super_resolution, reference):
        """
        Optimizes the SQUIRREL parameters

        :param super_resolution: A `numpy.ndarray` of the super-resolution image
        :param reference: A `numpy.ndarray` of the reference image

        :returns : An `OptimizedResult`
        """
        result = optimize.minimize(
            self.squirrel, self.x0, args=(super_resolution, reference),
            method=self.method, bounds=(self.bounds)
        )
        return result

    def convolve(self, img, alpha, beta, sigma):
        """
        Convolves an image with the given parameters
        """
        return gaussian_filter(img * alpha + beta, sigma=sigma)

# class Squirrel(Objective):
#     """
#     Implements the `Squirrel` objective
#
#     :param method: A `str` of the method used to optimize
#     :param normalize: A `bool` wheter to normalize the images
#     """
#     def __init__(self, method="L-BFGS-B", normalize=False, use_foreground=False):
#
#         self.method = method
#         self.bounds = (-numpy.inf, numpy.inf), (-numpy.inf, numpy.inf), (0, numpy.inf)
#         self.x0 = (1, 0, 1)
#         self.normalize = normalize
#         self.select_optimal = numpy.argmin
#         self.use_foreground = use_foreground
#
#     def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
#         """
#         Evaluates the objective
#
#         :param sted_stack: A list of STED images.
#         :param confocal_init: A confocal image acquired before the STED stack.
#         :param concofal_end: A confocal image acquired after the STED stack.
#         :param sted_fg: A background mask of the first STED image in the stack
#                         (2d array of bool: True on foreground, False on background).
#         :param confocal_fg: A background mask of the initial confocal image
#                             (2d array of bool: True on foreground, False on background).
#         """
#         # Optimize
#         if not numpy.any(sted_stack[0]):
#             return 1.
# #             return mean_squared_error(confocal_init[confocal_fg], sted_stack[0][confocal_fg], squared=False)
#         # Optimize
#         result = self.optimize(sted_stack[0], confocal_init)
#         if self.use_foreground:
#             return 1.0 - self.squirrel(result.x, sted_stack[0], confocal_init, confocal_fg=confocal_fg)
#         else:
#             return 1.0 - self.out_squirrel(result.x, sted_stack[0], confocal_init)
#
#     def squirrel(self, x, *args, **kwargs):
#         """
#         Computes the reconstruction error between
#         """
#         alpha, beta, sigma = x
#         super_resolution, reference = args
#         confocal_fg = kwargs.get("confocal_fg", numpy.ones_like(super_resolution, dtype=bool))
#
#         non_zero = super_resolution != 0.
#         _super_resolution = super_resolution.copy()
#         _super_resolution[non_zero] = reference[non_zero]
#
#         convolved = self.convolve(_super_resolution, alpha, beta, sigma)
#         if self.normalize:
#             reference = (reference - reference.min()) / (reference.max() - reference.min() + 1e-9)
#             convolved = (convolved - convolved.min()) / (convolved.max() - convolved.min() + 1e-9)
#         error = mean_squared_error(reference[confocal_fg], convolved[confocal_fg], squared=True)
# #         error = numpy.quantile(numpy.abs(reference[confocal_fg] - convolved[confocal_fg]), [0.95]).item()
# #         error = numpy.mean((reference[confocal_fg] - convolved[confocal_fg]))
#         return error
#
#     def out_squirrel(self, x, *args, **kwargs):
#         """
#         Computes the reconstruction error between
#         """
#         alpha, beta, sigma = x
#         super_resolution, reference = args
#         confocal_fg = kwargs.get("confocal_fg", numpy.ones_like(super_resolution, dtype=bool))
#
#         non_zero = super_resolution != 0.
#         _super_resolution = super_resolution.copy()
#         _super_resolution[non_zero] = reference[non_zero]
#
#         convolved = self.convolve(_super_resolution, alpha, beta, sigma)
#         if self.normalize:
#             reference = (reference - reference.min()) / (reference.max() - reference.min() + 1e-9)
#             convolved = (convolved - convolved.min()) / (convolved.max() - convolved.min() + 1e-9)
# #         error = numpy.mean(numpy.abs(reference[confocal_fg] - convolved[confocal_fg]))
# #         error = numpy.std(numpy.abs(reference[confocal_fg] - convolved[confocal_fg]))
#         error = structural_similarity(reference, convolved)
#         return error
#
#     def optimize(self, super_resolution, reference):
#         """
#         Optimizes the SQUIRREL parameters
#
#         :param super_resolution: A `numpy.ndarray` of the super-resolution image
#         :param reference: A `numpy.ndarray` of the reference image
#
#         :returns : An `OptimizedResult`
#         """
#         result = optimize.minimize(
#             self.squirrel, self.x0, args=(super_resolution, reference),
#             method="L-BFGS-B", bounds=(self.bounds)
#         )
#         return result
#
#     def convolve(self, img, alpha, beta, sigma):
#         """
#         Convolves an image with the given parameters
#         """
#         return gaussian_filter(img * alpha + beta, sigma=sigma)

def exp_func(x, a, b):
    return a*numpy.exp(-b*x)

def apply_hamming_window(data):
    """
    Apply Hamming window to data
    :param data (numpy.ndarray): An N-dimensional Numpy array to be used in Windowing
    :return:
    """
    assert issubclass(data.__class__, numpy.ndarray)

    return _nd_window(data, numpy.hamming)

def _nd_window(data, filter_function, **kwargs):
    """
    Performs on N-dimensional spatial-domain data.
    This is done to mitigate boundary effects in the FFT.
    
    :params data: ndarray
           Input data to be windowed, modified in place.
    :params filter_function: 1D window generation function
           Function should accept one argument: the window length.
           Example: scipy.signal.hamming
    """
    result = data.copy().astype(numpy.float64)
    for axis, axis_size in enumerate(data.shape):
        # set up shape for numpy broadcasting
        filter_shape = [1, ] * data.ndim
        filter_shape[axis] = axis_size
        window = filter_function(axis_size, **kwargs).reshape(filter_shape)
        # scale the window intensities to maintain array intensity
        numpy.power(window, (1.0/data.ndim), out=window)
        result *= window
    return result

class FFTMetric(Objective):
    """
    This is an example of how the metric should be implemented
    """
    def __init__(self, label="FFT"):
        """
        Implements the `FFTMetric` objective
        """
        self.label = label
        self.select_optimal = numpy.argmin

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        """
        Evaluates the objective

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack. #conf1
        :param concofal_end: A confocal image acquired after the STED stack. #conf2
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).        
        """  

        sted_stack = sted_stack[0]
        sted_stack = apply_hamming_window(sted_stack)

        #Test radial profil and exp fit
        FFT_sted = scipy.fft.fftshift(scipy.fft.fft2(sted_stack))
        h, w = FFT_sted.shape #image dimensions
        val_sted = []
        yc, xc = int((h+1)/2) + 1, int((w+1)/2) + 1 #center position of x and y 
        rmax = min([w-xc, h-yc]) #cercle diameter

        for r in range(rmax): #cercles, r from 0 to rmax-1
            x, y = numpy.ogrid[0:FFT_sted.shape[0], 0:FFT_sted.shape[1]] #matrix with the same size as the original image
            circle_maskA = numpy.sqrt((x-xc)**2 + (y-yc)**2) >= r-0.5 #creates difference circle mask centered and with radius r-0.5
            circle_maskB = numpy.sqrt((x - xc)**2 + (y - yc)**2) < r + 0.5 #creates circle mask centered and with radius r+0.5
            mask = circle_maskA * circle_maskB #creates ring masks
            abs_val = numpy.sum(numpy.abs(FFT_sted[mask]**2)) #mask application on the FFT image
            val_sted.append(abs_val)

        val_sted = numpy.array(val_sted)

        if val_sted[1] == 0: #zero images exception
            metric = 1
        else:
            max_index = 1
            m, M = numpy.min(val_sted[max_index:]), numpy.max(val_sted[max_index:])
            val_sted_new = (val_sted[max_index:] - m) / (M - m)
            x = numpy.linspace(0, len(val_sted_new)-1, len(val_sted_new))
            popt, pcov = scipy.optimize.curve_fit(exp_func, x, val_sted_new, bounds = (0, [5, 10]))
            opt_func = exp_func(x, *popt) #exp fit to data
            delta = (val_sted_new - opt_func)**2
#             metric = (numpy.sqrt(numpy.mean(delta)))*5 # x5 to be changed or justified
            metric = numpy.sqrt(numpy.average(delta, weights=numpy.linspace(1, 5, len(delta)))) * 5

        return min([1, metric])

class Crosstalk(Objective):
    def __init__(self, label="Crosstalk"):
        self.label = label
        self.select_optimal = numpy.argmin

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs):
        """
        Evaluates the objective

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack. #conf1
        :param concofal_end: A confocal image acquired after the STED stack. #conf2
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).
        """
        sted_stack = sted_stack[0]
        other_sted = kwargs.get("other_sted_image", numpy.zeros_like(sted_stack))

        sted_stack = (sted_stack - sted_stack.min()) / (sted_stack.max() - sted_stack.min() + 1e-9)
        other_sted = (other_sted - other_sted.min()) / (other_sted.max() - other_sted.min() + 1e-9)

        _, S = structural_similarity(sted_stack, other_sted, full=True, data_range=1.0)
        error = numpy.mean(S[~confocal_fg])
        print("Crosstalk", error)
        return error