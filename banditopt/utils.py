
"""The :mod:`utils` modules contains tools used in other modules.
"""

import os

import numpy
import matplotlib; matplotlib.use("TkAgg")
from matplotlib import pyplot

from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist

from skimage import filters

import deap
from deap import base
from deap import creator
from deap import tools


def avg_area(img, radius, point):
    """Compute the average of the area defined by a *radius* around a given
    *point* on an image.

    :param 2d-array img: The image.
    :param tuple radius: Vertical and horizontal radius to consider around the *point*.
    :param tuple point: Coordinates.

    :returns: The average of the area around the given *point*.
    """
    x, y = int(point[0]), int(point[1])
    x_max, y_max = img.shape[1] - 1, img.shape[0] - 1
    assert y >= 0 and x >= 0 and y <= y_max and x <= x_max

    y_start, x_start = max(0, y-radius), max(0, x-radius)
    y_end, x_end = min(y_max, y_start+2*radius), min(x_max, x_start+2*radius)
    return numpy.mean(img[y_start:y_end+1, x_start:x_end+1])


def estimate_signal(img, radius, points):
    """Estimate the signal of an image using the mean of averages over areas of given
    *radius* around all the given *points*.

    :param 2d-array img: The image.
    :param tuple radius: Vertical and horizontal radius to consider around *points*.
    :param list points: Coordinates tuples.

    :returns: The estimated signal.
    """
    avg_areas = [avg_area(img, radius, point) for point in points]
    return numpy.mean(avg_areas)


def gaussian_fit(positions, values, visual=False):
    """Fit the parameters *popt* of a Gaussian distribution given *values* observed
    at *positions*. This function uses the function :func:`curve_fit` from module
    :mod:`scipy.optimize` to minimize the sum of the squared residuals of
    :math:`f(*positions*, *popt) - values`, where :math:`f` is given by function
    :func:`gauss`. Prints the error and displays a plot of *positions* and *values*
    if the fit fails.

    :param list positions: Positions :math:`x`.
    :param list values: Amplitudes :math:`y`.
    :param bool visual: If True, display a plot of *positions* and *values*.

    :returns popt: Optimal values for the parameters to fit function :func:`gauss` to
                   the given data if fit is successful, else None.
    """
    y0 = numpy.mean(values)
    mu = numpy.mean(positions)
    sigma = numpy.std(positions)
    a = numpy.max(values) * 2 * sigma
    try:
        popt = curve_fit(gauss, positions, values, p0=[y0, a, mu, sigma])[0]
        if visual:
            pyplot.figure()
            pyplot.plot(positions, values, "bo")
            pyplot.title("sigma = "+str(sigma))
            pyplot.show(block=True)
    except (RuntimeError, TypeError, NameError) as err:
        print("Gaussian fit failed:", err)
        pyplot.figure("Failed to fit these data")
        pyplot.plot(positions, values, "bo")
        pyplot.show(block=True)
        popt = None
    return popt


def points2regions(points, pixelsize, resolution):
    """Translate *points* corresponding to indices in a 2d array (image) into
    positions describing regions in an image (in nm).

    :param list points: List of (x, y) indices of regions center positions.
    :param tuple pixelsize: Tuple of pixel size (x, y) in nm.
    :param tuple resolution: Tuple of region size (x, y) in number of pixels.

    :returns: List of (x, y) positions of regions upper corners (in nm).
    """
    x_pixels, y_pixels = resolution
    x_size, y_size = pixelsize
    print("points2regions")
    print("resolution", x_pixels, "(x),", y_pixels, "(y)")
    print("pixelsize", x_size, "(x),", y_size, "(y)")
    return [((x-x_pixels/2)*x_size, (y-y_pixels/2)*y_size) for (x, y) in points]


def gauss(x, y0, a, mu, sigma):
    """Evaluate the Gaussian function

    .. math::
        f(x) = y_0 + a \\sqrt{\\pi/2} / (2 \\sigma) e^{-2(x-\\mu)^2/(2\\sigma)^2}

    at value *x*.

    :param x: Scalar value at which to evaluate the function.
    :param y0: Baseline value.
    :param a: Amplitude value.
    :param mu: Mean of the Gaussian function.
    :param sigma: Standard deviation of the Gaussian function.

    :returns: The value of the function evaluated at *x*.
    """
    w = 2 * sigma
    return y0 + (a / w * numpy.sqrt(numpy.pi/2)) * numpy.exp(-2*(x-mu)**2/w**2)



def get_foreground(img):
    """Return a background mask of the given image using the OTSU method to threshold.

    :param 2d-array img: The image.

    :returns: A mask (2d array of bool: True on foreground, False on background).
    """
    val = filters.threshold_otsu(img)
    return img > val


def find_first_min(data, start_idx=0):
    """Find the first minimum in *data* after *start_idx*.

    :param data: List of Values.
    :param int start_idx: Index from after which to look for first minimum.

    :returns: The value at first minimum and the corresponding index.
    """
    for i in range(start_idx, len(data)-1):
        if data[i] < data[i+1]:
            return data[i], i
    return data[-1], len(data) - 1


def find_first_max(data, start_idx=0):
    """Find the first maximum in *data* after *start_idx*.

    :param data: List of values.
    :param int start_idx: Index from after which to look for first maximum.

    :returns: The value at first maximum and the corresponding index.
    """
    for i in range(start_idx, len(data)-1):
        if data[i] > data[i+1]:
            return data[i], i
    return data[-1], len(data) - 1


def plot_regression(objectives, algos, X_pred, param_idx, param_label, output, t):
    """Plots and saves the given prediction of the algorithms on the parameter
    space.

    :param objectives: List of objectives names.
    :param algos: List of algorithms dedicated to every objectives.
    :param 2d-array X_pred: The parameter space.
    :param int param_idx: The id of the parameter.
    :param str param_label: The labels of the parameters.
    :param str output: The folder where to save the figures.
    :param t: The time of the optimization.
    """
    for algo, obj in zip(algos, objectives):
        mean, std = algo.predict(X_pred)
        pyplot.figure()
        pyplot.plot(X_pred[:, param_idx], mean, "--")
        pyplot.fill_between(X_pred[:, param_idx], mean - std, mean + std, alpha=0.4)
        if algo.X is not None:
            pyplot.plot(algo.X[:, param_idx], algo.y, "o")
        pyplot.ylim(0, None)
        pyplot.xlabel(param_label)
        pyplot.ylabel(obj.label)
        name = os.path.join(output, "Regression", "{}_{}_{}.pdf".format(obj.label, param_label.replace("/", ""), t))
        pyplot.savefig(name, bbox_inches="tight", pad_inches=0.03, frameon=None)
        pyplot.close()


def rescale(X, X_max, X_min):
    """Rescale the given data between the provided maximum and minimum values.

    :param X: Data.
    :param X_max: Maximum value.
    :param X_min: Minimum value.

    :returns: The rescaled data.
    """
    return (X - X_min) / (X_max - X_min)


def img2float(img):
    """Transform (possibly unsigned) integer image data into a float image.

    :param 2d-array img: An image.

    :returns: The image with pixels in float.
    """
    if img.dtype == numpy.uint16:
        return rescale(img, float(2**16-1), 0.0)
    elif img.dtype == numpy.int16:
        return rescale(img, float(2**15-1), -float(2**15))
    elif img.dtype == numpy.uint8:
        return rescale(img, float(2**8-1), 0.0)
    elif img.dtype == numpy.int8:
        return rescale(img, float(2**7-1), -float(2**7))
    else:
        raise TypeError


# def pareto_front(points):
#     """
#     this function returns indexes of the pareto front of a minimization problem
#     parameters:
#         points   2D array of point with as many columns as objectives
#     """
#     if "FitnessMult" in dir(creator):
#         del creator.FitnessMult
#     if "Individual" in dir(creator):
#         del creator.Individual
#     creator.create("FitnessMult", base.Fitness, weights=tuple([-1.]*points.shape[1]))
#     creator.create("Individual", list, fitness=creator.FitnessMult)
#     points_list = points.tolist()
#     for i in range(len(points_list)):
#         points_list[i] = deap.creator.Individual(points_list[i])
#         points_list[i].fitness.values = tuple(points_list[i])
#         points_list[i].id = i
#     individuals = tools.sortLogNondominated(points_list, k=len(points_list), first_front_only=True)
# #     front = np.array(individuals)
#     return [x.id for x in individuals]

def pareto_front(points, discretization=100):
    """
    Implements a Pareto grid search over all points.

    This method is very efficient in terms of calculation. To save
    a significant amount of time we separate the distribution of points
    in specific cuboids.
    The pareto front calculation is performed on the position of the
    cuboids in the input space instead of all points separately. This
    allows to reduce the number of points during the pareto calculation.
    We then extract the points from each cuboids which are the points
    returned by this method.

    :param points: A `numpy.ndarray` of points with shape (N, D)
    :param discretization: An `int` of the number of grids to use (default : 100)

    :returns : A `numpy.ndarray` of optimal points with shape (N, D)
    """
    def nondominated(points):
        """
        Calculates the indices of the pareto front for a minimization problem

        :param points: A `numpy.ndarray` of points

        :returns : A `numpy.ndarray` of selected indices
        """
        if "FitnessMult" in dir(creator):
            del creator.FitnessMult
        if "Individual" in dir(creator):
            del creator.Individual
        creator.create("FitnessMult", base.Fitness, weights=tuple([-1.]*points.shape[1]))
        creator.create("Individual", list, fitness=creator.FitnessMult)
        points_list = points.tolist()
        for i in range(len(points_list)):
            points_list[i] = deap.creator.Individual(points_list[i])
            points_list[i].fitness.values = tuple(points_list[i])
            points_list[i].id = i
        individuals = tools.sortLogNondominated(points_list, k=len(points_list), first_front_only=True)
        return numpy.array([x.id for x in individuals])

    ndim = points.shape[-1]
    if ndim < 2:
        return numpy.min(points)

    # Retrieves the index of the hypervolume histogram
    indices, _bins = [], []
    for pts in points.T:
        bins = numpy.linspace(pts.min(), pts.max(), discretization)
        ind = numpy.digitize(pts, bins)
        indices.append(ind)
        _bins.append(bins)
    # Merges all indices in the hypervolume
    indices = numpy.array(indices).T
    bins = numpy.array(_bins)

    # We create the hypervolume of where points are contained
    # We add a value of 2 since points may be out of bounds in digitize
    ary = numpy.zeros([discretization + 2] * ndim, dtype=bool)
    # Indexing the array this way is very fast
    ary[tuple([indices[:, i] for i in range(ndim)])] = True
    # We remove the extra pad on the array
    ary = ary[tuple([slice(1, -1) for _ in range(ndim)])]
    # We return where the array contains points
    where = numpy.argwhere(ary)

    # Gets the value of each histogram bins which forms the pareto points
    pareto_points = []
    for i in range(ndim):
        pareto_points.append(bins[i, where[:, i]])
    pareto_points = numpy.array(pareto_points).T

    # Calculate the pareto front from the created cells
    # In this case the pareto points refer to the specific cells
    pareto_points = nondominated(pareto_points)

    # Recovers the indices of the cells
    cells = where[pareto_points]

    # We recover the points contained within each of the pareto
    # cells
    return_indices = []
    for cell in cells:
        # We add 1 to the cell ID since digitize returns 1 for the
        # first cell and 0 for out of bounds data
        where = numpy.prod(indices == (cell + 1), axis=-1)
        if numpy.any(where):
            return_indices.extend(numpy.argwhere(where).ravel())
    # We apply the unique operation to ensure that no points are
    # present multiple time
    return numpy.unique(return_indices)
