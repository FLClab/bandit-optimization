
"""The :mod:`utils` modules contains tools used in other modules.
"""

import os

import numpy
import matplotlib; #matplotlib.use("TkAgg")
from matplotlib import pyplot

from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist

from skimage import filters
from functools import partial

import deap
from deap import base
from deap import creator
from deap import tools

import pandas as pd
import array
import random
from math import sqrt
from deap import algorithms
from deap import base
from deap.benchmarks.tools import diversity, convergence, hypervolume
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


def pareto_front(points, weights=None):
    """
    this function returns indexes of the pareto front of a minimization problem
    parameters:
        points   2D array of point with as many columns as objectives
    """
    if "FitnessMult" in dir(creator):
        del creator.FitnessMult
    if "Individual" in dir(creator):
        del creator.Individual
    if isinstance(weights, (type(None))):
        weights = tuple([-1.]*points.shape[1])
    creator.create("FitnessMult", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMult)
    points_list = points.tolist()
    for i in range(len(points_list)):
        points_list[i] = deap.creator.Individual(points_list[i])
        points_list[i].fitness.values = tuple(points_list[i])
        points_list[i].id = i
    individuals = tools.sortLogNondominated(points_list, k=len(points_list), first_front_only=True)
#     front = numpy.array(individuals)
    return [x.id for x in individuals]

def bigger_than(values, low, high, bound_low, **kwargs):
    """
    Replaces the values within the desired range of values

    :param values: A `list` of possible values
    :param low: An `int` of the index of first option
    :param high: An `int` of the index of second option
    :param bound_low: A `list` of lower bounds

    :returns : A `list` with valid values
    """
    if values[low] > values[high]:
        values[low] = random.uniform(bound_low[low], values[high])
    return values

def conditioned_uniform(low, up, conditions, size=None):
    """
    Generates a conditioned uniform sampling

    The list of conditions is assumed to contain `functools.partial` instantiated
    methods. For example
    ```
    values = [2.2, 1.3]
    conditions = [partial(bigger_than, low=0, high=1, bound_low=[0, 0])]
    for condition in conditions:
        values = condition(values)

    # Would return
    values = [random.uniform(0, 1.3), 1.3]
    ```

    :param low: A `list` of lower bounds
    :param high: A `list` of higher bounds
    :param conditions: A `list` of conditions
    :param size: An optional parameter

    :returns : A valid uniform sampling
    """
    values = [random.uniform(a, b) for a, b in zip(low, up)]
    for condition in conditions:
        values = condition(values)
    return values

def uniform(low, up, size=None):
    # This function is used by NSGAII
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

def NSGAII(optim_func, BOUND_LOW, BOUND_UP, weights, NGEN=250, MU=100,
           CXPB=0.9, eta_cx=20.0, eta_mu=20.0, indpb_nom = 1,
           L = 6, min_std=None, seed=None, prnt=False, return_niter=True,
           conditions=[], integer_params=[], param_names=None, verbose=False, *args, **kwargs):
    """
    ---- Problem parameters ----
    param optim_func:   Function to optimize
    param BOUND_LOW:    List of the lower bounds of the variables
    param BOUND_UP:     List of the upper bounds of the variables
    param weights:      List of weight. -1.0 if minize, +1.0 if maximize.
    ---- NSGA-II parameters ----
    param NGEN:         Number of generations
    param MU:           Population size
    param CXPB:         Probalbility of crossover
    param eta_cx:       Crowding degree of the crossover
    param eta_mu:       Crowding degree of the mutation
    param indpb_nom:    Independent probability for each attribute to be exchanged = indpb_nom/NDIMS
    param seed:         Random seed
    param L:            Number values of max crowding distance on which the std is calculated
    param min_std:      If not None, threshold under which the algorithm is stopped
    param conditions:   A `list` of conditions
    """

    def _max(x):
        x = numpy.array(x)
        where = x != numpy.inf
        if numpy.any(where):
            return numpy.max(x[where])
        return 1.0e+9
    def _min(x):
        x = numpy.array(x)
        where = x != numpy.inf
        if numpy.any(where):
            return numpy.min(x[where])
        return 1.0e+9
    def _mean(x):
        x = numpy.array(x)
        where = x != numpy.inf
        if numpy.any(where):
            return numpy.mean(x[where])
        return 1.0e+9
    def filter_individual(individuals, conditions):
        filtered = []
        for ind in individuals:
            values = ind.tolist()
            for condition in conditions:
                values = condition(values)
            filtered.append(creator.Individual(values))
        return filtered

    NDIM = len(BOUND_LOW)
    indpb = indpb_nom/NDIM

    creator.create("FitnessMin", base.Fitness, weights=weights)
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    if conditions:
        toolbox.register("attr_float", conditioned_uniform, BOUND_LOW, BOUND_UP, conditions, NDIM)
    else:
        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#    toolbox.register("evaluate", optim_func)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=eta_cx)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=eta_mu, indpb=indpb)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("filter", filter_individual)


    random.seed(seed)

    stats = tools.Statistics(lambda ind: ind.fitness.crowding_dist)
    stats.register("avg", _mean)
#     stats.register("std", numpy.std)
    stats.register("min", _min)
    stats.register("max", _max)

    logbook = tools.Logbook()
    # logbook.header = "gen", "evals", "std", "min", "avg", "max"
    logbook.header = "gen", "evals", "min", "max", "avg"

    pop = kwargs.get("pop", None)
    if isinstance(pop, type(None)):
        pop = toolbox.population(n=MU)
    else:
        # Generates a random population to ensure diversity
        random_pop = toolbox.population(n=int(kwargs.get("percent-replace") * MU))
        for i, random_ind in enumerate(random_pop):
            pop[i] = random_ind
        # Removes previous fitness from individuals
        for ind in pop:
            del ind.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = optim_func(invalid_ind, params_to_round=integer_params, weights=weights)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    if prnt:
        print(logbook.stream)

    # Begin the generational process
    if verbose:
        from tqdm.auto import trange
        vrange = partial(trange, desc="Generations")
    else:
        vrange = range
    for gen in vrange(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Ensures the offspring also contains conditioned values
        offspring = toolbox.filter(offspring, conditions)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = optim_func(invalid_ind, params_to_round=integer_params, weights=weights)
        # print(fitnesses)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if prnt:
            print(logbook.stream)


        if gen >= L-1:
            # Stopping criterion from Roudenko et al; A Steady Performance Stopping Criterion for Pareto-based Evolutionary Algorithms
            # # Calculate the rolling std of the max crowding distance
            std = numpy.std([d['max'] - d['min'] for d in logbook[-L:]])
            if (min_std is not None) and (std < min_std):
                # Stop the optimization
                break

    X = numpy.array(tools.sortLogNondominated(pop, len(pop),first_front_only=True))
    for param in integer_params:
        X[:, param_names.index(param)] = numpy.round(X[:, param_names.index(param)])

    del creator.FitnessMin
    del creator.Individual
    if prnt:
        print("Final population hypervolume is %f" % hypervolume(pop, [11.0]*len(weights)))

    if return_niter:
        return X, pd.DataFrame(logbook), gen+1, pop
    else:
        return X, pd.DataFrame(logbook), pop
