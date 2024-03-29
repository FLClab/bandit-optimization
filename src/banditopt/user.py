
"""The :mod:`user` module contains classes and tools to interact
with the user during optimization. It relies on :mod:`matplotlib` to display
content.
"""

import numpy
import matplotlib; #matplotlib.use("TkAgg")
from matplotlib import gridspec, pyplot , widgets
pyplot.ion()

import skimage

try:
    from . import microscope
except:
    print("Could not load microscope interface. Some functions may not be available.")
from . import utils


class LinePicker:
    """This class allows the user to draw lines on an image and extract the
    position of pixels covered by the line along with the associated line profile
    given an image.

    :param figure: The matplotlib :class:`figure` used for display.
    :param aximg: The figure axis displaying the image on which to trace the line.
    :param axprofile: The figure axis displaying the profile of the line.
    :param img: The image displayed by `aximg`.
    :param minlen: The minimal length in pixels of the line.
    :param deltas: List of pixels to consider for averaging above (negative) and
                   below (positive) the line.
    """
    def __init__(self, figure, aximg, axprofile, img, minlen, deltas=[0]):
        figure.canvas.mpl_connect("button_press_event", self.on_press)
        figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.aximg = aximg
        self.axprofile = axprofile
        self.img = img
        self.minlen = minlen
        self.deltas = deltas
        self.pressed = False
        self.start = None
        self.end = None
        self.positions = None
        self.profile = None

    def on_press(self, event):
        """This method verifies that no items are selected in the toolbar and that
        the event is in `aximg`. If True, the start coordinate are keep in memory.

        :param event: A `matplotlib.backend_bases.Event`.
        """
        if pyplot.get_current_fig_manager().toolbar.mode != "":
            print("Not considering event because something is clicked in the toolbar!")
        elif event.inaxes == self.aximg:
            self.start = (int(event.xdata), int(event.ydata))
            self.pressed = True

    def on_release(self, event):
        """This method verifies that the release event is in `aximg` and that no
        items are selected in the toolbar. If true and the user has pressed the `aximg`
        the point of release is registered. Using the :mod:`skimage` module the
        positions and the profile are acquired.

        :param event: A `matplotlib.backend_bases.Event`.
        """
        if pyplot.get_current_fig_manager().toolbar.mode != "":
            print("Not considering event because something is clicked in the toolbar!")
        elif event.inaxes == self.aximg:
            if self.pressed:
                point = (int(event.xdata), int(event.ydata))
                if point == self.start:
                    self.start = None
                    print("Pick lines, not points!")
                else:
                    self.end = point
                    cc, rr = skimage.draw.line(*self.start, *self.end)
                    self.positions = numpy.sqrt((rr - self.start[1])**2 + (cc - self.start[0])**2)
                    print(self.start, self.end)
                    print("cc", cc)
                    print("rr", rr)
                    print("self.positions", self.positions)
                    data = []
                    for delta in self.deltas:
                        data.append(self.img[rr+delta, cc])
                    data = numpy.asarray(data)
                    profile = numpy.mean(data, axis=0)
                    self.axprofile.clear()
                    # enforces a profile line of at least minlen pixels
                    if profile.shape[0] < self.minlen:
                        self.profile = None
                        print("Pick a longer profile line!")
                    else:
                        # keep only the first 40 pixels
                        self.profile = profile[:40]
                        self.axprofile.plot(self.positions, self.profile)

                        # Plot line picked
                        self.aximg.scatter(cc,rr, s=1,)
                        # Plot start stop
                        self.aximg.scatter([self.start[0], self.end[0]], [self.start[1], self.end[1]])
                        # Plot gaussian_fit an FWHM
                        x_gauss = numpy.linspace(self.positions.min(), self.positions.max(), 100)
                        fit_values = utils.gaussian_fit(self.positions, self.profile)
                        y0, a, mu, sigma = tuple(utils.gaussian_fit(self.positions, self.profile))
                        y_gauss = utils.gauss(x_gauss, y0, a, mu, sigma)
                        self.axprofile.plot(x_gauss, y_gauss, "--", alpha=0.6)
                        self.axprofile.hlines(y_gauss.max()/2, x_gauss.min(), x_gauss.max(), alpha=0.4, label=str(round(sigma*2.3548, 4)))
                        self.axprofile.legend()
                self.pressed = False


class PointPicker:
    """This class allows the user to select points on an image.

    :param figure: The matplotlib :class:`figure` used for display.
    """
    def __init__(self, figure):
        figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.points = []

    def on_press(self, event):
        """This method appends the x and y coordinates of the selection in the
        matplotlib :class:`figure` displayed to a list of points.

        :param event: A `matplotlib.backend_bases.Event`.
        """
        self.points.append((int(event.xdata), int(event.ydata)))


def get_lines(img, n, maxratio=0.4, figsize=(10, 10), **kwargs):
    """This function uses the class :class:`LinePicker` to ask the user
    to select *n* lines and return the indices (in *img*) that they cover
    and associated profiles based on a given image.

    :param img: The image to compute the line profile.
    :param n: The number of lines to select.
    :param maxratio: The ratio of maximal image intensity used to define the
                     max colorbar value.
    :param figsize: The size of figure to display.
    :param `**kwargs`: This method also takes the keyword arguments for initializing :class:`LinePicker_`.
    :return: A list of line positions and profiles.
    """
    lines = []
    while len(lines) < n:
        fig = pyplot.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, width_ratios=[1], height_ratios=[4, 1])
        aximg = pyplot.subplot(gs[0])
        im = aximg.imshow(img, interpolation=None, cmap="gray", vmax=maxratio*numpy.max(img))
        aximg.grid(True)
        axprofile = pyplot.subplot(gs[1])
        lp = LinePicker(fig, aximg, axprofile, img, **kwargs)
        pyplot.show(block=True)
        lines.append([lp.positions, lp.profile])
        if not lines:
            print("Early return in function get_lines (user.py)!")
            break
    return lines


def get_points(img, at_least_n, label="", pixelsize=None, x_offset=None, y_offset=None, resolution=None):
    """This function uses the class :class:`PointPicker` to ask the user
    to select at least `at_least_n` points on a given image and return their
    positions (indices in `img`).

    :param img: The image on which to select points.
    :param at_least_n: The minimum number of points to select.
    :param label: An additional sufffix for the window title.
    :return: A list of points positions.
    """
    points = []
    points_left = at_least_n
    while points_left > 0:
        fig = pyplot.figure()
        fig.canvas.set_window_title("Pick at least {} {}".format(points_left, label))
        pp = PointPicker(fig)

        # if pixelsize is not None:
        #     # pyplot.gca().set_yticklabels(numpy.around(pyplot.yticks()[0]*pixelsize[1] + (y_offset-resolution[1]/2)*pixelsize[1], 2))
        #     # pyplot.gca().set_xticklabels(numpy.around(pyplot.xticks()[0]*pixelsize[0] + (x_offset-resolution[0]/2)*pixelsize[0], 2))
        #     minx = (x_offset-resolution[0]/2)*pixelsize[0]
        #     maxx = (x_offset+resolution[0]/2)*pixelsize[0]
        #     miny = (y_offset-resolution[1]/2)*pixelsize[1]
        #     maxy = (y_offset+resolution[1]/2)*pixelsize[1]
        pyplot.imshow(img, interpolation=None, picker=True, cmap="gray",  vmax=0.4*numpy.max(img))#, extent=(minx, maxx, miny, maxy))
        # pyplot.locator_params(axis='y', nbins=21)
        # pyplot.locator_params(axis='x', nbins=21)
        # import pdb; pdb.set_trace()
        if len(points)>0:
            print(points)
            break
            pyplot.scatter([point[0] for point in points], [point[1] for point in points])
        pyplot.colorbar()



        pyplot.grid(True)
        pyplot.show(block=True)
        points.extend(pp.points)
        points_left -= len(points)
        if not points:
            print("early return in function get_points (user.py)")
            break

    return points


# def get_regions(at_least_n=1, config=None, overview_name=None):
#     """Acquire an overview from a specific configuration, ask the user to select
#     regions and return their offsets.
#
#     :param at_least_n: The minimum number of regions to select.
#     :param config: The microscope configuration.
#     :param overview_name: The name of the overview.
#     :return: The offsets of the selected regions.
#     """
#     if config is None:
#         config = microscope.get_config("Setting configuration for overview", "overview_logo.png")
#     img = microscope.get_overview(config, name=overview_name)
#     points = get_points(img, at_least_n, " subregions within the overview")
#     regions = utils.points2regions(points, microscope.get_pixelsize(config), microscope.get_resolution(config))
#     x_offset, y_offset = microscope.get_offsets(config)
#     regions_offset = [(x + x_offset, y + y_offset) for (x, y) in regions]
#     return regions_offset


def get_regions(at_least_n=1, config=None, overview_name=None):
    """Acquire an overview from a specific configuration, ask the user to select
    regions and return their offsets.

    :param at_least_n: The minimum number of regions to select.
    :param config: The microscope configuration.
    :param overview_name: The name of the overview.
    :return: The offsets of the selected regions.
    """
    if config is None:
        config = microscope.get_config("Setting configuration for overview", "overview_logo.png")
    img = microscope.get_overview(config, name=overview_name)
    x_offset, y_offset = microscope.get_offsets(config)
    # import pdb; pdb.set_trace()
    pixelsize = microscope.get_pixelsize(config)
    resolution = microscope.get_resolution(config)
    points = get_points(img, at_least_n, " subregions within the overview", pixelsize=pixelsize, x_offset=x_offset, y_offset=y_offset, resolution=resolution)
    regions = utils.points2regions(points, pixelsize, resolution)

    regions_offset = [(x + x_offset, y + y_offset) for (x, y) in regions]
    return regions_offset

def select(thetas, objectives, with_time, times, figsize=(10, 10), borders=None):
    """Asks the user to select the best option by clicking on the points from the
    :mod:`matplotlib` figure. If several points overlap, select the one that minimizes
    the time (or third objective).

    :param thetas: A 2d-array of options sampled from the algorithms.
    :param objectives: A list of objectives name.
    :param with_time: (bool) Wheter of not to consider *times* as an objective.
    :param times: An array of time for acquiring an image using each configuration in *thetas*.
    :param figsize: The size of figure to display.
    :return: The index of the selected point.
    """
    print("Asking user to select best option...")

    if borders is not None:
        for idx, y in enumerate(thetas):
            min, max = borders[idx]
            y[y<min] = min
            y[y>max] = max

    fig = pyplot.figure(figsize=figsize)
    ax = fig.gca()

    # set to your favorite colormap (see https://matplotlib.org/users/colormaps.html)
#    cmap = pyplot.cm.get_cmap("nipy_spectral")
    cmap = pyplot.cm.get_cmap("jet")

    if with_time:
        title = ax.set_title(f"Pick the best option by clicking on the point. Tmin={numpy.min(times)}, Tmax={numpy.max(times)}")
    else:
        title = ax.set_title(f"Pick the best option by clicking on the point.")

    # 3 points tolerance
    if with_time:
        time_range = times.max() - times.min() + 1e-11


    if len(thetas)==3:
        if with_time:
            sc = ax.scatter(thetas[0], thetas[1], s=(times-times.min())/time_range * 60 + 20, c=thetas[2], marker="o", alpha=0.5, picker=3, cmap=cmap)
        else:
            sc = ax.scatter(thetas[0], thetas[1], c=thetas[2], marker="o", alpha=0.5, picker=3, cmap=cmap)#, vmin=0, vmax=1)
        pyplot.colorbar(sc, ax=ax)
    elif len(thetas)==2:
        if with_time:
            sc = ax.scatter(thetas[0], thetas[1], s=(times-times.min())/time_range * 60 + 20, marker="o", alpha=0.5, picker=3)
        else:
            sc = ax.scatter(thetas[0], thetas[1], marker="o", alpha=0.5, picker=3)
    ax.grid(True)
    if borders is not None:
        xmin, xmax = borders[0]
        delta = xmax-xmin
        ax.set_xlim(xmin-0.1*delta, xmax+0.1*delta)
        xmin, xmax = borders[1]
        delta = xmax-xmin
        ax.set_ylim(xmin-0.1*delta, xmax+0.1*delta)
        if len(borders)>=3:
            sc.set_clim(borders[2])
    else:
#            pyplot.sca(ax)
#            pyplot.axis('tight')
        xmin, xmax = numpy.min(thetas[0]), numpy.max(thetas[0])
        print(xmin, xmax)
        dx = xmax-xmin
        if xmin!=xmax:
            ax.set_xlim(xmin-0.1*dx, xmax+0.1*dx)
        ymin, ymax = numpy.min(thetas[1]), numpy.max(thetas[1])
        print(ymin, ymax)
        dy = ymax-ymin
        if ymin!=ymax:
            ax.set_ylim(ymin-0.1*dy, ymax+0.1*dy)





#        lgnd = pyplot.legend(loc="lower left", numpoints=1, fontsize=100)
#    elif len(objectives) > 2:
#        sc = ax.scatter(thetas[0], thetas[1], s=100, c=times, marker="o", alpha=0.5, picker=3, cmap=cmap)
#        pyplot.colorbar(sc, ax=ax)
#
#
#    else:
#        ax.scatter(thetas[0], thetas[1], s=200, marker="o", alpha=0.5, picker=3)
#    ax.grid(True)
#    ax.set_xlabel(objectives[0].label)
#    ax.set_ylabel(objectives[1].label)

#    # handle adding custom ticks on mirror axis
#    new_ticks_x = objectives[0].mirror_ticks(ax.get_xticks())
#    if new_ticks_x is not None:
#        ax2 = ax.twiny()
#        ax2.set_xlim(ax.get_xlim())
#        ax2.set_xticklabels(new_ticks_x)
#        title.set_y(1.05)
#
#        if with_time:
#            import pdb; pdb.set_trace()
#            sc.remove()
#            sc = ax2.scatter(thetas[0], thetas[1], s=100, c=times, marker="o", alpha=0.5, picker=3, cmap=cmap)
#        elif len(objectives) > 2:
#            sc.remove()
#            sc = ax2.scatter(thetas[0], thetas[1], s=100, c=thetas[2], marker="o", alpha=0.5, picker=3, cmap=cmap)
#            print("--------------------------------------awrefsdrgsdrfgsdf dsf---------1")
#
#        else:
#            sc.remove()
#            sc = ax2.scatter(thetas[0], thetas[1], s=200, marker="o", alpha=0.5, picker=3)
#
#    new_ticks_y = objectives[1].mirror_ticks(ax.get_yticks())
#    if new_ticks_y is not None:
#        ax3 = ax.twinx()
#        ax3.set_ylim(ax.get_ylim())
#        ax3.set_yticklabels(new_ticks_y)
#
#        if with_time:
#            import pdb; pdb.set_trace()
#            sc.remove()
#            sc = ax3.scatter(thetas[0], thetas[1], s=100, c=times, marker="o", alpha=0.5, picker=3,
#                             cmap=cmap)
#        elif len(objectives) > 2:
#            sc.remove()
#            sc = ax3.scatter(thetas[0], thetas[1], s=100, c=thetas[2], marker="o", alpha=0.5, picker=3,
#                             cmap=cmap)
#        else:
#            sc.remove()
#            sc = ax3.scatter(thetas[0], thetas[1], s=200, marker="o", alpha=0.5, picker=3)

    def onpick(event):
        """Handles the event from the :mod:`matplotlib` to select the points. It
        also handles the situation where several points overlap.

        :param event: A `matplotlib.backend_bases.Event`.
        """
        global index
        # handle the situation where several points overlap
#        if with_time:
#            print("Selected points:", event.ind)
#            min_z = numpy.min(times[event.ind])
#            candidates = event.ind[times[event.ind] == min_z]
#            candidates = event.ind
#        elif len(objectives) > 2:
#            print("Selected points:", event.ind)
#            # objectives are minimized (see objectives.py)
#            min_z = numpy.min(thetas[2][event.ind])
#            candidates = event.ind[thetas[2][event.ind] == min_z]
#        else:
#            candidates = event.ind
        candidates = event.ind
        print("Picking at random in", candidates)
        index = numpy.random.choice(candidates)

    fig.canvas.mpl_connect("pick_event", onpick)
    pyplot.show(block=True)
    # while pyplot.waitforbuttonpress():
    #     pass
    # pyplot.close()
    assert index is not None, "User did not pick any point!"
    return index

## Initial version, with no varying point size and color
#def select(thetas, objectives, with_time, times, figsize=(10, 10)):
#    """Asks the user to select the best option by clicking on the points from the
#    :mod:`matplotlib` figure. If several points overlap, select the one that minimizes
#    the time (or third objective).
#
#    :param thetas: A 2d-array of options sampled from the algorithms.
#    :param objectives: A list of objectives name.
#    :param with_time: (bool) Wheter of not to consider *times* as an objective.
#    :param times: An array of time for acquiring an image using each configuration in *thetas*.
#    :param figsize: The size of figure to display.
#    :return: The index of the selected point.
#    """
#    print("Asking user to select best option...")
#
#    fig = pyplot.figure(figsize=figsize)
#    ax = fig.gca()
#
#    # set to your favorite colormap (see https://matplotlib.org/users/colormaps.html)
#    cmap = pyplot.cm.get_cmap("nipy_spectral")
#
#    title = ax.set_title("Pick the best option by clicking on the point.")
#
#    # 3 points tolerance
#    if with_time:
#        sc = ax.scatter(thetas[0], thetas[1], s=100, c=times, marker="o", alpha=0.5, picker=3, cmap=cmap)
#        pyplot.colorbar(sc, ax=ax)
#    elif len(objectives) > 2:
#        sc = ax.scatter(thetas[0], thetas[1], s=100, c=thetas[2], marker="o", alpha=0.5, picker=3, cmap=cmap)
#        pyplot.colorbar(sc, ax=ax)
#    else:
#        ax.scatter(thetas[0], thetas[1], s=200, marker="o", alpha=0.5, picker=3)
#    ax.grid(True)
#    ax.set_xlabel(objectives[0].label)
#    ax.set_ylabel(objectives[1].label)
#
#    # handle adding custom ticks on mirror axis
#    new_ticks_x = objectives[0].mirror_ticks(ax.get_xticks())
#    if new_ticks_x is not None:
#        ax2 = ax.twiny()
#        ax2.set_xlim(ax.get_xlim())
#        ax2.set_xticklabels(new_ticks_x)
#        title.set_y(1.05)
#
#        if with_time:
#            sc.remove()
#            sc = ax2.scatter(thetas[0], thetas[1], s=100, c=times, marker="o", alpha=0.5, picker=3, cmap=cmap)
#        elif len(objectives) > 2:
#            sc.remove()
#            sc = ax2.scatter(thetas[0], thetas[1], s=100, c=thetas[2], marker="o", alpha=0.5, picker=3, cmap=cmap)
#        else:
#            sc.remove()
#            sc = ax2.scatter(thetas[0], thetas[1], s=200, marker="o", alpha=0.5, picker=3)
#
#    new_ticks_y = objectives[1].mirror_ticks(ax.get_yticks())
#    if new_ticks_y is not None:
#        ax3 = ax.twinx()
#        ax3.set_ylim(ax.get_ylim())
#        ax3.set_yticklabels(new_ticks_y)
#
#        if with_time:
#            sc.remove()
#            sc = ax3.scatter(thetas[0], thetas[1], s=100, c=times, marker="o", alpha=0.5, picker=3,
#                             cmap=cmap)
#        elif len(objectives) > 2:
#            sc.remove()
#            sc = ax3.scatter(thetas[0], thetas[1], s=100, c=thetas[2], marker="o", alpha=0.5, picker=3,
#                             cmap=cmap)
#        else:
#            sc.remove()
#            sc = ax3.scatter(thetas[0], thetas[1], s=200, marker="o", alpha=0.5, picker=3)
#
#    def onpick(event):
#        """Handles the event from the :mod:`matplotlib` to select the points. It
#        also handles the situation where several points overlap.
#
#        :param event: A `matplotlib.backend_bases.Event`.
#        """
#        global index
#        # handle the situation where several points overlap
#        if with_time:
#            print("Selected points:", event.ind)
#            min_z = numpy.min(times[event.ind])
#            candidates = event.ind[times[event.ind] == min_z]
#        elif len(objectives) > 2:
#            print("Selected points:", event.ind)
#            # objectives are minimized (see objectives.py)
#            min_z = numpy.min(thetas[2][event.ind])
#            candidates = event.ind[thetas[2][event.ind] == min_z]
#        else:
#            candidates = event.ind
#        print("Picking at random in", candidates)
#        index = numpy.random.choice(candidates)
#
#    fig.canvas.mpl_connect("pick_event", onpick)
#    pyplot.show(block=False)
#    while pyplot.waitforbuttonpress():
#        pass
#    pyplot.close()
#    assert index is not None, "User did not pick any point!"
#    return index


def give_score(confocal, sted, label, figsize=(10, 10)):
    """A :mod:`matplotlib` figure is shown to the user and asks to give a score
    on the STED image using the slider widget. The confocal image is shown as a
    comparaison to the user.

    :param confocal: A confocal image.
    :param sted: A sted image.
    :param label: The name of the objective.
    :param figsize: The size of figure to display.
    :return: The score of the image normalized between 0 and 1.
    """
    fig = pyplot.figure(figsize=figsize)
    fig.canvas.set_window_title("Score right-most image")

    ax = pyplot.axes([0.05, 0.15, 0.4, 0.75])
    ax.imshow(confocal, interpolation=None, cmap="gray", vmax=int(0.6*numpy.max(confocal)))
    ax.set_title("Confocal")
    ax.set_axis_off()

    ax = pyplot.axes([0.5, 0.15, 0.4, 0.75])
    im = ax.imshow(sted, interpolation=None, cmap="gray", vmax=int(0.6*numpy.max(sted)))
    ax.set_title("STED")
    ax.set_axis_off()

    ax = pyplot.axes([0.91, 0.15, 0.03, 0.75])
    pyplot.colorbar(im, cax=ax)

    axslider = fig.add_axes([0.5, 0.05, 0.4, 0.03])
    slider = widgets.Slider(axslider, label, 0, 100, valinit=50)

    pyplot.show(block=True)
    return slider.val / 100
