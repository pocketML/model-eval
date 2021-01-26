import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

import read_data

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def plot(dimensions, entries, results, title, ticks, yscale="linear"):
    N = len(dimensions)
    theta = radar_factory(N, frame='circle')#'polygon')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='radar')

    plt.yscale(yscale)
    ax.set_rgrids(ticks)

    for r in results:
        ax.plot(theta, r) #, color=colors[i])
        #ax.fill(theta, r, alpha=0.25) #, facecolor=colors[i])
    ax.set_varlabels(dimensions)
    plt.yticks()

    ax.legend(entries, bbox_to_anchor=(1.05, 1), loc='upper left', labelspacing=0.1, fontsize='small')

    ax.set_title(title)
    #fig.text(0.5, 0.965, title,
    #         horizontalalignment='center', color='black', weight='bold',
    #         size='large')

    plt.show()

if __name__ == '__main__':
    languages, taggers, results = read_data.get_accuracy_data(True)
    ticks = np.arange(0.0, 100.0, step=5.0)
    title = 'Token accuracy for 9 taggers across all 10 languages'
    plot(languages, taggers, results, title, ticks)
    #metrics, taggers, results = read_data.get_size_data()
    #for r in results:
    #    print(r)
    #ticks = [10**i for i in range(7)]
    #plot(metrics, taggers, results, 'Sizes', ticks, 'log')