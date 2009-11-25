"""
Logo design inspired by the matplotlib logo by Tony Yu <tsyu80@gmail.com>.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab

mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.edgecolor'] = 'gray'

axalpha = 0.05
#figcolor = '#EFEFEF'
figcolor = 'white'
dpi = 80
fig = plt.figure(figsize=(3, 1.1),dpi=dpi)
fig.figurePatch.set_edgecolor(figcolor)
fig.figurePatch.set_facecolor(figcolor)


def add_timeseries():
    ax = fig.add_axes([0., 0., 1., 1.])
    x = np.linspace(0,1,1000)
    freqs = [8,16,32,64]
    # y = np.zeros(1000)
    # for f in freqs:
    #     y = y + np.sin(x*np.pi*f*4 + f/60.)*(10.0/(f))
    # y = y+.5
    y = np.sin(x*np.pi*32)*.45 + .5
    lines = plt.plot(x,y,
                     transform=ax.transAxes,
                     color="#11557c", alpha=0.25,)

    ax.set_axis_off()
    return ax

def add_ptsa_text(ax):
    ax.text(0.95, 0.5, 'ptsa', color='#11557c', fontsize=65,
               ha='right', va='center', alpha=1.0, transform=ax.transAxes)

def add_pizza():
    ax = fig.add_axes([0.025, 0.075, 0.3, 0.85], polar=True, resolution=50)

    ax.axesPatch.set_alpha(axalpha)
    ax.set_axisbelow(True)
    N = 8
    arc = 2. * np.pi
    theta = np.arange(0.0, arc, arc/N)
    radii = 10 * np.array([0.82, 0.76, 0.81, 0.77, 0.79, 0.83, 0.8, 0.78])
    width = np.pi / 4 * np.array([1.0]*N)
    bars = ax.bar(theta, radii, width=width, bottom=0.0)
    for r, bar in zip(radii, bars):
        bar.set_facecolor(cm.hot(r/10.))
        bar.set_edgecolor('r')
        bar.set_alpha(0.6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_visible(False)

    for line in ax.get_ygridlines() + ax.get_xgridlines():
        line.set_lw(0.8)
        line.set_alpha(0.9)
        line.set_ls('-')
        line.set_color('0.5')

    ax.set_yticks(np.arange(1, 9, 2))
    ax.set_rmax(9)

if __name__ == '__main__':
    main_axes = add_timeseries()
    add_pizza()
    add_ptsa_text(main_axes)
    plt.show()


