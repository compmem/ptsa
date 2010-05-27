#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


import numpy as np
import pylab as pl

def errorfill(xvals,yvals,errvals,**kwargs):
    """
    Plot an errorbar as a filled polygon that can be transparent.

    See the pylab.fill method for kwargs.
    """
    # set the xrange
    x_range = np.concatenate((xvals,np.flipud(xvals)))
    y_range = np.concatenate((yvals+errvals,np.flipud(yvals-errvals)))

    # do the fill
    return pl.fill(x_range,y_range,**kwargs)
