#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

# from numpy import *
import numpy as N
import os.path

def reshapeTo2D(data,axis):
    """Reshape data to 2D with specified axis as the 2nd dimension."""
    # get the shape, rank, and the length of the chosen axis
    dshape = data.shape
    rnk = len(dshape)
    n = dshape[axis]
    # convert negative axis to positive axis
    if axis < 0: 
        axis = axis + rnk
    # determine the new orde of the axes
    newdims = N.r_[0:axis,axis+1:rnk,axis]

    # reshape and transpose the data
    newdata = N.reshape(N.transpose(data,tuple(newdims)),(N.prod(dshape,axis=0)/n,n))
    
    # make sure we have a copy
    newdata = newdata.copy()

    return newdata

def reshapeFrom2D(data,axis,dshape):
    """Reshape data from 2D back to specified dshape."""

    # set the rank of the array
    rnk = len(dshape)

    # fix negative axis to be positive
    if axis < 0: 
        axis = axis + rnk

    # determine the dims from reshapeTo2D call
    newdims = N.r_[0:axis,axis+1:rnk,axis]

    # determine the transposed shape and reshape it back
    tdshape = N.take(dshape,newdims,0)
    ret = N.reshape(data,tuple(tdshape))

    # figure out how to retranspose the matrix
    vals = range(rnk)
    olddims = vals[:axis] + [rnk-1] +vals[axis:rnk-1]
    ret = N.transpose(ret,tuple(olddims))
    
    # make sure we have a copy
    ret = ret.copy()
    return ret

def repeat_to_match_dims(x,y,axis=-1):
    
    rnk = len(y.shape)
    
    # convert negative axis to positive axis
    if axis < 0: 
        axis = axis + rnk

    for d in range(axis)+range(axis+1,rnk):
        # add the dimension
        x = N.expand_dims(x,d)
        # repeat to fill that dim
        x = x.repeat(y.shape[d],d)

    return x


def deg2rad(degrees):
    """Convert degrees to radians."""
    return degrees/180.*N.math.pi

def rad2deg(radians):
    """Convert radians to degrees."""
    return radians/N.math.pi*180.

def pol2cart(theta,radius,z=None,radians=True):
    """Converts corresponding angles (theta), radii, and (optional) height (z)
    from polar (or, when height is given, cylindrical) coordinates
    to Cartesian coordinates x, y, and z.
    Theta is assumed to be in radians, but will be converted
    from degrees if radians==False."""
    if radians:
        x = radius*N.cos(theta)
        y = radius*N.sin(theta)
    else:
        x = radius*N.cos(deg2rad(theta))
        y = radius*N.sin(deg2rad(theta))
    if z is not None:
        # make sure we have a copy
        z=z.copy()
        return x,y,z
    else:
        return x,y

def cart2pol(x,y,z=None,radians=True):
    """Converts corresponding Cartesian coordinates x, y, and (optional) z
    to polar (or, when z is given, cylindrical) coordinates
    angle (theta), radius, and z.
    By default theta is returned in radians, but will be converted
    to degrees if radians==False."""    
    if radians:
        theta = N.arctan2(y,x)
    else:
        theta = rad2deg(N.arctan2(y,x))
    radius = N.hypot(x,y)
    if z is not None:
        # make sure we have a copy
        z=z.copy()
        return theta,radius,z
    else:
        return theta,radius

def lockFile(filename,lockdirpath=None,lockdirname=None):
    if lockdirname is None:
        lockdirname=filename+'.lock'
    if not(lockdirpath is None):
        lockdirname = lockdirpath+lockdirname
    if os.path.exists(lockdirname):
        return False
    else:
        try:
            os.mkdir(lockdirname)
        except:
            return False
    return True

def releaseFile(filename,lockdirpath=None,lockdirname=None):
    if lockdirname is None:
        lockdirname=filename+'.lock'
    if not(lockdirpath is None):
        lockdirname = lockdirpath+lockdirname
    try:
        os.rmdir(lockdirname)
    except:
        return False
    return True
      
def nextPow2(n):
    """Returns p such that 2 ** p >= n """
    p   = N.floor(N.log2(n))
    if 2 **  p ==  n:
        return p
    else:
        return p + 1

