from numpy import *


def reshapeTo2D(data,axis):
    """Reshape data to 2D with specified axis as the 2nd dimension."""
    # get the shape, rank, and the length of the chosen axis
    dshape = data.shape
    rnk = len(dshape)
    N = dshape[axis]
    # convert negative axis to positive axis
    if axis < 0: 
        axis = axis + rnk
    # determine the new orde of the axes
    newdims = r_[0:axis,axis+1:rnk,axis]

    # reshape and transpose the data
    newdata = reshape(transpose(data,tuple(newdims)),(prod(dshape,axis=0)/N,N))
    
    # make sure we have a copy
    newdata = newdata.copy()

    return newdata

def reshapeFrom2D(data,axis,dshape):
    """Reshape data from 2D back to specified dshape."""
    # get the length of axis that was shifted to end
    N = dshape[axis]

    # set the rank of the array
    rnk = len(dshape)

    # fix negative axis to be positive
    if axis < 0: 
        axis = axis + rnk

    # determine the dims from reshapeTo2D call
    newdims = r_[0:axis,axis+1:rnk,axis]

    # determine the transposed shape and reshape it back
    tdshape = take(dshape,newdims,0)
    ret = reshape(data,tuple(tdshape))

    # figure out how to retranspose the matrix
    vals = range(rnk)
    olddims = vals[:axis] + [rnk-1] +vals[axis:rnk-1]
    ret = transpose(ret,tuple(olddims))
    
    # make sure we have a copy
    ret = ret.copy()
    return ret
