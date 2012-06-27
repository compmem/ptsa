#
# Pythonic fancy indexing of DimArrays!
#

import numpy as np
from dimarray import Dim,DimArray,AttrArray

if __name__ == "__main__":

    dims = [Dim(data=np.arange(20), name='time'),
            Dim(data=np.arange(10), name='freqs'),
            Dim(data=np.arange(30), name='events')]

    dat = DimArray(data=np.random.rand(20,10,30), dims=dims)

    # select some data
    ind = ((dat['time'] > 10) &
           ((dat['events']<10) | (dat['events']>20)) &
           (dat['freqs'].is_in(range(0,10,2))))

    subdat = dat[ind]

    print dat.shape
    print subdat.shape
