#
#
#

import numpy as np
from dimarray import Dim,DimArray,AttrArray

class DimIndex(tuple):
    def __new__(typ, ind, bool_ind, parent):
        res = tuple.__new__(typ, ind)
        res._bool_ind = bool_ind
        res._parent = parent        
        return res
    
    def __and__(self, other):
        # compare each bool
        # check other is DimIndex
        ind = []
        for l,r in zip(self._bool_ind,other._bool_ind):
            ind.append(l&r)
        return DimIndex(np.ix_(*ind),ind,self._parent)

    def __or__(self, other):
        # compare each bool
        # check other is DimIndex
        ind = []
        for l,r in zip(self._bool_ind,other._bool_ind):
            ind.append(l|r)
        return DimIndex(np.ix_(*ind),ind,self._parent)

    def __xor__(self, other):
        # compare each bool
        # check other is DimIndex
        ind = []
        for l,r in zip(self._bool_ind,other._bool_ind):
            ind.append(l^r)
        return DimIndex(np.ix_(*ind),ind,self._parent)

class DimSelect():
    def __init__(self, name, parent):
        # set the kwargs to have dims as an ndarray
        self._name = name
        self._parent = parent

    def __lt__(self, other):
        # get starting indicies
        ind = [np.ones(dim.shape, dtype=np.bool) for dim in self._parent.dims]

        # do the comparison along the desired dimension
        ind[self._parent.dim_names.index(self._name)] = self._parent[self._name] < other

        # create the final master index from the list of filtered indices
        return DimIndex(np.ix_(*ind),ind,self._parent)

    def __gt__(self, other):
        # get starting indicies
        ind = [np.ones(dim.shape, dtype=np.bool) for dim in self._parent.dims]

        # do the comparison along the desired dimension
        ind[self._parent.dim_names.index(self._name)] = self._parent[self._name] > other

        # create the final master index from the list of filtered indices
        return DimIndex(np.ix_(*ind),ind,self._parent)


if __name__ == "__main__":

    dims = [Dim(data=np.arange(20), name='time'),
            Dim(data=np.arange(10), name='freqs'),
            Dim(data=np.arange(30), name='events')]

    dat = DimArray(data=np.random.rand(20,10,30), dims=dims)

    da = {}
    for name in dat.dim_names:
        da[name] = DimSelect(name,dat)
    ind = ((da['time'] > 10) &
           ((da['events']<10) | (da['events']>20)))
    subdat = dat[ind]

    print dat.shape
    print subdat.shape
