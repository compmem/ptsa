#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
import copy as copylib
from ptsa.data.attrarray import AttrArray

###############################
# New dimensioned array class
###############################

class Dim(AttrArray):
    """
    Dim(data, name, dtype=None, copy=False, **kwargs)
    
    Class that defines a dimension.  It has one required attribute
    (name), but other custom attributes (e.g., units) can be specified.

    Parameters
    ----------
    data : {1-D array_like}
        The values of the dimension (e.g., time points for a time dimension)
    name : {object}
        The name of the dimension (e.g., 'time')
    dtype : {numpy.dtype},optional
        The data type.
    copy : {bool},optional
        Flag specifying whether or not data should be copied.
    **kwargs : {**kwargs},optional
        Additional custom attributes (e.g., units='ms').
    """
    _required_attrs = ['name']
    
    def __new__(cls, data, name, dtype=None, copy=True, **kwargs):
        # set the kwargs to have name
        kwargs['name'] = name
        # make new AttrArray:
        dim = AttrArray(data,dtype,copy,**kwargs)
        # convert to Dim and return:
        return dim.view(cls)

    def __array_finalize__(self, obj):
        # XXX perhaps save the copy state and only copy if requested
        self._attrs = copylib.copy(getattr(obj, '_attrs', {}))
        # Set all attributes:
        self._setAllAttr()
        # Ensure that the required attributes are present:
        self._chkReqAttr()
        
        # make sure the data is 1-D:
        if self.ndim == 1: # if 1-D, return
            return
        # massage the array into 1-D if possible:
        elif self.ndim > 1:
            # using self.squeeze() can lead to nasty recursions and
            # 0-D arrays so we do it by hand:
            newshape = tuple([x for x in self.shape if x > 1])
            ndim = len(newshape)
            if ndim == 1:
                self.shape = newshape
                return
            else:
                raise ValueError("Dim instances must be 1-dimensional!\ndim:\n"+
                                 str(self))
        # if the array is 0-D, make it 1-D:
        elif self.ndim == 0:
            self.shape = (1)
            return
        else:
            # This would require negative self.ndim which would
            # indicate a serious bug in ndarray.
            raise ValueError("Invalid number of dimensions!")



class DimArray(AttrArray):
    """
    DimArray(data, dims, dtype=None, copy=False, **kwargs)

    Class that keeps track of the dimensions of a NumPy ndarray.  The
    dimensions are specified in the dims attribute as a list of Dim
    instances that match the shape of the data array.

    The DimArray class provides a number of conveniences above and
    beyond normal ndarrays.  These include the ability to refer to
    dimensions by name and to select subsets of the data based on
    complex queries using the dimension names.

    Parameters
    ----------
    data : {array_like}
        The dimensioned data.
    dims : {list of Dim instances}
        The dimensions of the data.
    dtype : {numpy.dtype},optional
        The data type.
    copy : {bool},optional
        Flag specifying whether or not data should be copied.
    **kwargs : {**kwargs},optional
        Additional custom attributes.    
    """
    _required_attrs = ['dims']
    
    def __new__(cls, data, dims, dtype=None, copy=True, **kwargs):
        # set the kwargs to have name
        kwargs['dims'] = dims
        # make new AttrArray:
        dimarray = AttrArray(data,dtype,copy,**kwargs)
        # convert to DimArray and return:
        return dimarray.view(cls)


    def __array_finalize__(self,obj):
        # XXX perhaps save the copy state and only copy if requested
        self._attrs = copylib.copy(getattr(obj, '_attrs', {}))
        # Set all attributes:
        self._setAllAttr()
        # Ensure that the required attributes are present:
        self._chkReqAttr()
        # ensure that the dims attribute is valid:
        self._chkDims()
 
    def _chkDims(self):
        """
        Ensure that the dims attribute is a list of Dim instances that match the array shape.
        """
        # Ensure list:
        if not isinstance(self.dims,list):
            raise ValueError("The dims attribute must be a list "+
                             "of Dim instances!\ndims:\n"+str(self.dims))
        # Ensure that list is made up of only Dim instances:
        if not np.array([isinstance(x,Dim) for x in self.dims]).all():
            raise ValueError("The dims attribute must contain "+
                             "only Dim instances!\ndims:\n"+str(self.dims))
        # Ensure that the lengths of the Dim instances match the array shape:
        if self.shape != tuple([len(d) for d in self.dims]):
            raise ValueError("The length of the dims must match the shape of "+
                             "the DimArray!\nDimArray shape: "+str(self.shape)+
                             "\nShape of the dims:\n"+
                             str(tuple([len(d) for d in self.dims])))
