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
    Class that defines a dimension.  It has one required attribute
    (name), but you can also specify other custom attributes, such as
    units.
    """
    _required_attrs = ['name']
    
    def __new__(cls, data, name, copy=False, **kwargs):
        # set the kwargs to have name
        kwargs['name'] = name       
        # call the parent class's new
        return AttrArray.__new__(cls, data, copy=copy, **kwargs) #dim

    def __array_finalize__(self, obj):
        # take care of attributes:
        AttrArray.__array_finalize__(self,obj)
       
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
    Class that keeps track of the dimensions of a NumPy ndarray.  You
    must specify a list of Dim instances that match the number and
    size of the array.

    The DimArray class provides a number of conveniences above and
    beyond normal ndarrays.  These include the ability to refer to
    dimensions by name and to select subsets of the data based on
    complex queries using the dimension names.
    
    """
    _required_attrs = ['dims']
    
    def __new__(cls, data, dims, copy=False, **kwargs):

        # set the kwargs to have name
        kwargs['dims'] = dims
        
        # call the parent classes new
        return AttrArray.__new__(cls, data, copy=copy, **kwargs)

    def __array_finalize__(self,obj):
        # take care of attributes:
        self._attrs = copylib.copy(getattr(obj, '_attrs', {}))
        # Set all attributes:
        self._setAllAttr()
        # Ensure that the required attributes are present:
        self._chkReqAttr()
        #AttrArray.__array_finalize__(self,obj)

        # check that dims is a 1-D container:
        self._chkDimsDim()
        # check that dims only contains Dim instances:
        self._chkDimsClasses()
        # check that the lenghts of the Dim instances conform to the
        # shape of the array:
        self._chkDimsShape()

    def _chkDimsDim(self):
        """
        Ensure that the dims attribute is a 1-D container, by converting to a
        1-D array, if possible.
        """
        print self._attr
        self.dims = np.atleast_1d(self.dims)
        if self.dims.ndim > 1:
            dims_shape = tuple([x for x in self.dims.shape if x > 1])
            dims_ndim = len(dims_shape)
            if dims_ndim == 1:
                self.dims.shape = dims_shape
            else:
                raise ValueError("The dims attribute must be a 1-D container "+
                                 "of Dim instances!\ndims:\n"+str(self.dims))

    def _chkDimsClasses(self):
        """
        Ensure that all items in the dims attribute are Dim instances.
        """
        if not np.array([isinstance(x,Dim) for x in self.dims]).all():
            raise ValueError("The dims attribute must contain "+
                             "only Dim instances!\ndims:\n"+str(self.dims))
        
    def _chkDimsClasses(self):
        """
        Ensure that the lengths of the Dim instances match the array shape.
        """
        if self.shape != tuple([len(d) for d in self.dims]):
            raise ValueError("The length of the dims must match the shape of "+
                             "the DimArray!\nDimArray shape: "+str(self.shape)+
                             "\nShape of the dims:\n"+
                             str(tuple([len(d) for d in self.dims])))
