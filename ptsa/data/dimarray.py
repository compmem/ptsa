#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import re
import numpy as N

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

        # XXX You can force to be 1D here
        
        # call the parent class's new
        return AttrArray.__new__(cls, data, copy, **kwargs)


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

        # XXX do things to make sure dims are correct size

        # set the kwargs to have name
        kwargs['dims'] = dims
        
        # call the parent classes new
        return AttrArray.__new__(cls, data, copy, **kwargs)
