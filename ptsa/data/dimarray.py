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
    _required_attrs = ['name']
    
    def __new__(cls, data, name, copy=False, **kwargs):
        # set the kwargs to have name
        kwargs['name'] = name

        # XXX You can force to be 1D here
        
        # call the parent classes new
        return AttrArray.__new__(cls, data, copy, **kwargs)


class DimArray(AttrArray):
    """
    """
    _required_attrs = ['dims']
    
    def __new__(cls, data, dims, copy=False, **kwargs):

        # XXX do things to make sure dims are correct size

        # set the kwargs to have name
        kwargs['dims'] = dims
        
        # call the parent classes new
        return AttrArray.__new__(cls, data, copy, **kwargs)
