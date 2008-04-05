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


###################################
# New array based dimension class
##################################

class NamedArray(N.ndarray):
    def __new__(subcls, data, name=None, dtype=None, copy=False):
        
        # ensure array, and copy the data if requested:
        subarr = N.array(data, dtype=dtype, copy=copy)

        # Transform 'subarr' from an ndarray to our new subclass
        subarr = subarr.view(subcls)

        # use the specified 'name' parameter if given:
        if name is not None:
            subarr.name = name
        # otherwise, use data's name attribute if it exists:
        elif hasattr(data, 'name'):
            subarr.name = data.name

        # Finally, we must return the newly created object:
        return subarr

    def __array_finalize__(self,obj):
        self.name = getattr(obj, 'name',None)
        
    def __repr__(self):
        desc="NamedArray: %(name)s\nData: %(data)s"""
        return desc % {'name': self.name, 'data':str(self)}   


class Dimension(NamedArray):
    """
    """
    def __new__(subcls, data, name, dtype=None, copy=False, **kwargs):
        # ensure array, and copy the data if requested:
        subarr = NamedArray(data, name, dtype=dtype, copy=copy)

        # Transform 'subarr' from an NamedArray to a Dimension
        subarr = subarr.view(subcls)

        # use the specified 'name' parameter if given:
        if name is not None:
            subarr.name = name
        # otherwise, use data's name attribute if it exists:
        elif hasattr(data, 'name'):
            subarr.name = data.name
        else:
            raise AttributeError(
                "A 'name' attribute must be specified for this class!")

        # Finally, we must return the newly created object:
        return subarr

    def __array_finalize__(self,obj):
        self.name = getattr(obj, 'name')
        
        
    def __repr__(self):
        desc="Dimension: %(name)s\nData: %(data)s"""
        return desc % {'name': self.name, 'data':str(self)}

