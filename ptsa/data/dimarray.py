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


###############################
# New dimensioned array class
###############################


class DimArray(N.ndarray):
    """
    """
    def __new__(cls, data, dims, copy=False, **kwargs):
        # copy the data if necessary
        if copy:
            result = data.copy()
        else:
            result = data

        # set the view of the result
        result = result.view(cls)

        # set the dims as an attr
        result._attr = kwargs
        result._attr['dims'] = dims

        # loop and add kwargs as attributes
        # maybe leave for finalize if that's what it's supposed to do

        # do I need to copy or not?

        # return the result
        return result

    def __array_finalize__(self,obj):
        # XXX change this to copy the _attr then make them all attr
        for tag in ['_attr']:
            setattr(self, tag, copylib.copy(getattr(obj, tag, None)))

