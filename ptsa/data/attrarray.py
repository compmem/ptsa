#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as N
import copy as copylib

###############################
# New array class with attributes
###############################

class AttrArray(N.ndarray):
    """
    Try this on for size:

    x = AttrArray(N.random.rand(5), name='jubba')
    print x.name
    x.othername = 'wubba'
    print x.othername
    xs = N.sqrt(x)
    print x.othername
    
    """

    # subclasses will fill this with things like 'name'
    _required_attrs = []

    def __new__(cls, data, copy=False, **kwargs):
        # copy the data if necessary
        if copy:
            result = data.copy()
        else:
            result = data

        # set the view of the result
        result = result.view(cls)

        # set the attrs, copying if necessary
        if copy:
            result._attrs = copylib.copy(kwargs)
        else:
            result._attrs = kwargs
        if not result._attrs is None:
            for tag in result._attrs:
                setattr(result, tag, result._attrs[tag])

        # make sure they set the required attributes
        for attr in cls._required_attrs:
            if not result._attrs.has_key(attr):
                 raise AttributeError, \
                       "Attribute %s is required to initialize dataset" % \
                       attr
        
        # return the result
        return result

    def __array_finalize__(self,obj):
        # XXX perhaps save the copy state and only copy if requested
        self._attrs = copylib.copy(getattr(obj, '_attrs', {}))
        for tag in self._attrs:
            setattr(self, tag, self._attrs[tag])
    
    def __setattr__(self, name, value):
        # set the value in the attribute list
        #ret = super(self.__class__,self).__setattr__(name, value)
        ret = N.ndarray.__setattr__(self, name, value)
        if name != '_attrs':
            # do add attrs to itself
            self._attrs[name] = value
        return ret

    def __delattr__(self, name):
        if name in self._required_attrs:
            raise AttributeError, \
                  "Attribute %s is required, so you can not delete it." % \
                  name
        #ret = super(self.__class__,self).__delattr__(name)
        ret = N.ndarray.__delattr__(self, name)
        if self._attrs.has_key(name):
            del self._attrs[name]
        return ret


