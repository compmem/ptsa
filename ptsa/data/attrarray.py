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
    Subclass of NumPy's ndarray class that allows you to set custom
    array attributes both as kwargs during instantiation and on the
    fly.
    
    Try this on for size:

    x = AttrArray(N.random.rand(5), name='jubba')
    print x.name
    x.othername = 'wubba'
    print x.othername
    xs = N.sqrt(x)
    print x.othername
    
    """

    # required attributes (can be specified by subclasses)
    _required_attrs = []

    def __new__(cls, data, dtype=None, copy=True, **kwargs):
        if isinstance(data,AttrArray):
            # If data is already an AttrArray, we just need to worry
            # about dtype, copying, and any new attributes specified
            # in kwargs.
            # Use dtype of data, if nothing is specified:
            dtype2 = data.dtype
            if (dtype is None):
                dtype == dtype2
            if copy:
                # if dtype hasn't changed we can just copy data,
                # otherwise we need to recast data:
                if (dtype==dtype2):
                    result = data.copy()
                else:
                    result = data.astype(dtype)
                # update any attributes that may have been copied from
                # data with the new attributes specified in kwargs:
                newattrs = copylib.copy(kwargs)
                if (result._attrs is None):
                    result._attrs = newattrs
                else:
                    result._attrs.update(newattrs)
            else:
                # if dtype hasn't changed we can just copy data,
                # otherwise we need to recast data (NOTE: recasting
                # always produces a copy, so if a new dtype is
                # specified, we are always making a copy of data, even
                # if copy=False is specified):
                if (dtype == dtype2):
                    result = data
                else:
                    result = data.astype(dtype)
                # update any attributes that may already be present in
                # data with the new attributes specified in kwargs:
                if (result._attrs is None):
                    result._attrs = kwargs
                else:
                    result._attrs.update(kwargs)
        elif isinstance(data,N.ndarray):
            # If data is already a numpy ndarray, we just need to
            # produce a new view and take care of dtype, copying, and
            # any new attributes specified in kwargs.
            # Use dtype of data, if nothing is specified:
            dtype2 = data.dtype
            if (dtype is None):
                dtype == dtype2
            if copy:
                # if dtype hasn't changed we can just produce a view
                # of the copied data, otherwise we need to recast data
                # first:
                if (dtype==dtype2):
                    result = data.copy().view(cls)
                else:
                    result = data.astype(dtype).view(cls)
                # numpy ndarrays don't have an _attrs attribute, so we
                # can just copy the kwargs:
                result._attrs = copylib.copy(kwargs)
            else:
                # if dtype hasn't changed we can just produce a view
                # of the copied data, otherwise we need to recast data
                # first (NOTE: recasting always produces a copy, so if
                # a new dtype is specified, we are always making a
                # copy of data, even if copy=False is specified):
                if (dtype==dtype2):
                    result = data.view(cls)
                else:
                    result = data.astype(dtype).view(cls)
                # numpy ndarrays don't have an _attrs attribute, so we
                # can just assign the kwargs:
                result._attrs = kwargs
        else:
            # If data is not a numpy ndarray, we need to make an array from
            # it and produce a AttrArray view:
            if dtype is None:
                result = N.array(data,copy=copy).view(cls)
            else:
                result = N.array(data,dtype=dtype,copy=copy).view(cls)
            # Now assign any attributes:
            if copy:
                result._attrs = copylib.copy(kwargs)
            else:
                result._attrs = kwargs
        # Set all attributes in result._attr:
        result._setAllAttr()
        # Ensure that the required attributes are present:
        result._chkReqAttr()
        # Return the result:
        return result

    def __array_finalize__(self,obj):
        # XXX perhaps save the copy state and only copy if requested
        self._attrs = copylib.copy(getattr(obj, '_attrs', {}))
        self._setAllAttr()
    
    def __setattr__(self, name, value):
        # set the value in the attribute list
        #ret = super(self.__class__,self).__setattr__(name, value)
        if (value is None) and (name in self._required_attrs):
            raise AttributeError("Attribute "+name +" is required, and cannot "+
                                 "be set to None!")
        ret = N.ndarray.__setattr__(self, name, value)
        if name != '_attrs':
            # do add attrs to itself
            self._attrs[name] = value
        return ret

    def __delattr__(self, name):
        if name in self._required_attrs:
            raise AttributeError("Attribute "+name +" is required, and cannot "+
                                 "be deleted!")
        #ret = super(self.__class__,self).__delattr__(name)
        ret = N.ndarray.__delattr__(self, name)
        if self._attrs.has_key(name):
            del self._attrs[name]
        return ret

    def _setAllAttr(self):
        """
        Set all attributes in self._attr
        """
        if (not self._attrs is None):
            for tag in self._attrs:
                setattr(self, tag, self._attrs[tag])

    def _chkReqAttr(self):
        """
        Make sure the required attributes are set
        """
        for attr in self._required_attrs:
            if ((not self._attrs.has_key(attr)) or
                (self._attrs[attr] is None)):
                raise AttributeError("Attribute "+attr+" is required, and must"+
                                     " be provided for initialization!")        
