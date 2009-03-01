#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
import copy as copylib


#################################
# New array class with attributes
#################################

def __newobj__ ( cls, *args ):
    """ Unpickles new-style objects.
    """
    return cls.__new__( cls, *args )


class AttrArray(np.ndarray):
    """
    Attribute Array
    
    Subclass of NumPy's ndarray that allows specification of custom
    attributes both as kwargs during instantiation and on the fly.
    
    Examples
    --------
    x = AttrArray(np.random.rand(5), name='jubba')
    print x.name
    x.othername = 'wubba'
    print x.othername
    xs = np.sqrt(x)
    print x.othername
    """

    # required attributes (can be specified by subclasses): a
    # dictionary with with required attribute names as keys and the
    # required attribute data types as values. If no particular data
    # type is required, "object" should be specified. E.g.,
    # {'name':str} or {'misc':object}
    _required_attrs = None

    def __new__(cls, data, dtype=None, copy=False, **kwargs):
        # get the data in the proper format, copied if desired
        result = np.array(data, dtype=dtype, copy=copy)

        # transform the data to the new class
        result = result.view(cls)

        # get the new attrs, kwargs has priority
        # see if do deep copy of attrs
        newattrs = {}
        if hasattr(data,'_attrs'):
            # add those to the list of attributes
            if copy:
                newattrs = copylib.deepcopy(data._attrs)
            else:
                newattrs = data._attrs
        if copy:
            newattrs.update(copylib.deepcopy(kwargs))
        else:
            newattrs.update(kwargs)

        # Set and check all attributes:
        result._attrs = newattrs
        result._set_all_attr()
        result._chk_req_attr()

        return result
            
    
    def __array_finalize__(self,obj):
        if not hasattr(self, '_attrs'):
            self._attrs = copylib.deepcopy(getattr(obj, '_attrs', {}))
        # Set all attributes:
        self._set_all_attr()
        # Ensure that the required attributes are present:
        self._chk_req_attr()

    
    def __setattr__(self, name, value):
        # Do not allow changes to _required_attrs
        if name == '_required_attrs':
            raise AttributeError(
                "The attribute '_required_attrs' is read-only!")
        # set the value in the attribute list
        if self._required_attrs:
            if (self._required_attrs.has_key(name) and
                (not isinstance(value,self._required_attrs[name]))):
                raise AttributeError("Attribute '"+name +"' must be "+
                                     str(self._required_attrs[name])+
                                     "\nSupplied value and type:\n"+
                                     str(value)+"\n"+str(type(value)))

        # save whether it already existed
        # must do this before the call to ndarray.__setattr__
        if hasattr(self,name):
            attr_existed = True
        else:
            attr_existed = False

        # call the ndarray set attr
        ret = np.ndarray.__setattr__(self, name, value)

        # update the attrs if necessary
        # CTW: shouln't _attr be always updated?
        if self._attrs.has_key(name) or \
                (name != '_attrs' and not attr_existed):
            self._attrs[name] = value

        return ret

    def __delattr__(self, name):
        # Do not allow deletion of _required_attrs
        if name == '_required_attrs':
            raise AttributeError(
                "The attribute '_required_attrs' is read-only!")
        if name in self._required_attrs.keys():
            raise AttributeError("Attribute '"+name +"' is required, and cannot "+
                                 "be deleted!")
        ret = np.ndarray.__delattr__(self, name)
        if self._attrs.has_key(name):
            del self._attrs[name]
        return ret

    def _set_all_attr(self):
        """
        Set all attributes in self._attr
        """
        if self._attrs:
            for tag in self._attrs:
                setattr(self, tag, self._attrs[tag])

    def _chk_req_attr(self):
        """
        Make sure the required attributes are set
        """
        # if there are no required attributes, no check is required:
        if self._required_attrs is None: return
        
        for name in self._required_attrs.keys():
            if ((not self._attrs.has_key(name)) or
                (not isinstance(self._attrs[name], self._required_attrs[name]))):
                raise AttributeError("Attribute '"+name+"' is required, and "+
                                     "must be "+str(self._required_attrs[name]))
            

    def __repr__(self):
        # make the attribute kwargs list
        if len(self._attrs) > 0:
            attrstr = ', '.join([k+'='+repr(self._attrs[k]) \
                                 for k in self._attrs])
            retrepr = "AttrArray(%s, %s)" % \
                      (np.ndarray.__repr__(self),
                       attrstr)
        else:
            retrepr = "AttrArray(%s)" % \
                      (np.ndarray.__repr__(self))
            
        return retrepr
    
    def __reduce_ex__(self, protocol):
        """
        pickling function for classes which inherit from tuple.
        
        __reduce_ex__ must be overloaded for pickling to work. Refer to the docs
        in the pickle source code for details as to why.
        
        """
        # must save the _required_attrs and the _attrs
        state = (self._required_attrs, self._attrs,
                 super(AttrArray, self).__reduce_ex__(protocol))
        return ( __newobj__, ( self.__class__, ()), state )

    def __setstate__(self, state):
        """
        unpickling function
        """
        
        super(AttrArray, self).__setstate__(state[2][2])
        self._required_attrs = state[0]
        self._attrs = state[1]
        self._set_all_attr()

