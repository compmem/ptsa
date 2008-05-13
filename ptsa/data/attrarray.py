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

#################################
# New array class with attributes
#################################

class TestArray(np.ndarray):
    def __new__(cls, data, info):
        # ensure ndarray
        result = np.array(data)

        # transform the data to the new class
        result = result.view(cls)

        # set the custom attribute
        result.info = info

        # return new custom array
        return result

    def __array_finalize__(self, obj):
        # provide info for what's happening
        print "finalize:\t%s\n\t\t%s" % (self.__class__, obj.__class__)
        # set the custom attribute
        self.info = getattr(obj,'info','')
        # provide more info
        if hasattr(obj,'info'):
            print "\t\t%s : %s" % (self.info, obj.info)
        else:
            print "\t\t%s : None" % (self.info)


class AttrArray(np.ndarray):
    """
    Subclass of NumPy's ndarray class that allows you to set custom
    array attributes both as kwargs during instantiation and on the
    fly.
    
    Try this on for size:

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
        # always do deep copy of attrs
        newattrs = {}
        if hasattr(data,'_attrs'):
            # add those to the list of attributes
            newattrs = copylib.deepcopy(data._attrs)
        newattrs.update(copylib.deepcopy(kwargs))

        # Set and check all attributes:
        result._attrs = newattrs
        result._setAllAttr()
        result._chkReqAttr()

        return result
            
    
    def __array_finalize__(self,obj):
        # XXX perhaps save the copy state and only copy if requested
        if not hasattr(self, '_attrs'):
            self._attrs = copylib.deepcopy(getattr(obj, '_attrs', {}))

        # Set all attributes:
        self._setAllAttr()

        # Ensure that the required attributes are present:
        self._chkReqAttr()

    
    def __setattr__(self, name, value):
        # set the value in the attribute list
        #ret = super(self.__class__,self).__setattr__(name, value)
        #if (value is None) and (name in self._required_attrs.keys()):
        if not (self._required_attrs is None):
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
        if self._attrs.has_key(name) or \
                (name != '_attrs' and not attr_existed):
            self._attrs[name] = value

        return ret

    def __delattr__(self, name):
        if name in self._required_attrs.keys():
            raise AttributeError("Attribute '"+name +"' is required, and cannot "+
                                 "be deleted!")
        ret = np.ndarray.__delattr__(self, name)
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
        # if there are no required attributes, no check is required:
        if self._required_attrs is None: return
        
        for name in self._required_attrs.keys():
            if ((not self._attrs.has_key(name)) or
                (not isinstance(self._attrs[name], self._required_attrs[name]))):
                raise AttributeError("Attribute '"+name+"' is required, and "+
                                     "must be "+str(self._required_attrs[name]))
            


