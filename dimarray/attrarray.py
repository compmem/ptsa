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

class AttrArray(np.ndarray):
    """
    AttrArray(data, dtype=None, copy=False, **kwargs)

    An AttrArray (short for Attribute Array) is simply a NumPy ndarray
    which allows the specification of custom attributes. These
    attributes can be specified as keyword arguments or set and
    changed on the fly as shown in the examples below.

    AttrArray instances are initialized just like ndarray instances
    but they accept arbitrary keyword arguments that can be used to
    specify custom attributes during initialization. Every AttrArray
    has a protected (read-only) _required_attrs attribute, which is
    None when no attributes are required (as is the case for instances
    of AttrArray) or a dictionary that specifies required attributes
    (for child classes of AttrArray, such as Dim and DimArray).
    
    Examples
    --------
    >>> import numpy as np
    >>> import dimarray as da
    >>> data = da.AttrArray(np.random.rand(5), hello='world')
    >>> print data.hello
    world
    >>> data.hello = 'good bye'
    >>> print data.hello
    good bye
    >>> data.version = 1.0
    >>> print data.version
    1.0

    These custom attributes are maintained when copying or
    manipulating the data in an AttrArray:
    
    >>> data2 = data.mean()
    >>> data2.hello
    good bye    
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
        # PBS: I don't think we need to call this here
        #self._chk_req_attr()

    
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
            

#     def __repr__(self):
#         # make the attribute kwargs list
#         if len(self._attrs) > 0:
#             attrstr = ', '.join([k+'='+repr(self._attrs[k]) \
#                                  for k in self._attrs])
#             retrepr = "AttrArray(%s, %s)" % \
#                       (np.ndarray.__repr__(self),
#                        attrstr)
#         else:
#             retrepr = "AttrArray(%s)" % \
#                       (np.ndarray.__repr__(self))
            
#         return retrepr

    def __reduce__(self):
        # reduced state as ndarray
        object_state = list(np.ndarray.__reduce__(self))

        # append the custom object attributes
        subclass_state = (self._attrs,)
        object_state[2] = (object_state[2],subclass_state)

        # convert back to tuple and return
        return tuple(object_state)
    
    def __setstate__(self,state):
        # get the ndarray state and the subclass state
        nd_state, own_state = state

        # refresh the ndarray state
        np.ndarray.__setstate__(self,nd_state)

        # get the subclass attributes
        attrs, = own_state

        # set the attributes
        #self._required_attrs = req_attrs
        self._attrs = attrs
        self._set_all_attr()

    def nanstd(a, axis=None, dtype=None, ddof=0):
        """
        Compute the standard deviation along the specified axis
        ignoring nans.

        Returns the standard deviation, a measure of the spread of a
        distribution, of the array elements, treating nans as missing
        values. The standard deviation is computed for the flattened
        array by default, otherwise over the specified axis.
        
        Parameters
        ----------
        a : array_like
            Calculate the standard deviation of these values.
        axis : int, optional
            Axis along which the standard deviation is computed. The
            default is to compute the standard deviation of the
            flattened array.
        ddof : int, optional
            Means Delta Degrees of Freedom.  The divisor used in
            calculations is ``N - ddof``, where ``N`` represents the
            number of elements.  By default `ddof` is zero (biased
            estimate).
            
        Returns
        -------
        standard_deviation : {ndarray, scalar}
            If `out` is None, return a new array containing the
            standard deviation, otherwise return a reference to the
            output array. Dtype is always at least np.float64.

        See Also
        --------
        numpy.std : Standard deviation
        scipy.stats.nanstd : Standard deviation ignorning nans
        numpy.var : Variance
        numpy.mean : Average

        Notes
        -----
        If no nan values are present, returns the same value as
        numpy.std, otherwise, the standard deviation is calculated as
        if the nan values were not present.
        
        The standard deviation is the square root of the average of
        the squared deviations from the mean, i.e., ``var =
        sqrt(mean(abs(x - x.mean())**2))``.

        The mean is normally calculated as ``x.sum() / N``, where ``N
        = len(x)``.  If, however, `ddof` is specified, the divisor ``N
        - ddof`` is used instead.

        Note that, for complex numbers, std takes the absolute value
        before squaring, so that the result is always real and
        nonnegative.

        This docstring is based on that for numpy.std, the code is
        based on scipy.stats.nanstd.

        Examples
        --------
        >>> a = AttrArray([[1, 2], [3, 4], [5, 6]])
        >>> a.nanstd()
        1.707825127659933
        >>> a = AttrArray([[np.nan, 2], [3, 4], [5, 6]])
        >>> a.nanstd()
        1.4142135623730951
        >>> a.std(0)
        AttrArray([ 1.,  1.6329931618554521])
        >>> a.std(1)
        AttrArray([ 0.0,  0.5,  0.5])
        """
        
        if axis is None:
            return a[~np.isnan(a)].std()

        # make copy to not change the input array:
        a = a.copy()

        # number of all observations
        n_orig = a.shape[axis]

        # number of nans:
        n_nan = np.float64(np.sum(np.isnan(a),axis))

        # number of non-nan values:
        n = n_orig - n_nan

        # compute the mean for all non-nan values:
        a[np.isnan(a)] = 0.
        m1 = np.sum(a,axis)/n

        # Kludge to subtract m1 from the correct axis
        if axis!=0:
            shape = np.arange(a.ndim).tolist()
            shape.remove(axis)
            shape.insert(0,axis)
            a = a.transpose(tuple(shape))
            d = (a-m1)**2.0
            shape = tuple(np.array(shape).argsort())
            d = d.transpose(shape)
        else:
            d = (a-m1)**2.0

        # calculate numerator for variance:
        m2 = np.float64(np.sum(d,axis)-(m1*m1)*n_nan)
        
        # devide by appropriate denominator:
        m2c = m2 / (n - ddof)
        return np.sqrt(m2c)

