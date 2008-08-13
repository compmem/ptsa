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
import re

from attrarray import AttrArray

###############################
# New dimensioned array class
###############################

class Dim(AttrArray):
    """
    Dim(data, name, dtype=None, copy=False, **kwargs)
    
    Class that defines a dimension.  It has one required attribute
    (name), but other custom attributes (e.g., units) can be specified.

    Parameters
    ----------
    data : {1-D array_like}
        The values of the dimension (e.g., time points for a time dimension)
    name : {object}
        The name of the dimension (e.g., 'time')
    dtype : {numpy.dtype},optional
        The data type.
    copy : {bool},optional
        Flag specifying whether or not data should be copied.
    **kwargs : {**kwargs},optional
        Additional custom attributes (e.g., units='ms').
    """
    _required_attrs = {'name':str}
    
    def __new__(cls, data, name, dtype=None, copy=False, **kwargs):
        # set the kwargs to have name
        kwargs['name'] = name
        # make new AttrArray:
        dim = AttrArray(data,dtype=dtype,copy=copy,**kwargs)
        # convert to Dim and return:
        return dim.view(cls)

    def __array_finalize__(self, obj):
        AttrArray.__array_finalize__(self,obj)
        # XXX perhaps save the copy state and only copy if requested
        #self._attrs = copylib.copy(getattr(obj, '_attrs', {}))
        # Set all attributes:
        #self._setAllAttr()
        # Ensure that the required attributes are present:
        #self._chkReqAttr()
        
        #self._getitem = False
        #if (isinstance(obj, Dim) and obj._getitem): return

        # make sure the data is 1-D:
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
            elif ndim == 0:
                self.shape = (1,)
            else:
                raise ValueError("Dim instances must be 1-dimensional!\ndim:\n"+
                                 str(self)+"\nnewshape:",newshape)
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
    DimArray(data, dims, dtype=None, copy=False, **kwargs)

    Class that keeps track of the dimensions of a NumPy ndarray.  The
    dimensions are specified in the dims attribute as a list of Dim
    instances that match the shape of the data array.

    The DimArray class provides a number of conveniences above and
    beyond normal ndarrays.  These include the ability to refer to
    dimensions by name and to select subsets of the data based on
    complex queries using the dimension names.

    Parameters
    ----------
    data : {array_like}
        The dimensioned data.
    dims : {list of Dim instances}
        The dimensions of the data.
    dtype : {numpy.dtype},optional
        The data type.
    copy : {bool},optional
        Flag specifying whether or not data should be copied.
    **kwargs : {**kwargs},optional
        Additional custom attributes.    
    """
    _required_attrs = {'dims':list}
    dim_names = property(lambda self:
                  [dim.name for dim in self.dims],
                  doc="Dimension names (read only)")
    _dim_namesRE = property(lambda self:
                     re.compile('(?<!.)\\b'+
                     '\\b(?!.)|(?<!.)\\b'.join(self.dim_names)
                     + '\\b(?!.)'))
    T = property(lambda self: self.transpose())
    
    def __new__(cls, data, dims, dtype=None, copy=False, **kwargs):
        # set the kwargs to have name
        kwargs['dims'] = dims
        # make new AttrArray:
        dimarray = AttrArray(data,dtype=dtype,copy=copy,**kwargs)
        # convert to DimArray and return:
        return dimarray.view(cls)

    def __array_finalize__(self,obj):
        # call the AttrArray finalize
        AttrArray.__array_finalize__(self,obj)
        # ensure _getitem flag is off
        self._getitem = False
        # if this method is called from __getitem__, don't check dims
        # (they are adjusted later by __getitem__):
        if (isinstance(obj,DimArray) and obj._getitem): return
        # ensure that the dims attribute is valid:
        self._chkDims()

    def _chkDims(self):
        """
        Ensure that the dims attribute is a list of Dim instances that match the array shape.
        """
        # Ensure list:
        if not isinstance(self.dims,list):
            raise AttributeError("The dims attribute must be a list "+
                             "of Dim instances!\ndims:\n"+str(self.dims))
        
        # Ensure that list is made up of only Dim instances:
        if not np.array([isinstance(x,Dim) for x in self.dims]).all():
            raise AttributeError("The dims attribute must contain "+
                             "only Dim instances!\ndims:\n"+str(self.dims))
        
        # Ensure that the lengths of the Dim instances match the array shape:
        if self.shape != tuple([len(d) for d in self.dims]):
            raise AttributeError("The length of the dims must match the shape of "+
                             "the DimArray!\nDimArray shape: "+str(self.shape)+
                             "\nShape of the dims:\n"+
                             str(tuple([len(d) for d in self.dims])))
        
        # Ensure unique dimension names:
        if len(np.unique(self.dim_names)) != len(self.dim_names):
            raise AttributeError("Dimension names must be unique!\nnames: "+
                                 str(self.dim_names))



    def _select_ind(self,*args,**kwargs):
        """
        Returns a tuple of index arrays for the selected conditions and an array
        of Boolean index arrays.     
        """
        # get starting indicies
        ind = [np.ones(dim.shape,np.bool) for dim in self.dims]

        # process the args
        for arg in args:
            # arg must be a string
            filterStr = arg

            # figure out which dimension we're dealing with
            foundDim = False
            for d,k in enumerate(self.dim_names):
                # RE makes sure to not replace substrings
                if re.search(r'\b'+k+r'\b',filterStr) is not None:
                    # this is our dimension
                    foundDim = True

                    # replace the string
                    filterStr = re.sub(r'\b'+k+r'\b','self["'+k+'"]',filterStr)

                    # get the new index
                    newind = eval(filterStr)
                    
                    # apply it to the dimension index
                    ind[d] = ind[d] & newind

                    # break this loop to continue the next
                    #break
            # if we get to here, the provided string did not specify any dimensions
            if not foundDim:
                # XXX eventually this should be a custom exception
                raise ValueError("The provided filter string did not specify "+
                                 "any valid dimensions: "+str(filterStr))
            
        # loop over the kwargs
        for key,value in kwargs.iteritems():
            if key in self.dim_names:
                # get the proper dimension to cull
                d = self.dim_names.index(key)
                ind[d] = ind[d] & value

        # create the final master index
        m_ind = np.ix_(*ind)

        return m_ind,ind


    def __getitem__(self, index):
        # embedd in try block to ensure that _getitem flag is reset (in finally)
        try:
            if isinstance(index,str):
                # see if it's just a single dimension name
                res = self._dim_namesRE.search(index)
                if res:
                    # we have a single name, so return the
                    # corresponding dimension
                    return self.dims[self.dim_names.index(res.group())]
                else:
                    # call select to do the work
                    return self.select(index)
            elif isinstance(index,int):
                # a single int as index eliminates the first dimension:
                newdims = copylib.deepcopy(self.dims)
                newdims.pop(0)
            elif isinstance(index,slice) or isinstance(index,np.ndarray):
                # a single slice is taken over the first dimension:
                newdims = copylib.deepcopy(self.dims)
                newdims[0]=newdims[0][index]
            elif isinstance(index,tuple):
                # for tuples, loop over the elements:
                newdims = copylib.deepcopy(self.dims)
                adj_i = 0 # adjusted index (if dimensions are eliminated)
                for i,ind in enumerate(index):
                    if isinstance(ind,int):
                        # eliminate respective dim and update adj_i accordingly:
                        newdims.pop(adj_i)
                        adj_i -= 1
                    elif isinstance(ind,slice) or isinstance(ind,np.ndarray):
                        # apply the slice or array to the respective dimension
                        newdims[adj_i] = newdims[adj_i][ind]
                    else: # not sure if there are other legitimate indices here
                        raise NotImplementedError("This index is not (yet?) "+
                                                  " implemented!",type(ind),
                                                  str(ind),str(i),str(adj_i),
                                                  type(index),str(index))
                    # increment adjusted index:
                    adj_i += 1
            else: # not sure if there are other legitimate indices here
                raise NotImplementedError("This index is not (yet?) "+
                                          "implemented!",type(index),str(index))
            
            # Now that the dimensions are updated, we need to get the data:
            # set _getitem flag for __array_finalize__:
            self._getitem = True
            # get the data:
            ret = np.ndarray.__getitem__(self,index)
            # if the resulting data is scalar, return it:
            if ret.ndim == 0:
                return ret
            else: # othewise, adjust the dimensions:
                # set new dimensions:
                ret.dims = newdims
                # finalize the new array and return:
                ret.__array_finalize__(ret)
                return ret            
        finally:
            # reset the _getitem flag:
            self._getitem = False        

    def find(self,*args,**kwargs):
        """
        Returns a tuple of index arrays for the selected conditions. 

        data.find('time>0','events.recalled==True')
        or
        data.find(time=data['time']>0,events=data['events'].recalled==True)
        or 
        data.find("time>kwargs['t']","events.recalled==kwargs['val']",t=0,val=True)

        data[ind], where ind is the return value of the find method
        and data.select(filterstring) return the same slices provided
        that the same filterstring is used.
        """
        m_ind,ind = self._select_ind(*args,**kwargs)
        return m_ind

    def select(self,*args,**kwargs):
        """
        Return a slice of the data filtered with the select conditions.

        data.select('time>0','events.recalled==True')
        or
        data.select(time=data['time']>0,events=data['events'].recalled==True)
        or 
        data.select("time>kwargs['t']","events.recalled==kwargs['val']",t=0,val=True)

        To get a tuple of index arrays for the selected conditions use the find method.
        """
        m_ind,ind = self._select_ind(*args,**kwargs)
        return self[m_ind]

    def _split_bins(self, dim, bins, function, unit, bin_labels,
                    dim_unit, error_on_nonexact, **kwargs):
        
        if error_on_nonexact:
            split = np.split
        else:
            split = np.array_split

        split_dim = split(self.dims[dim],bins)
        if bin_labels == 'function':
            new_dim_dat = np.array([function(x,**kwargs) for x in split_dim])
            new_dim = Dim(new_dim_dat,self.dim_names[dim],unit=dim_unit)
        elif bin_labels == 'sequential':
            new_dim = Dim(np.arange(len(split_dim)),
                          self.dim_names[dim], unit=dim_unit)
        elif ((len(np.atleast_1d(bin_labels).shape) == 1) and
              (len(np.atleast_1d(bin_labels)) == bins)):
            new_dim = Dim(np.atleast_1d(bin_labels),
                          self.dim_names[dim], unit=dim_unit)
        else:
            raise ValueError("Invalid value for bin_labels. Allowed values are "+
            "'function','sequential', or a 1-D list/array of labels of the same "+
            "length as bins.\n bins: "+str(bins)+"\n bin_labels: "+str(bin_labels))

        split_dat = split(self.view(AttrArray),bins,axis=dim)
        #new_dat = np.array([function(x,axis=dim,**kwargs) for x in split_dat])
        for n,x in enumerate(split_dat):
            self.view(AttrArray)[self.dim_names[n]==self.dims[n]
                                 ] = function(x,axis=dim,**kwargs)
        #Transpose data!
        self.view(AttrArray).dims[dim] = new_dim
        self.view(self.__class__)

    def _select_bins(self, dim, bins, function, unit, bin_labels,
                     dim_unit, error_on_nonexact, **kwargs):
        dimbin_indx = np.array([((self.dims[dim]>=x[0]) &
                                 (self.dims[dim]<x[1])) for x in bins])

        if np.shape(bins[-1])[-1] == 3:
            new_dim_dat = np.array([x[2] for x in bins])
        elif bin_labels == 'function':
            new_dim_dat = np.array([function(x,**kwargs) for x in
                                    [self.dims[dim][indx] for indx in
                                     dimibin_indx]])
        elif bin_labels == 'sequential':
            new_dim_dat = np.arange(len(dimbin_indx))
        elif ((len(np.atleast_1d(bin_labels).shape) == 1) and
              (len(np.atleast_1d(bin_lables)) == bins)):
            new_dim_dat = np.altleast_1d(bin_labels)
        else:
            raise ValueError("Invalid value for bin_labels. Allowed values are "+
            "'function','sequential', or a 1-D list/array of labels of the same "+
            "length as bins.\n bins: "+str(bins)+"\n bin_labels: "+str(bin_labels))

        new_dim = Dim(new_dim_dat,self.dim_names[dim],unit=dim_unit)
        ## Unfinished
                 

    def get_axis(self,axis):
        """
        Get the axis number for a dimension name.

        Provides a convenient way to ensure an axis number, because it
        converts dimension names to axis numbers, but returns
        non-string input unchanged. Should only be needed in
        exceptional cases outside a class definition as any function
        that takes an axis keyword should also accept the
        corresponding dimension name.

        Parameters
        __________
        axis : {str}
            The name of a dimension.
            
        Returns
        _______
        The axis number corresponding to the dimension name.
        If axis is not a string, it is returned unchanged.        
        """
        if isinstance(axis,str):
            # must convert to index dim
            axis = self.dim_names.index(axis)
        return axis
             
    def _ret_func(self, ret, axis):
        """
        Return function output for functions that take an axis
        argument after adjusting dims properly.
        """
        if axis is None:
            # just return what we got
            return ret.view(AttrArray)
        else:
            # pop the dim
            ret.dims.pop(axis)
            return ret.view(self.__class__)
    
    
    def all(self, axis=None, out=None):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).all(axis=axis, out=out)
        return self._ret_func(ret,axis)
    
    def any(self, axis=None, out=None):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).any(axis=axis, out=out)
        return self._ret_func(ret,axis)        

    def argmax(self, axis=None, out=None):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).argmax(axis=axis, out=out)
        return self._ret_func(ret,axis)        

    def argmin(self, axis=None, out=None):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).argmin(axis=axis, out=out)
        return self._ret_func(ret,axis)        

    def argsort(self, axis=-1, kind='quicksort', order=None):
        if axis is None:
            return self.view(AttrArray).argsort(axis=axis, kind=kind,
                                                 order=order)
        else:
            axis = self.get_axis(axis)
            ret = self.view(AttrArray).argsort(axis=axis, kind=kind, order=order)
            return ret.view(self.__class__)

    def compress(self, condition, axis=None, out=None):
        if axis is None:
            return self.view(AttrArray).compress(condition, axis=axis, out=out)
        else:
            axis = self.get_axis(axis)
            ret = self.view(AttrArray).compress(condition, axis=axis, out=out)
            cnd = np.array(condition)
            ret.dims[axis] = ret.dims[axis][cnd]
            return ret.view(self.__class__)

    def cumprod(self, axis=None, dtype=None, out=None):
        if axis is None:
            return self.view(AttrArray).cumprod(axis=axis, dtype=dtype,
                                                  out=out)
        else:
            axis = self.get_axis(axis)
            ret = self.view(AttrArray).cumprod(axis=axis, dtype=dtype, out=out)
            return ret.view(self.__class__)

    def cumsum(self, axis=None, dtype=None, out=None):
        if axis is None:
            return self.view(AttrArray).cumsum(axis=axis, dtype=dtype,
                                                out=out)
        else:
            axis = self.get_axis(axis)
            ret = self.view(AttrArray).cumsum(axis=axis, dtype=dtype, out=out)
            return ret.view(self.__class__)

    def diagonal(self, *args, **kwargs):
        return self.view(AttrArray).diagonal(*args, **kwargs)

    def flatten(self, *args, **kwargs):
        return self.view(AttrArray).flatten(*args, **kwargs)

    def max(self, axis=None, out=None):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).max(axis=axis, out=out)
        return self._ret_func(ret,axis)

    def mean(self, axis=None, dtype=None, out=None):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).mean(axis=axis, dtype=dtype, out=out)
        return self._ret_func(ret,axis)

    def min(self, axis=None, out=None):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).min(axis=axis, out=out)
        return self._ret_func(ret,axis)

    def nonzero(self, *args, **kwargs):
        return self.view(AttrArray).nonzero(*args, **kwargs)

    def prod(self, axis=None, dtype=None, out=None):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).prod(axis=axis, dtype=dtype, out=out)
        return self._ret_func(ret,axis)

    def ptp(self, axis=None, out=None):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).ptp(axis=axis, out=out)
        return self._ret_func(ret,axis)

    def ravel(self, *args, **kwargs):
        return self.view(AttrArray).ravel(*args, **kwargs)

    def repeat(self, repeats, axis=None):
        if axis is None:
            return self.view(AttrArray).repeat(repeats=repeats, axis=axis)
        else:
            axis = self.get_axis(axis)
            ret = self.view(AttrArray).repeat(repeats, axis=axis)
            ret.dims[axis] = ret.dims[axis].repeat(repeats)
            return ret.view(self.__class__)

    def reshape(self, shape, order='C'):
        return np.reshape(self.view(AttrArray),shape,order)

    def resize(self, *args, **kwargs):
        """Resizing is not possible for dimensioned arrays. Calling
        this method will throw a NotImplementedError exception. If
        resizing is desired the array needs to be converted to a
        different data type (e.g., numpy.ndarray), first!"""
        raise NotImplementedError("Resizing is not possible for dimensioned "+
                                  "arrays. Convert to (e.g.) numpy.ndarray!")

    def sort(self, axis=-1, kind='quicksort', order=None):
        if axis is None:
            raise ValueError("Please specify an axis! To sort a flattened "+
                             "array convert to (e.g.) numpy.ndarray.")
        axis = self.get_axis(axis)
        self.view(AttrArray).sort(axis=axis, kind=kind, order=order)
        self.view(self.__class__)
        self.dims[axis].sort(axis=axis, kind=kind, order=order)

    def squeeze(self):
        ret = self.view(AttrArray).squeeze()
        d = 0
        for dms in ret.dims:
            if len(ret.dims[d]) == 1:
                ret.dims.pop(d)
            else:
                d += 1
        return ret.view(self.__class__)

    def std(self, axis=None, dtype=None, out=None, ddof=0):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).std(axis=axis, dtype=dtype, out=out, ddof=0)
        return self._ret_func(ret,axis)

    def sum(self, axis=None, dtype=None, out=None):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).sum(axis=axis, dtype=dtype, out=out)
        return self._ret_func(ret,axis)

    def swapaxes(self, axis1, axis2):
        axis1 = self.get_axis(axis1)
        axis2 = self.get_axis(axis2)
        ret = self.view(AttrArray).swapaxes(axis1,axis2)
        tmp = ret.dims[axis1]
        ret.dims[axis1] = ret.dims[axis2]
        ret.dims[axis2] = tmp
        return ret.view(self.__class__)

    def take(self, indices, axis=None, out=None, mode='raise'):
        if axis is None:
            return self.view(AttrArray).take(indices, axis=axis, out=out, mode=mode)
        else:
            axis = self.get_axis(axis)
            ret = self.view(AttrArray).take(indices, axis=axis, out=out, mode=mode)
            ret.dims[axis] = ret.dims[axis].take(indices, axis=0, out=out, mode=mode)
            return ret.view(self.__class__)
        
    def trace(self, *args, **kwargs):
        return self.view(AttrArray).trace(*args, **kwargs)

    def transpose(self, *axes):
        axes = np.squeeze(axes)
        if len(axes.shape)==len(self):
            axes = [self.get_axis(a) for a in axes]
            ret = self.view(AttrArray).transpose(*axes)
            ret.dims = [ret.dims[a] for a in axes]
        else:
            ret = self.view(AttrArray).transpose()
            ret.dims.reverse()
        return ret.view(self.__class__)

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).var(axis=axis, dtype=dtype, out=out, ddof=ddof)
        return self._ret_func(ret,axis)


# set the doc strings
castMsg =\
"""
*********************************************************************************
 ***  CAUTION: the output of this method is downcast to AttrArray. 
  *   Some attributes may no longer be valid after this Method is applied!\n\n"""
DimArray.all.im_func.func_doc = np.ndarray.all.__doc__            
DimArray.any.im_func.func_doc = np.ndarray.any.__doc__            
DimArray.argmax.im_func.func_doc = np.ndarray.argmax.__doc__            
DimArray.argmin.im_func.func_doc = np.ndarray.argmin.__doc__            
DimArray.mean.im_func.func_doc = np.ndarray.mean.__doc__            
DimArray.argsort.im_func.func_doc = np.ndarray.argsort.__doc__            
DimArray.compress.im_func.func_doc = np.ndarray.compress.__doc__            
DimArray.cumprod.im_func.func_doc = np.ndarray.cumprod.__doc__            
DimArray.cumsum.im_func.func_doc = np.ndarray.cumsum.__doc__            
DimArray.diagonal.im_func.func_doc = castMsg+np.ndarray.diagonal.__doc__            
DimArray.flatten.im_func.func_doc = castMsg+np.ndarray.flatten.__doc__            
DimArray.max.im_func.func_doc = np.ndarray.max.__doc__            
DimArray.mean.im_func.func_doc = np.ndarray.mean.__doc__            
DimArray.min.im_func.func_doc = np.ndarray.min.__doc__            
DimArray.nonzero.im_func.func_doc = np.ndarray.nonzero.__doc__            
DimArray.prod.im_func.func_doc = np.ndarray.prod.__doc__            
DimArray.ptp.im_func.func_doc = np.ndarray.ptp.__doc__            
DimArray.ravel.im_func.func_doc = castMsg+np.ndarray.ravel.__doc__            
DimArray.repeat.im_func.func_doc = np.ndarray.repeat.__doc__            
DimArray.reshape.im_func.func_doc = castMsg+np.ndarray.reshape.__doc__            
#DimArray.resize.im_func.func_doc = castMsg+np.ndarray.resize.__doc__            
DimArray.sort.im_func.func_doc = np.ndarray.sort.__doc__            
DimArray.squeeze.im_func.func_doc = np.ndarray.squeeze.__doc__            
DimArray.std.im_func.func_doc = np.ndarray.std.__doc__            
DimArray.sum.im_func.func_doc = np.ndarray.sum.__doc__            
DimArray.swapaxes.im_func.func_doc = np.ndarray.swapaxes.__doc__            
DimArray.take.im_func.func_doc = np.ndarray.take.__doc__            
DimArray.trace.im_func.func_doc = castMsg+np.ndarray.trace.__doc__            
DimArray.transpose.im_func.func_doc = np.ndarray.transpose.__doc__            
DimArray.var.im_func.func_doc = np.ndarray.var.__doc__            

    
