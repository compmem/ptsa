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
    dims : {numpy.ndarray or list of dimarray.Dim instances}
        The dimensions of the data.
    dtype : {numpy.dtype},optional
        The data type.
    copy : {bool},optional
        Flag specifying whether or not data should be copied.
    **kwargs : {**kwargs},optional
        Additional custom attributes.    
    """
    _required_attrs = {'dims':np.ndarray}
    dim_names = property(lambda self:
                  [dim.name for dim in self.dims],
                  doc="Dimension names (read only)")
    _dim_namesRE = property(lambda self:
                     re.compile('(?<!.)\\b'+
                     '\\b(?!.)|(?<!.)\\b'.join(self.dim_names)
                     + '\\b(?!.)'))
    T = property(lambda self: self.transpose())
    _skip_dim_check = False
    
    def __new__(cls, data, dims, dtype=None, copy=False, **kwargs):
        # Ensure that any array_like container of dims is turned into a
        # numpy.ndarray of Dim dtype:
        dimarr = np.empty(len(dims),dtype=Dim)
        dimarr[:] = dims
        # set the kwargs to have dims as an ndarray
        kwargs['dims'] = dimarr #np.array(dims,dtype=np.object)

        # make new AttrArray:
        dimarray = AttrArray(data,dtype=dtype,copy=copy,**kwargs)
        # convert to DimArray and return:
        return dimarray.view(cls)

    def __array_finalize__(self,obj):
        # call the AttrArray finalize
        AttrArray.__array_finalize__(self,obj)
        # ensure _getitem flag is off
        self._skip_dim_check = False
        # if this method is called from __getitem__, don't check dims
        # (they are adjusted later by __getitem__):
        if (isinstance(obj,DimArray) and obj._skip_dim_check): return
        # ensure that the dims attribute is valid:
        self._chkDims()

    def _chkDims(self):
        """
        Ensure that the dims attribute is a list of Dim instances that match the array shape.
        """
        # Ensure list:
#         if not isinstance(self.dims,list):
#             raise AttributeError("The dims attribute must be a list "+
#                              "of Dim instances!\ndims:\n"+str(self.dims))
        
        # Ensure that list is made up of only Dim instances:
        if not np.array([isinstance(x,Dim) for x in self.dims]).all():
            print [x.__class__ for x in self.dims]
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
        # process whether we using fancy string-based indices
        if isinstance(index,str):
            # see if it's just a single dimension name
            res = self._dim_namesRE.search(index)
            if res:
                # we have a single name, so return the
                # corresponding dimension
                return self.dims[self.dim_names.index(res.group())]
            else:
                # call find to get the new index from the string
                index = self.find(index)
        elif isinstance(index,tuple) and isinstance(index[0],str):
            # Use find to get the new index from the list of stings
            index = self.find(*index)

        # process the data
        self._skip_dim_check = True
        ret = np.ndarray.__getitem__(self,index)

        # process the dims if necessary
        if isinstance(ret,DimArray):
            # see which to keep and modify the rest
            tokeep = np.arange(len(self.dims))
            # turn into a tuple for easier processing
            if not isinstance(index,tuple):
                indlist = (index,)
            else:
                indlist = index
            for i,ind in enumerate(indlist):
                if isinstance(ind,int):
                    # remove that dimension
                    tokeep = tokeep[tokeep!=i]
                else:
                    ret.dims[i] = ret.dims[i][ind]
            # remove the empty dims
            ret.dims = ret.dims[tokeep]

        return ret

    def __getitem__old(self, index):
        # embedd in try block to ensure that _getitem flag is reset (in finally)
        print "in __getitem__(%s)" % (str(index))
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
                    #return self.select(index)
                    index = self.find(index)
            elif isinstance(index,tuple) and isinstance(index[0],str):
                index = self.find(index)
                    
            if isinstance(index,int):
                # a single int as index eliminates the first dimension:
                newdims = copylib.deepcopy(self.dims)
                newdims.pop(0)
            elif isinstance(index,slice) or isinstance(index,np.ndarray):
                # a single slice is taken over the first dimension:
                newdims = copylib.deepcopy(self.dims)
                newdims[0]=newdims[0][index]
            elif isinstance(index,tuple):
                # for tuples, if strs, send to select
                if isinstance(index[0],str):
                    # PBS: CTW, this is new, we need to add in catches
                    return self.select(*index)
                else:
                    # loop over the elements:
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
            self._skip_dim_check = True
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
            # reset the _skip_dim_check flag:
            self._skip_dim_check = False        

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

    def _split_bins(self, dim, bins, function, bin_labels,
                    error_on_nonexact, **kwargs):
        """
        Internal method for making bins when the number of bins or the
        indices at which to split the data into bins is
        specified. Cf. make_bins method.
        """
        # Determine which function to use for splitting:
        if error_on_nonexact:
            split = np.split
        else:
            split = np.array_split

        # Create the new dimension:
        split_dim = split(self.dims[dim],bins)
        if bin_labels == 'function':
            new_dim_dat = np.array([function(x,**kwargs) for x in split_dim])
            new_dim = Dim(new_dim_dat,self.dim_names[dim])
        elif bin_labels == 'sequential':
            new_dim = Dim(np.arange(len(split_dim)),
                          self.dim_names[dim])
        elif ((len(np.atleast_1d(bin_labels).shape) == 1) and
              (len(np.atleast_1d(bin_labels)) == bins)):
            new_dim = Dim(np.atleast_1d(bin_labels),
                          self.dim_names[dim])
        else:
            raise ValueError("Invalid value for bin_labels. Allowed values are "+
            "'function','sequential', or a 1-D list/array of labels of the same "+
            "length as bins.\n bins: "+str(bins)+"\n bin_labels: "+str(bin_labels))

        # Create the new data:
        split_dat = split(self.view(AttrArray),bins,axis=dim)
        new_dat = np.array([function(x,axis=dim,**kwargs) for x in split_dat])
        
        # Now the dimensions of the array need be re-arranged in the correct
        # order:
        dim_order = np.arange(len(new_dat.shape))
        dim_order[dim] = 0
        dim_order[0:dim] = np.arange(1,dim+1)
        dim_order[dim+1:len(new_dat.shape)] = np.arange(dim+1,len(new_dat.shape))
        new_dat = new_dat.transpose(dim_order)
        
        # Create and return new DimArray object:
        new_dims = copylib.deepcopy(self.dims)
        new_dims[dim] = new_dim
        new_attrs = self._attrs.copy()
        new_attrs['dims'] = new_dims
        return self.__class__(new_dat,**new_attrs)

    def _select_bins(self, dim, bins, function, bin_labels,
                     error_on_nonexact, **kwargs):
        """
        Internal method for making bins when the bins are specified as
        a list of intervals. Cf. make_bins method.
        """
        # Create the new dimension:
        dimbin_indx = np.array([((self.dims[dim]>=x[0]) &
                                 (self.dims[dim]<x[1])) for x in bins])
        if np.shape(bins[-1])[-1] == 3:
            new_dim_dat = np.array([x[2] for x in bins])
        elif bin_labels == 'function':
            new_dim_dat = np.array([function(x,**kwargs) for x in
                                    [self.dims[dim][indx] for indx in
                                     dimbin_indx]])
        elif bin_labels == 'sequential':
            new_dim_dat = np.arange(len(dimbin_indx))
        elif ((len(np.atleast_1d(bin_labels).shape) == 1) and
              (len(np.atleast_1d(bin_lables)) == bins)):
            new_dim_dat = np.altleast_1d(bin_labels)
        else:
            raise ValueError("Invalid value for bin_labels. Allowed values are "+
            "'function','sequential', or a 1-D list/array of labels of the same "+
            "length as bins.\n bins: "+str(bins)+"\n bin_labels: "+str(bin_labels))
        
        new_dim = Dim(data=new_dim_dat,name=self.dim_names[dim])
        
        # Create the new data:
        # We need to transpose the data array so that dim is the first
        # dimension. We store the new order of dimensions in totrans:
        totrans = range(len(self.shape))
        totrans[0] = dim
        totrans[dim] = 0
        
        # Now we are ready to do the transpose:
        tmpdata = self.copy()
        tmpdata = np.transpose(tmpdata.view(np.ndarray),totrans)
        
        # Now loop through each bin applying the function and concatenate the
        # data:
        new_dat = None
        for b,bin in enumerate(bins):
            bindata = function(tmpdata[dimbin_indx[b]],axis=0,**kwargs)
            if new_dat is None:
                new_dat = bindata[np.newaxis,:]
            else:
                new_dat = np.r_[new_dat,bindata[np.newaxis,:]]
        
        # transpose back:
        new_dat = new_dat.transpose(totrans)
        
        # Create and return new DimArray object:
        new_dims = copylib.deepcopy(self.dims)
        new_dims[dim] = new_dim
        new_attrs = self._attrs.copy()
        new_attrs['dims'] = new_dims
        return self.__class__(new_dat,**new_attrs)


    def make_bins(self,axis,bins,function,bin_labels='function',
                  error_on_nonexact=True,**kwargs):
        """
        Return a copy of the data with dimension (specified by axis)
        binned as specified.
        
        :Example usage:
        data.make_bins('time',10,numpy.mean,number_bins=False)
        data.make_bins('time',[[-100,0,'baseline'],[0,100,'timebin 1'],
                      [100,200,'timebin 2']],numpy.mean,number_bins=False)
                        
        :Parameters:
        - `axis`: The dimension to be binned. Can be name or number.
        - `bins`: Specifies how the data should be binned. Acceptable values
                  are:
                  * the number of bins (equally spaced, if possible, roughly
                    equally spaced if not and error_on_nonexact is False).
                    (Uses numpy.[array]split.)
                  * A 1-D container (list or tuple) of the indices where the
                    data should be split into bins. The value for
                    error_on_nonexact does not influence the result.
                    (Uses numpy.[array]split.)
                  * A 2-D container (lists or tuples) where each container in
                    the first dimension specifies the min (inclusive) and the max
                    (exlusive) values and (optionally) a label for each bin. The
                    value for error_on_nonexact must be True. If labels are
                    specified in bins, they are used and the value of bin_labels
                    is ignored.
        - `function`: The function to aggregate over within the bins. Needs to
                      take the data as the first argument and an additional axis
                      argument (numpy.mean is an example of a valid function).
        - `bins_labels` (optional): {'function','sequential',array_like}
                         'function' applies the function that is used for binning to the dimension,
                         'sequential' numbers the bins sequentially. Alternatively, a 1-D container
                         that contains the bin labels can be specified.
        - `error_on_nonexact` (optional): Specifies whether roughly equal bin sizes are
                               acceptable when the data cannot be evenly split in
                               the specified number of bins (this parameter is
                               only applicable when bins is an integer specifying
                               the number of bins). When True, the function
                               numpy.split is used, when False the function
                               numpy.array_split is used.
        - `kwargs` (optional): Optional key word arguments to be passed on to function.
        
        :Returns:
        A new DimArray instance in which one of the dimensions is binned as
        specified.
        """
        # Makes sure dim is index (convert dim name if necessary):
        dim = self.get_axis(axis)
        tmp_bins = np.atleast_2d(bins)
        if len(tmp_bins.shape)>2:
            raise ValueError('Invalid bins! Acceptable values are: number of '+
                             'bins, 1-D container of index values, 2-D '+
                             'container of min and max values and (optionally) '+
                             'a label for each bin. Provided bins: '+str(bins))
        if np.atleast_2d(bins).shape[1] == 1:
            return self._split_bins(dim,bins,function,bin_labels,
                               error_on_nonexact,**kwargs)
        elif np.atleast_2d(bins).shape[1] == 2:
            if not error_on_nonexact:
                raise ValueError('When bins are explicitly specified, '+
                                  'error_on_nonexact must be True. Provided '+
                                  'value: '+str(error_on_nonexact))
            return self._select_bins(dim,bins,function,bin_labels,
                                error_on_nonexact,**kwargs)
        elif np.atleast_2d(bins).shape[1] == 3:
            if bin_labels != 'function':
                raise ValueError('Simultaneously specification of bin labels '+
                                 'in bins and bin_labels is not allowed. '+
                                 'Provided bins: '+str(bins)+' Provided '+
                                 'bin_labels: '+ str(bin_labels))
            if not error_on_nonexact:
                raise ValueError('When bins are explicitly specified, '+
                                  'error_on_nonexact must be True. Provided '+
                                  'value: '+str(error_on_nonexact))
            return self._select_bins(dim,bins,function,bin_labels,
                                     error_on_nonexact,**kwargs)
        else:
            raise ValueError('Invalid bins! Acceptable values are: number of '+
                             'bins, 1-D container of index values, 2-D '+
                             'container of min and max values and (optionally) '+
                             'a label for each bin. Provided bins: '+str(bins))

    
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
            #ret.dims.pop(axis)
            ret.dims = ret.dims[np.arange(len(ret.dims))!=axis]
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
                #ret.dims.pop(d)
                ret.dims = ret.dims[np.arange(len(ret.dims))!=d]
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
            #ret.dims = [ret.dims[a] for a in axes]
            ret.dims = ret.dims[axes]
        else:
            ret = self.view(AttrArray).transpose()
            #ret.dims.reverse()
            ret.dims = ret.dims[-1::-1]
        return ret.view(self.__class__)

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).var(axis=axis, dtype=dtype, out=out, ddof=ddof)
        return self._ret_func(ret,axis)


# set the doc strings
castMsg =\
"""
*********************************************************************************
 ***  CAUTION: the output of this method is cast to an AttrArray instance. 
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

    
