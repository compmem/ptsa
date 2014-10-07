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
import re

from attrarray import AttrArray

###############################
# New dimensioned array class
###############################

class Dim(AttrArray):
    """
    Dim(data, name, dtype=None, copy=False, **kwargs)

    Dim is a child class of AttrArray with the constraints that each
    instance be 1-dimensional and have a name attribute. If multi
    dimensional input is specified during initialization, an attempt
    is made to convert it to one dimension by collapsing over
    dimensions that only have one level (if that fails an error is
    raised).

    Parameters
    ----------
    data : {1-D array_like}
        The values of the dimension (e.g., time points for a time
        dimension)
    name : str
        The name of the dimension (e.g., 'time'). When used in a
        DimArray this must be a valid identifier name (i.e., consist
        only of letters, numbers and underscores and not start with a
        number).
    dtype : numpy.dtype, optional
        The data type.
    copy : bool, optional
        Flag specifying whether or not data should be copied.
    **kwargs : {key word arguments}, optional
        Additional custom attributes (e.g., units='ms').

    Examples
    --------
    >>> import numpy as np
    >>> import dimarray as da
    >>> test = da.Dim([[1,2,3]], name='dimension_1')
    >>> print test
    [1 2 3]
    """
    _required_attrs = {'name':str}
    
    def __new__(cls, data, name=None, dtype=None, copy=False, **kwargs):
        if name is None:
            # if 'name' is not specified see if data already has a
            # name attribute:
            name = getattr(data,'name',None)
        if name is None:
            raise AttributeError("A 'name' attribute must be specified!")
        # set the kwargs to have name
        kwargs['name'] = name

        # make new AttrArray:
        dim = AttrArray(data,dtype=dtype,copy=copy,**kwargs)

        # check the dim
        if dim.ndim > 1:
            # using self.squeeze() can lead to nasty recursions and
            # 0-D arrays so we do it by hand:
            newshape = tuple([x for x in dim.shape if x > 1])
            ndim = len(newshape)
            if ndim == 1:
                dim.shape = newshape
            elif ndim == 0:
                dim.shape = (1,)
            else:
                raise ValueError("Dim instances must be 1-dimensional!\ndim:\n"+
                                 str(dim)+"\nnewshape:",newshape)
        # if the array is 0-D, make it 1-D:
        elif dim.ndim == 0:
            dim.shape = (1,)
        
        #if dim.shape[0] != np.unique(np.asarray(dim)).shape[0]:
        #    raise ValueError("Data for Dim objects must be unique!")

        # convert to Dim and return:
        return dim.view(cls)

class DimIndex(tuple):
    """
    Tuple representing a fancy index of a Dim along with its siblings.
    """
    def __new__(typ, ind, bool_ind):
        res = tuple.__new__(typ, ind)
        res._bool_ind = bool_ind
        return res
    
    def __and__(self, other):
        # compare each bool
        # check other is DimIndex
        ind = []
        for l,r in zip(self._bool_ind,other._bool_ind):
            ind.append(l&r)
        return DimIndex(np.ix_(*ind),ind)

    def __or__(self, other):
        # compare each bool
        # check other is DimIndex
        ind = []
        for l,r in zip(self._bool_ind,other._bool_ind):
            ind.append(l|r)
        return DimIndex(np.ix_(*ind),ind)

    def __xor__(self, other):
        # compare each bool
        # check other is DimIndex
        ind = []
        for l,r in zip(self._bool_ind,other._bool_ind):
            ind.append(l^r)
        return DimIndex(np.ix_(*ind),ind)

class DimSelect(Dim):
    """
    Dim that supports boolean comparisons for fancy indexing.
    """
    _required_attrs = {'name':str,
                       '_parent_dim_shapes':list,
                       '_parent_dim_index':int}
    
    def __new__(cls, dim, parent_dim_shapes, parent_dim_index):

        # verify that dim is a Dim instance
        
        # set the kwargs to have what we need
        kwargs = {}
        kwargs['name'] = dim.name
        kwargs['_parent_dim_shapes'] = parent_dim_shapes
        kwargs['_parent_dim_index'] = parent_dim_index

        # creat the new instance
        ds = Dim(dim, **kwargs)

        # convert to DimSelect and return
        return ds.view(cls)

    def __lt__(self, other):
        # get starting indicies
        ind = [np.ones(shape, dtype=np.bool) for shape in self._parent_dim_shapes]

        # do the comparison along the desired dimension
        ind[self._parent_dim_index] = np.asarray(self) < other

        # create the final master index from the list of filtered indices
        return DimIndex(np.ix_(*ind),ind)

    def __le__(self, other):
        # get starting indicies
        ind = [np.ones(shape, dtype=np.bool) for shape in self._parent_dim_shapes]

        # do the comparison along the desired dimension
        ind[self._parent_dim_index] = np.asarray(self) <= other

        # create the final master index from the list of filtered indices
        return DimIndex(np.ix_(*ind),ind)

    def __gt__(self, other):
        # get starting indicies
        ind = [np.ones(shape, dtype=np.bool) for shape in self._parent_dim_shapes]

        # do the comparison along the desired dimension
        ind[self._parent_dim_index] = np.asarray(self) > other

        # create the final master index from the list of filtered indices
        return DimIndex(np.ix_(*ind),ind)

    def __ge__(self, other):
        # get starting indicies
        ind = [np.ones(shape, dtype=np.bool) for shape in self._parent_dim_shapes]

        # do the comparison along the desired dimension
        ind[self._parent_dim_index] = np.asarray(self) >= other

        # create the final master index from the list of filtered indices
        return DimIndex(np.ix_(*ind),ind)

    def __eq__(self, other):
        # get starting indicies
        ind = [np.ones(shape, dtype=np.bool) for shape in self._parent_dim_shapes]

        # do the comparison along the desired dimension
        ind[self._parent_dim_index] = np.asarray(self) == other

        # create the final master index from the list of filtered indices
        return DimIndex(np.ix_(*ind),ind)

    def __ne__(self, other):
        # get starting indicies
        ind = [np.ones(shape, dtype=np.bool) for shape in self._parent_dim_shapes]

        # do the comparison along the desired dimension
        ind[self._parent_dim_index] = np.asarray(self) != other

        # create the final master index from the list of filtered indices
        return DimIndex(np.ix_(*ind),ind)

    def is_in(self, vals):
        """
        Elementwise boolean check for membership in a list.
        """
        # get starting indicies
        ind = [np.ones(shape, dtype=np.bool) for shape in self._parent_dim_shapes]

        # do the comparison along the desired dimension
        ind[self._parent_dim_index] = np.lib.arraysetops.in1d(np.asarray(self),vals)
        # i = self._parent_dim_index
        # self_array = np.asarray(self)
        # ind[i] = False
        # for val in vals:
        #     ind[i] = ind[i] | (self_array == val)

        # create the final master index from the list of filtered indices
        return DimIndex(np.ix_(*ind),ind)

    def index(self, index):
        """
        Index into elements along the dimension.
        """
        # get starting indicies
        ind = [np.ones(shape, dtype=np.bool) 
               for shape in self._parent_dim_shapes]

        # make sure 1d
        index = np.atleast_1d(index)

        if issubclass(index.dtype.type, np.bool_):
            # start with ones and apply logical &
            ind[self._parent_dim_index] = [self._parent_dim_index] & index
        else:
            # start with zeros and index into it
            ind[self._parent_dim_index][:] = False
            ind[self._parent_dim_index][index] = True

        # create the final master index from the list of filtered indices
        return DimIndex(np.ix_(*ind),ind)


class DimArray(AttrArray):
    """
    DimArray(data, dims=None, dtype=None, copy=False, **kwargs)
    
    A DimArray (short for Dimensioned Array) is a child class of
    AttrArray with the constraints that each instance and have a dims
    attribute which specifies the dimensions as an array of Dim
    instances. The name of the Dim instances in dims must be unique
    and they must correspond to the dimensions of the DimArray in the
    correct order. If dims is not specified, generic dimensions will
    be automatically generated from the data.

    The DimArray class provides a number of conveniences above and
    beyond normal ndarrays.  These include the ability to refer to
    dimensions by name and to select subsets of the data based on
    complex queries using the dimension names.
    
    Parameters
    ----------
    data : array_like
        The dimensioned data.
    dims : {container of Dim instances}, optional
        The dimensions of the data.
    dtype : dtype,optional
        The data type.
    copy : bool,optional
        Flag specifying whether or not data should be copied.
    **kwargs : {key word arguments}, optional
        Additional custom attributes.

    Examples
    --------
    >>> import numpy as np
    >>> import dimarray as da
    >>> dim1 = da.Dim(range(5),'Dim1')
    >>> dim2 = da.Dim(['a','b','c'],'Dim2')
    >>> data = da.DimArray(np.random.rand(5,3),[dim1,dim2])
    >>> data
    DimArray([[ 0.59645979,  0.92462876,  0.76882167],
           [ 0.3581822 ,  0.57873905,  0.76889117],   
           [ 0.40272846,  0.69372032,  0.59006832],   
           [ 0.69862889,  0.68334188,  0.10891802],   
           [ 0.14111733,  0.97454223,  0.73193147]])  
    >>> data.dims
    array([[0 1 2 3 4], ['a' 'b' 'c']], dtype=object)
    >>> data['Dim1']
    Dim([0, 1, 2, 3, 4])
    >>> data['Dim2']
    Dim(['a', 'b', 'c'],
          dtype='|S1')
    >>> data.dim_names
    ['Dim1', 'Dim2']
    >>> data.mean('Dim1')
    DimArray([ 0.43942333,  0.77099445,  0.59372613])
    >>> np.mean(data,'Dim1')
    DimArray([ 0.43942333,  0.77099445,  0.59372613])
    >>> data['Dim1 > 2']
    DimArray([[ 0.69862889,  0.68334188,  0.10891802],
           [ 0.14111733,  0.97454223,  0.73193147]])
    >>> data["Dim2 == 'c'"]
    DimArray([ 0.76882167,  0.76889117,  0.59006832,  0.10891802,  0.73193147])
    >>> data['Dim1 > 2',"Dim2 == 'a'"]
    DimArray([ 0.69862889,  0.14111733])
    >>> data['Dim1>2'] +=1
    >>> data
    DimArray([[ 0.59645979,  0.92462876,  0.76882167],
           [ 0.3581822 ,  0.57873905,  0.76889117],
           [ 0.40272846,  0.69372032,  0.59006832],
           [ 1.69862889,  1.68334188,  1.10891802],
           [ 1.14111733,  1.97454223,  1.73193147]])
    >>> data['Dim1>2',"Dim2>'a'"]
    DimArray([[ 1.68334188,  1.10891802],
            [ 1.97454223,  1.73193147]])
    >>> data['Dim1>2',"Dim2>'a'"]+=1
    >>> data
    DimArray([[ 0.59645979,  0.92462876,  0.76882167],
           [ 0.3581822 ,  0.57873905,  0.76889117],
           [ 0.40272846,  0.69372032,  0.59006832],
           [ 1.69862889,  2.68334188,  2.10891802],
           [ 1.14111733,  2.97454223,  2.73193147]])
    >>> data = da.DimArray(np.random.rand(4,5))
    >>> data.dim_names
    ['dim1', 'dim2']
    >>> data.dims
    array([[0 1 2 3], [0 1 2 3 4]], dtype=object)
    """
    _required_attrs = {'dims':np.ndarray}
    dim_names = property(lambda self:
                  [dim.name for dim in self.dims],
                  doc="""
                  List of dimension names

                  This is a property that is created from the dims
                  attribute and can only be changed through changes in
                  dims.

                  Examples
                  --------
                  >>> import numpy as np
                  >>> import dimarray as da
                  >>> dim1 = da.Dim(range(5),'First dimension')
                  >>> dim2 = da.Dim(['a','b','c'],'Second dimension')
                  >>> data = da.DimArray(np.random.rand(5,3),[dim1,dim2])
                  >>> data.dim_names
                  ['First dimension', 'Second dimension']
                  """)
    _dim_namesRE = property(lambda self:
                     re.compile('(?<!.)\\b'+
                     '\\b(?!.)|(?<!.)\\b'.join(self.dim_names)
                     + '\\b(?!.)'))
    T = property(lambda self: self.transpose())
    _skip_dim_check = False
    _valid_dimname = re.compile('[a-zA-Z_]\w*')

    def __new__(cls, data, dims=None, dtype=None, copy=False, **kwargs):
        if isinstance(data,str):
            data_shape = (0,)
        else:
            # got to make sure data is array-type
            data = np.asanyarray(data)
            data_shape = data.shape
        
        # see how to process dims
        if dims is None:
            # fill with default values
            dims = []
            for i,dlen in enumerate(data_shape):
                dims.append(Dim(np.arange(dlen), 'dim%d'%(i+1)))

        # Ensure that any array_like container of dims is turned into a
        # numpy.ndarray of Dim dtype:
        dimarr = np.empty(len(dims),dtype=Dim)
        dimarr[:] = dims

        # set the kwargs to have dims as an ndarray
        kwargs['dims'] = dimarr

        # make new AttrArray parent class
        dimarray = AttrArray(data,dtype=dtype,copy=copy,**kwargs)
        
        # View as DimArray and return:
        return dimarray.view(cls)

    def __array_finalize__(self,obj):
        # Check if obj is None (implies it's unpickling)
        # We must find out if there are other instances of it being None
        if obj is None:
            return
        # catch case where we multiply ndarray by a scaler
        # and the dimensions do not match
        if obj.shape == (1,) and self.shape != (1,):
            self = self.view(np.ndarray)
            return
        # call the AttrArray finalize
        AttrArray.__array_finalize__(self,obj)
        # ensure _skip_dim_check flag is off
        self._skip_dim_check = False
        # if this method is called with _skip_dim_check == True, don't
        # check dims (they need to be adjusted by whatever method
        # called __array_finalize__ with this flag set):
        if (isinstance(obj,DimArray) and obj._skip_dim_check):
            return

        # ensure that the dims attribute is valid:
        self._chk_dims()

    def _chk_dims(self):
        """
        Ensure that the dims attribute is a list of Dim instances that
        match the array shape.
        """
        # loop over the dims and make sure they are valid
        for i,d in enumerate(self.dims):
            # make sure it's a dim
            if not isinstance(d,Dim):
                raise AttributeError("The dims attribute must contain "+
                                     "only Dim instances!\ndim %d: %s\n" % \
                                     (i,str(type(d))))
            # make sure it is unique
            #if d.shape[0] != np.unique(np.asarray(d)).shape[0]:
            #    raise ValueError("Data for Dim objects must be unique!")
        
        # Ensure that the lengths of the Dim instances match the array shape:
        if self.shape != tuple([len(d) for d in self.dims]):
            raise AttributeError("The length of the dims must match the shape" +
                                 " of the DimArray!\nDimArray shape: "+
                                 str(self.shape)+"\nShape of the dims:\n"+
                                 str(tuple([len(d) for d in self.dims])))
        
        # Ensure unique dimension names (this will fail if not all Dims)
        if len(np.unique(self.dim_names)) != len(self.dim_names):
            raise AttributeError("Dimension names must be unique!\nnames: "+
                                 str(self.dim_names))

        # Ensure dimension names are at least 1 charater long
        if np.any([len(s)<1 for s in self.dim_names]):
            raise AttributeError("Dimension names must be at least 1 character"+
                                 " in lenght!\nnames: "+str(self.dim_names))

        # Ensure unique dimension names are valid identifiers
        if np.any([len(self._valid_dimname.findall(s)[0])!=len(s)
                   for s in self.dim_names]):
            raise AttributeError("Dimension names can only contain "+
                                 "alphanumeric characters and underscores, "+
                                 "and cannot begin with a number\nnames: "+
                                 str(self.dim_names))        

    def _select_ind(self, *args, **kwargs):
        """
        Returns a tuple of index arrays for the selected conditions
        and an array of Boolean index arrays.
        """
        # get starting indicies
        ind = [np.ones(dim.shape, dtype=np.bool) for dim in self.dims]

        # set to not remove any dimensions
        remove_dim = np.zeros(len(self.dims), dtype=np.bool)

        # process the args
        for arg in args:
            # arg must be a string
            if not isinstance(arg,str):
                raise TypeError('All args must be strings, ' + 
                                'but you passed: ' + str(type(arg)))

            # process the arg string
            filterStr = arg

            # figure out which dimension we're dealing with
            found_dim = False
            for d,k in enumerate(self.dim_names):
                # RE makes sure to not replace substrings
                if re.search(r'\b'+k+r'\b',filterStr) is not None:
                    # this is our dimension
                    found_dim = True

                    # replace the string to access the dimension like:
                    # self['dim1']
                    filterStr = re.sub(r'\b'+k+r'\b','np.asarray(self["'+k+'"])',filterStr)

                    # get the new index
                    newind = eval(filterStr)
                    
                    # apply it to the proper dimension index
                    ind[d] = ind[d] & newind

                    # see if we should remove the dim to emulate
                    # picking a specific index (e.g., x[10] same as
                    # x["time==4"])
                    # we normally require each value in a dim is unique,
                    # but a dim could be a list of events that you probe,
                    # which could have multiple values after an equality test,
                    # so we only remove a dimension if there is only one index left
                    if re.search('==',filterStr) and newind.sum()==1:
                        # we are using equality, so remove dim
                        remove_dim[d] = True

                    # break this loop to continue the next
                    break
                    
            # if we get to here, the provided string did not specify
            # any dimensions
            if not found_dim:
                # XXX eventually this should be a custom exception
                raise ValueError("The provided filter string did not specify "+
                                 "any valid dimensions: "+str(filterStr))
            
        # loop over the kwargs (the other way to filter)
        for key,value in kwargs.iteritems():
            if key in self.dim_names:
                # get the proper dimension to cull
                d = self.dim_names.index(key)
                # treat as boolean index
                ind[d] = ind[d] & value

        # create the final master index from the list of filtered indices
        m_ind = np.ix_(*ind)

        return m_ind,ind,remove_dim

    def __setitem__(self, index, obj):
        # process whether we using fancy string-based indices
        if isinstance(index,str):
            # see if it's just a single dimension name
            res = self._dim_namesRE.search(index)
            if res:
                # we have a single name, so set the
                # corresponding dimension
                self.dims[self.dim_names.index(res.group())] = obj
                return
            else:
                # call find to get the new index from the string
                index = self.find(index)
        elif isinstance(index,tuple) and \
                 np.any([isinstance(ind,str) for ind in index]):
            # Use find to get the new index from the list of stings
            index = self.find(*index)

        # perform the set
        AttrArray.__setitem__(self, index, obj)            

    def __getitem__(self, index):
        # process whether we using fancy string-based indices
        remove_dim = np.zeros(len(self.dims), dtype=np.bool)
        
        if isinstance(index,str):
            # see if it's just a single dimension name
            res = self._dim_namesRE.search(index)
            if res:
                # we have a single name, so return the
                # corresponding dimension as a DimSelect
                #return self.dims[self.dim_names.index(res.group())]
                ind = self.dim_names.index(res.group())
                return DimSelect(self.dims[ind],
                                 [d.shape for d in self.dims],
                                 ind)
            else:
                # call _select_ind to get the new index from the string
                index,o_ind,remove_dim = self._select_ind(index)
        elif isinstance(index,tuple) and \
                 np.any([isinstance(ind,str) for ind in index]):
            # Use _select_ind to get the new index from the list of stings
            index,o_ind,remove_dim = self._select_ind(*index)

        # try block to ensure the _skip_dim_check flag gets reset
        # in the following finally block
        try: 
            # skip the dim check b/c we're going to fiddle with them
            self._skip_dim_check = True
            ret = AttrArray.__getitem__(self,index)
        finally:
            # reset the _skip_dim_check flag:
            self._skip_dim_check = False

        # process the dims if necessary
        if isinstance(ret,DimArray):
            # see which to keep and modify the rest
            tokeep = np.arange(len(self.dims))
            # turn into a tuple for easier processing
            if not isinstance(index,tuple):
                indlist = (index,)
            else:
                indlist = index

            # see if we're gonna ignore removing dimensions
            if np.any([isinstance(ind,np.ndarray) and \
                           len(ind)==0 for ind in indlist]):
                # don't remove any b/c we selected nothing anyway
                remove_dim[:] = False
                
            # loop over the indlist, slicing the dimensions
            i = -1
            for ind in indlist:
                # increment the current dim
                i += 1
                if isinstance(ind,int) or isinstance(ind,long):
                    # if a changed dimension was reduced to one
                    # level, remove that dimension
                    tokeep = tokeep[tokeep!=i]
                elif ind is Ellipsis:
                    # do nothing
                    # PBS: Must add tests for [1,...], [1,...,2], [...,4]
                    # adjust current dim if necesary
                    if i < self.ndim-1:
                        # adjust to account for ellipsis
                        i = self.ndim - len(indlist) + i
                    continue
                elif ind is None:
                    # XXX Does not ensure new dim name does not exist
                    # It's a new axis, so make temp dim
                    newdim = Dim([0],'newaxis_%d'%(i))
                    newdims = ret.dims.tolist()
                    newdims.insert(i+1,newdim)
                    ret.dims = np.empty(len(newdims),dtype=Dim)
                    ret.dims[:] = newdims

                    # must also update the tokeep list
                    # must shift them up
                    tokeep[tokeep>=i]+=1
                    tokeep = tokeep.tolist()
                    tokeep.insert(i+1,i)
                    tokeep = np.array(tokeep)
                elif not isinstance(ind, slice) and len(ind)==0:
                    # handle where we selected nothing
                    ret.dims[i] = ret.dims[i][[]]
                else:
                    # slice the dims based on the index
                    #if isinstance(ind,np.ndarray) and ind.dtype==bool:
                    #    if len(ind.shape)>1:
                    #        ret = ret.view(AttrArray)
                    if not isinstance(ind, slice):
                        # squeeze it to maintain dimensionality
                        ind = np.asanyarray(ind)
                        tosqueeze = [0]*len(ind.shape)
                        tosqueeze[i] = slice(None)
                        ind = ind[tuple(tosqueeze)]
                        # if a boolean array is given as an index the
                        # dimensions get lost, so we need to cast to
                        # an AttrArray if there's more than 1 dim:
                        if isinstance(ind,np.ndarray) and ind.dtype==bool:
                            if len(self.shape)>1:
                                ret = ret.view(AttrArray)
                    ret.dims[i] = ret.dims[i][ind]

            # remove the empty dims
            ret.dims = ret.dims[tokeep]

            # remove the specified dims from the main array
            if np.any(remove_dim):
                ind = np.asarray([slice(None)]*len(ret.shape))
                ind[remove_dim] = 0
                ind = tuple(ind)
                ret = ret[ind]

        return ret
    
    def __getslice__(self,i,j):
        try: 
            # skip the dim check b/c we're going to fiddle with them
            self._skip_dim_check = True
            ret = AttrArray.__getslice__(self,i,j)           
        finally:
            # reset the _skip_dim_check flag:
            self._skip_dim_check = False
        ret.dims[0] = ret.dims[0].__getslice__(i,j)
        return ret
    
    def find(self,*args,**kwargs):
        """
        Returns a tuple of index arrays for the selected conditions. 

        There are three different ways to specify a filterstring
        illustrated in the examples.

        Notes
        -----
        data[data.find(filterstring)] returns the same slice as
        data.select(filterstring).

        See also
        --------
        DimArray.select

        Examples
        --------
        >>> import numpy as np
        >>> import dimarray as da
        >>> dim1 = da.Dim(range(5),'Dim1')
        >>> dim2 = da.Dim(['a','b','c'],'Dim2')
        >>> data = da.DimArray(np.random.rand(5,3),[dim1,dim2])
        >>> data.find('Dim1>2',"Dim2=='b'")
        (array([[3],
               [4]]), array([[1]]))
        >>> data.find(Dim1=data2['Dim1']>2,Dim2=data2['Dim2']=='b')
        (array([[3],
               [4]]), array([[1]]))
        >>> data.find("Dim1>kwargs['a']","Dim2==kwargs['b']",a=2,b='b')
        (array([[3],
               [4]]), array([[1]]))
        >>> data[data.find('Dim1>2',"Dim2=='b'")] == \
            data.select('Dim1>2',"Dim2=='b'")
        DimArray([[ True],
               [ True]], dtype=bool)
        >>> data[data.find(Dim1=data2['Dim1']>2,Dim2=data2['Dim2']=='b')] == \
            data.select(Dim1=data2['Dim1']>2,Dim2=data2['Dim2']=='b')
        DimArray([[ True],
               [ True]], dtype=bool)
        >>> data[data.find("Dim1>kwargs['a']","Dim2==kwargs['b']",a=2,b='b')] \
            == data.select("Dim1>kwargs['a']","Dim2==kwargs['b']",a=2,b='b')
        DimArray([[ True],
               [ True]], dtype=bool)
        """
        m_ind,ind,remove_dim = self._select_ind(*args,**kwargs)
        return m_ind

    def select(self,*args,**kwargs):
        """
        Returns a slice of the data filtered with the select conditions.

        There are three different ways to specify a filterstring
        illustrated in the examples.

        Notes
        -----
        data.select(filterstring) returns the same slice as
        data[data.find(filterstring)].

        See also
        --------
        DimArray.find

        Examples
        --------
        >>> import numpy as np
        >>> import dimarray as da
        >>> dim1 = da.Dim(range(5),'Dim1')
        >>> dim2 = da.Dim(['a','b','c'],'Dim2')
        >>> data = da.DimArray(np.random.rand(5,3),[dim1,dim2])
        >>> data
        DimArray([[ 0.52303181,  0.27313638,  0.28760072],
               [ 0.24885995,  0.40998977,  0.61080984],
               [ 0.43630142,  0.06662251,  0.61589201],
               [ 0.19332778,  0.27255998,  0.67924734],
               [ 0.57262178,  0.60912633,  0.80938473]])
        >>> data.select('Dim1>2',"Dim2=='b'")
        DimArray([[ 0.27255998],
               [ 0.60912633]])
        >>> data.select(Dim1=data2['Dim1']>2,Dim2=data2['Dim2']=='b')
        DimArray([[ 0.27255998],
               [ 0.60912633]])
        >>> data.select("Dim1>kwargs['a']","Dim2==kwargs['b']",a=2,b='b')
        DimArray([[ 0.27255998],
               [ 0.60912633]])
        >>> data.select('Dim1>2',"Dim2=='b'") == \
            data[data.find('Dim1>2',"Dim2=='b'")]
        DimArray([[ True],
               [ True]], dtype=bool)
        >>> data.select(Dim1=data2['Dim1']>2,Dim2=data2['Dim2']=='b') == \
            data[data.find(Dim1=data2['Dim1']>2,Dim2=data2['Dim2']=='b')]
        DimArray([[ True],
               [ True]], dtype=bool)
        >>> data.select("Dim1>kwargs['a']","Dim2==kwargs['b']",a=2,b='b') == \
            data[data.find("Dim1>kwargs['a']","Dim2==kwargs['b']",a=2,b='b')]
        DimArray([[ True],
               [ True]], dtype=bool)
        """
        m_ind,ind,remove_dim = self._select_ind(*args,**kwargs)
        return self[m_ind]

    def _split_bins(self, dim, bins, function, bin_labels,
                    error_on_nonexact, **kwargs):
        """
        Internal method for making bins when the number of bins or the
        indices at which to split the data into bins is
        specified. See make_bins method.
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
            raise ValueError("Invalid value for bin_labels. Allowed values " +
                             "are 'function','sequential', or a 1-D " +
                             " list/array of labels of the same length as " +
                             "bins.\n bins: " + str(bins) + "\n bin_labels: " +
                             str(bin_labels))

        # Create the new data:
        split_dat = split(self.view(AttrArray),bins,axis=dim)
        new_dat = np.array([function(x,axis=dim,**kwargs) for x in split_dat])
        
        # Now the dimensions of the array need be re-arranged in the correct
        # order:
        dim_order = np.arange(len(new_dat.shape))
        dim_order[dim] = 0
        dim_order[0:dim] = np.arange(1,dim+1)
        dim_order[dim+1:len(new_dat.shape)] = np.arange(dim+1,
                                                        len(new_dat.shape))
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
        a list of intervals. See make_bins method.
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
              (len(np.atleast_1d(bin_labels)) == bins)):
            new_dim_dat = np.altleast_1d(bin_labels)
        else:
            raise ValueError("Invalid value for bin_labels. Allowed values " +
                             "are 'function','sequential', or a 1-D " +
                             " list/array of labels of the same length as " +
                             "bins.\n bins: " + str(bins) + "\n bin_labels: " +
                             str(bin_labels))
        
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
                        
        Parameters
        ----------
        axis : int
            The dimension to be binned. Can be name or number.
        bins : {int,list,tuple}
            Specifies how the data should be binned. Acceptable values
            are:
            * the number of bins (equally spaced, if possible, roughly
              equally spaced if not and error_on_nonexact is False).
              (Uses numpy.[array]split.)
            * A 1-D container (list or tuple) of the indices where the
              data should be split into bins. The value for
              error_on_nonexact does not influence the result.  (Uses
              numpy.[array]split.)
            * A 2-D container (lists or tuples) where each container
              in the first dimension specifies the min (inclusive) and
              the max (exlusive) values and (optionally) a label for
              each bin. The value for error_on_nonexact must be
              True. If labels are specified in bins, they are used and
              the value of bin_labels is ignored.
        function : function
            The function to aggregate over within the bins. Needs to
            take the data as the first argument and an additional axis
            argument (numpy.mean is an example of a valid function).
        bins_labels : {'function','sequential',array_like}, optional
            'function' applies the function that is used for binning
            to the dimension, 'sequential' numbers the bins
            sequentially. Alternatively, a 1-D container that contains
            the bin labels can be specified.
        error_on_nonexact : {True, False}, optional
            Specifies whether roughly equal bin sizes are acceptable
            when the data cannot be evenly split in the specified
            number of bins (this parameter is only applicable when
            bins is an integer specifying the number of bins). When
            True, the function numpy.split is used, when False the
            function numpy.array_split is used.                       
        kwargs : keyword arguments, optional
            Optional key word arguments to be passed on to function.
        
        Returns
        -------
        binned : DimArray
            A new DimArray instance in which one of the dimensions is
            binned as specified.
        
        Examples
        --------
        >>> import numpy as np
        >>> import dimarray as da
        >>> data = da.DimArray(np.random.rand(4,5))
        >>> data
        DimArray([[ 0.74214411,  0.35124939,  0.52641061,  0.85086401,  0.38799751],
               [ 0.692385  ,  0.14314031,  0.61169269,  0.14904847,  0.65182813],
               [ 0.33258044,  0.07763733,  0.18474865,  0.67977018,  0.30520807],
               [ 0.05501445,  0.09936871,  0.55943639,  0.70683311,  0.10069493]])
        >>> data.make_bins('dim1',2,np.mean)
        DimArray([[ 0.71726456,  0.24719485,  0.56905165,  0.49995624,  0.51991282],
               [ 0.19379744,  0.08850302,  0.37209252,  0.69330165,  0.2029515 ]])
        >>> data.make_bins('dim1',2,np.mean).dims
        array([[ 0.5  2.5], [0 1 2 3 4]], dtype=object)
        >>> data.make_bins('dim1',2,np.mean)
        DimArray([[ 0.71726456,  0.24719485,  0.56905165,  0.49995624,  0.51991282],
               [ 0.19379744,  0.08850302,  0.37209252,  0.69330165,  0.2029515 ]])
        >>> data.make_bins('dim2',2,np.mean,error_on_nonexact=False).dims
        array([[0 1 2 3], [ 1.   3.5]], dtype=object)
        """
        # Makes sure dim is index (convert dim name if necessary):
        dim = self.get_axis(axis)
        tmp_bins = np.atleast_2d(bins)
        if len(tmp_bins.shape)>2:
            raise ValueError("Invalid bins! Acceptable values are: number of" +
                             " bins, 1-D container of index values, 2-D " +
                             "container of min and max values and (optionally)" +
                             " a label for each bin. Provided bins: "+str(bins))
        if np.atleast_2d(bins).shape[1] == 1:
            return self._split_bins(dim,bins,function,bin_labels,
                               error_on_nonexact,**kwargs)
        elif np.atleast_2d(bins).shape[1] == 2:
            if not error_on_nonexact:
                raise ValueError("When bins are explicitly specified, " +
                                 "error_on_nonexact must be True. Provided " +
                                 "value: " + str(error_on_nonexact))
            return self._select_bins(dim,bins,function,bin_labels,
                                error_on_nonexact,**kwargs)
        elif np.atleast_2d(bins).shape[1] == 3:
            if bin_labels != 'function':
                raise ValueError("Simultaneously specification of bin labels " +
                                 "in bins and bin_labels is not allowed. " +
                                 "Provided bins: " + str(bins)+" Provided " +
                                 "bin_labels: " + str(bin_labels))
            if not error_on_nonexact:
                raise ValueError("When bins are explicitly specified, " +
                                  "error_on_nonexact must be True. Provided " +
                                  "value: " + str(error_on_nonexact))
            return self._select_bins(dim,bins,function,bin_labels,
                                     error_on_nonexact,**kwargs)
        else:
            raise ValueError("Invalid bins! Acceptable values are: number of" +
                             " bins, 1-D container of index values, 2-D " +
                             "container of min and max values and (optionally)" +
                             " a label for each bin. Provided bins: "+str(bins))

    def extend(self, data, axis=0):
        """
        Extend a DimArray along a specified axis.

        Parameters
        ----------
        data : sequence of {DimArray} objects or single {DimArray}
            The DimArrays must have the same dimensions as the current
            DimArray, except for the dimension corresponding to `axis`
            (the first, by default).
        axis : {int,str},optional
            The axis along which the DimArray objects will be joined.

        Returns
        -------
        result : DimArray
            A new DimArray instance extended as specified and of the
            same class as the current object.

        Notes
        -----
        Only the attributes of the current object are preserved (with
        the exception of the updated dimension).
        """
        # make sure we have a list
        if isinstance(data,DimArray):
            data = [data]
        else:
            data = list(data)

        # make sure we have an axis number:
        axis = self.get_axis(axis)

        # make sure all dim_names match:
        dim_names_deviations = [np.sum(d.dim_names!=self.dim_names) for d in data]
        if np.sum(dim_names_deviations)>0:
            raise ValueError('Names of the dimensions do not match!')
                
        # make sure all dims except for the extended one match:
        dim_deviations = [np.sum(d.dims!=self.dims) for d in data]
        if np.sum(dim_deviations)>1:
            raise ValueError('Dimensions do not match!')
                
        # add the current DimArray to the beginning of list:
        data.insert(0,self)

        # list of dims to be concatenated:
        conc_dims = [d.dims[axis] for d in data]

        # convert all items to AttrArray view (necessary for call to
        # np.concatenate which transposes the arrays using the numpy
        # functions rather than the DimArray functions):
        data = [d.view(AttrArray) for d in data]
        #dim_names = [d.name for dat in data for d in dat.dims]
        
        new_dat = np.concatenate(data,axis=axis)
        new_dims = copylib.deepcopy(self.dims)
        new_dims[axis] = Dim(np.concatenate(conc_dims),self.dim_names[axis])
        new_attrs = self._attrs.copy()
        new_attrs['dims'] = new_dims
        return self.__class__(new_dat,**new_attrs)

        # # create new array:
        # result = np.concatenate(data,axis=axis).view(AttrArray)
        # result._attrs = self._attrs
        # result.__array_finalize__(self)
        # # update dims & return:
        # result.dims[axis] = Dim(np.concatenate(conc_dims),self.dim_names[axis])
        # return result.view(self.__class__)        

    def add_dim(self, dim):
        """
        Add a new Dim to a DimArray, repeating the existing data for
        each value of the new dim.
        """
        # add the axis and repeat the data
        ret = self[np.newaxis].repeat(len(dim),axis=0)
        # now replace the dim
        ret.dims[0] = dim
        # return as desired class
        return ret.view(self.__class__)

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
        ----------
        axis : str
            The name of a dimension.
            
        Returns
        -------
        The axis number corresponding to the dimension name.
        If axis is not a string, it is returned unchanged.        
        """
        if isinstance(axis,str):
            # must convert to index dim
            axis = self.dim_names.index(axis)
        return axis

    def get_dim_name(self, axis):
        """
        Get the dim name for an axis.

        Provides an convenient way to ensure a dim name from an axis
        specification.  If a string is passed in, it is assumed to be
        a dim name is returned.

        Parameters
        ----------
        axis : int
            The index of a dimension.

        Returns
        -------
        The dim name for the specified axis.
        """
        if isinstance(axis, str):
            dim_name = axis
        else:
            dim_name = self.dim_names[axis]
        return dim_name
    
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
            ret = self.view(AttrArray).argsort(axis=axis, kind=kind,
                                               order=order)
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

    def nanmean(self, axis=None):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).nanmean(axis=axis)
        return self._ret_func(ret,axis)

    def nanstd(self, axis=None, ddof=0):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).nanstd(axis=axis, ddof=0)
        return self._ret_func(ret,axis)

    def nanvar(self, axis=None, ddof=0):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).nanvar(axis=axis, ddof=0)
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
        axis = self.get_axis(axis)
        return self.view(AttrArray).repeat(repeats, axis=axis)

    def reshape(self, shape, order='C'):
        return np.reshape(self.view(AttrArray),shape,order)

    def resize(self, *args, **kwargs):
        """Resizing is not possible for dimensioned arrays. Calling
        this method will throw a NotImplementedError exception. If
        resizing is desired the array needs to be converted to a
        different data type (e.g., numpy.ndarray), first!"""
        raise NotImplementedError("Resizing is not possible for dimensioned " +
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
            return self.view(AttrArray).take(indices, axis=axis,
                                             out=out, mode=mode)
        else:
            axis = self.get_axis(axis)
            ret = self.view(AttrArray).take(indices, axis=axis,
                                            out=out, mode=mode)
            ret.dims[axis] = ret.dims[axis].take(indices, axis=0,
                                                 out=out, mode=mode)
            return ret.view(self.__class__)
        
    def trace(self, *args, **kwargs):
        return self.view(AttrArray).trace(*args, **kwargs)

    def transpose(self, *axes):
        axes = np.squeeze(axes)
        # PBS (I think this was wrong):if len(axes.shape)==len(self):
        if(len(np.shape(axes))>0): # needs to be evaluated separately
                                   # b/c len(axes) won't work on None
            if(len(axes) == self.ndim):
                axes = [self.get_axis(a) for a in axes]
                ret = self.view(AttrArray).transpose(*axes)
                # ret.dims = [ret.dims[a] for a in axes]
                ret.dims = ret.dims[axes]
                return ret.view(self.__class__)     
        ret = self.view(AttrArray).transpose()
        # ret.dims.reverse()
        ret.dims = ret.dims[-1::-1]
        return ret.view(self.__class__)

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        axis = self.get_axis(axis)
        ret = self.view(AttrArray).var(axis=axis, dtype=dtype,
                                       out=out, ddof=ddof)
        return self._ret_func(ret,axis)

# set the doc strings

# Methods that return DimArrays and take an axis argument:
axis_msg =\
"""

 **Below is the docstring from numpy.ndarray.**
 **For DimArray instances, the axis may be specified as string (dimension name).**

"""
axis_msg_aa =\
"""

 **Below is the docstring from AttrArray.**
 **For DimArray instances, the axis may be specified as string (dimension name).**

"""
axes_msg =\
"""

 **Below is the docstring from numpy.ndarray.**
 **The axes may be specified as strings (dimension names).**

"""
DimArray.all.im_func.func_doc = axis_msg+np.ndarray.all.__doc__            
DimArray.any.im_func.func_doc = axis_msg+np.ndarray.any.__doc__            
DimArray.argmax.im_func.func_doc = axis_msg+np.ndarray.argmax.__doc__            
DimArray.argmin.im_func.func_doc = axis_msg+np.ndarray.argmin.__doc__            
DimArray.argsort.im_func.func_doc = axis_msg+np.ndarray.argsort.__doc__            
DimArray.compress.im_func.func_doc = axis_msg+np.ndarray.compress.__doc__            
DimArray.cumprod.im_func.func_doc = axis_msg+np.ndarray.cumprod.__doc__            
DimArray.cumsum.im_func.func_doc = axis_msg+np.ndarray.cumsum.__doc__            
DimArray.max.im_func.func_doc = axis_msg+np.ndarray.max.__doc__            
DimArray.mean.im_func.func_doc = axis_msg+np.ndarray.mean.__doc__            
DimArray.min.im_func.func_doc = axis_msg+np.ndarray.min.__doc__            
DimArray.nanmean.im_func.func_doc = axis_msg_aa+AttrArray.nanmean.__doc__            
DimArray.nanstd.im_func.func_doc = axis_msg_aa+AttrArray.nanstd.__doc__            
DimArray.nanvar.im_func.func_doc = axis_msg_aa+AttrArray.nanvar.__doc__            
DimArray.prod.im_func.func_doc = axis_msg+np.ndarray.prod.__doc__            
DimArray.ptp.im_func.func_doc = axis_msg+np.ndarray.ptp.__doc__            
DimArray.sort.im_func.func_doc = axis_msg+np.ndarray.sort.__doc__            
DimArray.std.im_func.func_doc = axis_msg+np.ndarray.std.__doc__            
DimArray.sum.im_func.func_doc = axis_msg+np.ndarray.sum.__doc__            
DimArray.swapaxes.im_func.func_doc = axes_msg+np.ndarray.swapaxes.__doc__            
DimArray.take.im_func.func_doc = axis_msg+np.ndarray.take.__doc__            
DimArray.transpose.im_func.func_doc = axes_msg+np.ndarray.transpose.__doc__            
DimArray.var.im_func.func_doc = axis_msg+np.ndarray.var.__doc__            

# Methods that return DimArrays and do not take an axis argument:
DimArray.nonzero.im_func.func_doc = np.ndarray.nonzero.__doc__            
DimArray.squeeze.im_func.func_doc = np.ndarray.squeeze.__doc__            


# Methods that return AttrArrays: Prefic docstring with warning!
cast_msg =\
"""

 **CAUTION: the output of this method is cast to an AttrArray instance.**
 **Some attributes may no longer be valid after this Method is applied!**

"""
DimArray.diagonal.im_func.func_doc = cast_msg+np.ndarray.diagonal.__doc__
DimArray.flatten.im_func.func_doc = cast_msg+np.ndarray.flatten.__doc__
DimArray.ravel.im_func.func_doc = cast_msg+np.ndarray.ravel.__doc__            
DimArray.repeat.im_func.func_doc = cast_msg+np.ndarray.repeat.__doc__            
DimArray.reshape.im_func.func_doc = cast_msg+np.ndarray.reshape.__doc__  
#DimArray.resize.im_func.func_doc = cast_msg+np.ndarray.resize.__doc__            
DimArray.trace.im_func.func_doc = cast_msg+np.ndarray.trace.__doc__            
