import re
import numpy as N


##############################
# New dimensioned data classes
##############################

class Dim(object):
    """
    Holds the information describing a dimension of data.
    """
    def __init__(self,name,data,unit=None):
        """
        """
        self.name = name
        self.data = N.atleast_1d(data)
        self.unit = unit

    def copy(self):
        return Dim(self.name,self.data.copy(),self.unit)

    def extend(self,other):
        if type(self) != type(other):
            raise "Can only concatenate the same type of data."
        if self.unit != other.unit:
            raise "Can only concatenate data with the same units."

        self.data = N.concatenate((self.data,other.data),axis=0)
        #return Dim(self.name,N.concatenate((self.data,other.data),axis=0),self.units)
    
    def __str__(self):
        outstr = '%s: %s .. %s %s' % (self.name,
                                      self.data[0],
                                      self.data[-1],
                                      self.unit)
        return outstr

    def __repr__(self):
        outstr = 'Dim(%s,\n\t%s,\n\tunits=%s)' % (self.name.__repr__(),
                                                  self.data.__repr__(),
                                                  self.unit.__repr__())
        return outstr

    def __getitem__(self, item):
        """
        :Parameters:
            item : ``slice``
                The slice of the data to take.
        
        :Returns: ``numpy.ndarray``
        """
        return self.data[item]

    def select(self, item):
        """
        Return a new Dim instance of the specified slice.
        """
        return Dim(self.name,self.data.copy()[item],self.unit)
        
    def __setitem__(self, item, value):
        """
        :Parameters:
            item : ``slice``
                The slice of the data to write to
            value : A single value or array of type ``self.dtype``
                The value to be set.
        
        :Returns: ``None``
        """
        self.data[item] = value

    def __lt__(self, other):
        return self.data < other
    def __le__(self, other):
        return self.data <= other
    def __eq__(self, other):
        return self.data == other
    def __ne__(self, other):
        return self.data != other
    def __gt__(self, other):
        return self.data > other
    def __ge__(self, other):
        return self.data >= other


class Dims(object):
    """
    Holds an ordered set of dimensions.
    """
    def __init__(self,dims):
        """
        """
        # set the dims
        self.dims = dims

        # setup the regexp
        self._set_regexp()

    def _set_regexp(self):
        # save the names list and a regexp for it
        self.names = [dim.name for dim in self.dims]
        regexpNames = '\\b'+'\\b|\\b'.join(self.names)+'\\b'
        self._namesRE = re.compile(regexpNames)

        regexpNameOnly = '(?<!.)\\b' + '\\b(?!.)|(?<!.)\\b'.join(self.names) + '\\b(?!.)'
        self._nameOnlyRE = re.compile(regexpNameOnly)

    def index(self,name):
        return self.names.index(name)

    def insert(self,index,dim):
        """Insert a new Dim before index."""
        self.dims.insert(index,dim)
        self._set_regexp()

    def copy(self):
        return Dims([dim.copy() for dim in self.dims])

    def __getitem__(self, item):
        """
        :Parameters:
            item : ``index``
                The index into the list.
        
        :Returns: ``Dim``
        """
        # see if we're looking up by index
        if isinstance(item,str):
            item = self.index(item)

        # return the index into the list
        return self.dims[item]

    def __setitem__(self, item, value):
        """
        :Parameters:
            item : ``slice``
                The slice of the data to write to
            value : A single value or array of type ``self.dtype``
                The value to be set.
        
        :Returns: ``None``
        """
        self.dims[item] = value

    def __iter__(self):
        return self.dims.__iter__()

    def __str__(self):
        # loop over dimensions
        outstr = ''
        for dim in self.dims:
            if len(outstr) > 0:
                outstr += '\n'
            outstr += str(dim)
        return outstr
    def __repr__(self):
        outstr = 'Dims('
        outstr += self.dims.__repr__()
        outstr += ')'
        return outstr

    def select(self,*args,**kwargs):
        """Return a new Dims instance of only the Dims you want.  Not
        currently implemented, but let us know if you ever need it (we
        couldn't think of a reason.)
        """
        raise NotImplementedError


class DimData(object):
    """
    Dimensioned data class.

    data['time<0']
    
    """
    def __init__(self,data,dims,unit=None):
        """
        Data with defined dimensions.
        """
        # set the data and dims
        self.data = data
        if isinstance(dims,Dims):
            self.dims = dims
        else:
            # turn the list into a Dims class
            self.dims = Dims(dims)

        # set the unit
        self.unit = unit

        self._reset_data_stats()

    def _reset_data_stats(self):
        """
        """
        # make sure the lengths of the dims match the shape of the data
        if self.data.shape != tuple([len(d.data) for d in self.dims]):
            # raise error
            raise ValueError("The length of dims must match the data shape.\nData shape: "+
                             str(self.data.shape)+"\nShape of the dims: "+
                             str(tuple([len(d.data) for d in self.dims])))
        # describe the data
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.ndim = len(self.shape)

    def copy(self):
        """
        Return a copy of this DimData instance.
        """
        newdata = self.data.copy()
        newdims = self.dims.copy()
        return DimData(newdata,newdims,self.unit)
        
    def dim(self,name):
        """
        Return the numerical index (axis) of the named dimension.
        """
        return self.dims.index(name)

    def __getitem__(self,item):
        """
        :Parameters:
            item : ``slice``
                The slice of the data to take.
        
        :Returns: ``numpy.ndarray``
        """
        if isinstance(item,str):
            # see if it's just a single dimension name
            res = self.dims._nameOnlyRE.search(item)
            if res:
                # we have a single name, so return the data from that dimension
                return self.dims[res.group()].data
            else:
                # call select_ind and return the slice into the data
                m_ind,ind = self._select_ind(item)
                # set up the new data
                return self.data[m_ind]
        if isinstance(item,tuple):
            # call select
            # call select_ind and return the slice into the data
            m_ind,ind = self._select_ind(*item)
            # set up the new data
            return self.data[m_ind]
        else:
            # return the slice into the data
            return self.data[item]

    def __setitem__(self, item, value):
        """
        :Parameters:
            item : ``slice``
                The slice of the data to write to
            value : A single value or array of type ``self.dtype``
                The value to be set.
        
        :Returns: ``None``
        """
        if isinstance(item,str):
            # see if it's just a single dimension name
            res = self.dims._nameOnlyRE.search(item)
            if res:
                # we have a single name, so return the data from it
                self.dims[res.group()].data = value
            else:
                # call select_ind and return the slice into the data
                m_ind,ind = self._select_ind(item)
                # set up the new data
                self.data[m_ind] = value
        if isinstance(item,tuple):
            # call select
            # call select_ind and return the slice into the data
            m_ind,ind = self._select_ind(*item)
            # set up the new data
            self.data[m_ind] = value
        else:
            # return the slice into the data
            self.data[item] = value

    def _select_ind(self,*args,**kwargs):
        """
        Returns a tuple of index arrays for the selected conditions and an array
        of Boolean index arrays.     
        """
        # get starting indicies
        ind = [N.ones(dim.data.shape,N.bool) for dim in self.dims]

        # process the args
        for arg in args:
            # arg must be a string
            filterStr = arg

            # figure out which dimension we're dealing with
            for d,k in enumerate(self.dims.names):
                # RE makes sure to not replace substrings
                if re.search(r'\b'+k+r'\b',filterStr):
                    # this is our dimension
                    # replace the string
                    filterStr = re.sub(r'\b'+k+r'\b','self.dims["'+k+'"]',filterStr)

                    # get the new index
                    newind = eval(filterStr)
                    
                    # apply it to the dimension index
                    ind[d] = ind[d] & newind

                    # break this loop to continue the next
                    break

        # loop over the kwargs
        for key,value in kwargs.iteritems():
            if key in self.dims.names:
                # get the proper dimension to cull
                d = self.dims.index(key)
                ind[d] = ind[d] & value

        # create the final master index
        m_ind = N.ix_(*ind)

        return m_ind,ind

    def find(self,*args,**kwargs):
        """
        Returns a tuple of index arrays for the selected conditions. 

        data.find('time>0','events.recalled==True')
        or
        data.find(time=data['time']>0,events=data['events'].recalled==True)
        or 
        data.find("time>kwargs['t']","events.recalled==kwargs['val']",t=0,val=True)
        the above can be written with single quoted strings as follows:
        data.find('time>kwargs[\'t\']','events.recalled==kwargs[\'val\']',t=0,val=True)

        data.data[ind], where ind is the return value of the find method and
        data.select(filterstring).data return arrays with the same shapes and values,
        provided that the same filterstring is used. However, the data array from the
        select method belongs to a copy of the DimData instance and thus does not
        support assignment of values for the orignal DimData instance -- e.g.,
        data.data[ind] = data.data[ind] * 2.0.
        For a Boolean index, see the data_ind method.
        """
        m_ind,ind = self._select_ind(*args,**kwargs)
        return m_ind

    def data_ind(self,*args,**kwargs):
        """
        Returns a tuple of Boolean index arrays for the selected conditions. 

        data.data_ind('time>0','events.recalled==True')
        or
        data.data_ind(time=data['time']>0,events=data['events'].recalled==True)
        or 
        data.find("time>kwargs['t']","events.recalled==kwargs['val']",t=0,val=True)
        the above can be written with single quoted strings as follows:
        data.find('time>kwargs[\'t\']','events.recalled==kwargs[\'val\']',t=0,val=True)

        data.data[ind], where ind is the return value of the data_ind method and
        data.select(filterstring).data return arrays with the same shapes and values,
        provided that the same filterstring is used. However, the data array from the
        select method belongs to a copy of the DimData instance and thus does not
        support assignment of values for the orignal DimData instance -- e.g.,
        data.data[ind] = data.data[ind] * 2.0.
        For the actual index values rather than the Boolean index, see the find method.
        """
        m_ind = self.find(*args,**kwargs)
        bool_ind = N.zeros(self.data.shape,N.bool)
        bool_ind[m_ind] = True
        return bool_ind
        
    def select(self,*args,**kwargs):
        """
        Return a copy of the data filtered with the select conditions.

        data.select('time>0','events.recalled==True')
        or
        data.select(time=data['time']>0,events=data['events'].recalled==True)
        or 
        data.find("time>kwargs['t']","events.recalled==kwargs['val']",t=0,val=True)
        the above can be written with single quoted strings as follows:
        data.find('time>kwargs[\'t\']','events.recalled==kwargs[\'val\']',t=0,val=True)

        To get a tuple of index arrays for the selected conditions use the find method.
        """
        m_ind,ind = self._select_ind(*args,**kwargs)
        # set up the new data
        newdat = self.data[m_ind]

        # index the dims
        newdims = [dim.select(ind[d]) for dim,d in zip(self.dims,range(len(ind)))]
        
        # make the new DimData
        # by doing this as a copy it makes sure that we return the
        # child class as opposed to the parent
        newDimData = self.copy()
        newDimData.data = newdat
        newDimData.dims = Dims(newdims)
        newDimData._reset_data_stats()
        return newDimData

    def extend(self,other,dim):
        """
        Concatenate two data instances by extending over a given dimension.
        """
        # set the new dat
        self.data = N.concatenate((self.data,other.data),axis=dim)

        # set the new dims
        self.dims[dim].extend(other.dims[dim])

        self._reset_data_stats()
        
        #newdims = self.dims.copy()
        #newdims[dim] = newdims[dim].extend(other.dims[dim])

#         # make the new DimData
#         newDimData = self.copy()
#         newDimData.data = newdat
#         newDimData.dims = Dims(newdims)
#         newDimData._reset_data_stats()
#         return newDimData

    def apply_func(self,function,unit=None,*args,**kwargs):
        """
        """
        newDimData = self.copy()
        newDimData.data = function(newDimData.data,*args,**kwargs)
        return newDimData

    def aggregate(self,dims,function,unit=None,dimval=True,**kwargs):
        """
        Return a copy of the data aggregated over the dimensions
        specified in the list dims with the passed in function. The
        function needs to take an array and a dimension over which to
        apply it as an input (a suitable candidate is, e.g.,
        numpy.mean). The aggregation is done sequentially over the
        passed in dimensions in the order in which they are passed
        in. The unit is set to None, unless otherwise specified,
        because the transformation may change the unit.
        
        If dimval is False the aggregation is done over every
        dimension EXCEPT those specified in dims. In this case the
        data are aggregated over the not specified dimensions in the
        order of the dimension index values.

        *** WARNING: The sequential aggregation may not be appropriate
        *** for all functions. E.g., when numpy.std is used to
        *** aggregate over multiple dimensions, the result is a std of
        *** (std's of ...)  std's. To aggregate the data over all but
        *** one dimension non-sequentially, see the margin method.

        If data is a DimData instance with dimensions A, B, C, and D:        

        1) data.aggregate('A',numpy.mean) returns a DimData instance
        with dimensions B, C, and D, that contain the mean across the
        A dimension.

        2) data.aggregate('B',numpy.mean,dimval=False) returns a
        DimData instance with dimension B taking the mean across
        dimensions A, C, and D.

        3) data.aggregate(['A','C'],numpy.mean) returns a DimData
        instance with dimensions B and D in which the mean was taken
        across dimensions A and C in that order. Note that for some
        functions (such as numpy.std) passing in ['A','C'] as dims may
        produce a result different from passing in ['C','A'] in which
        the aggregation takes place in the reverse order -- see
        warning above!

        4) data.aggregate(['B','D',numpy.mean,dimval=False]) produces
        the same result as data.aggregate(['A','C'],numpy.mean) (see
        above).
        """

        # Find the index for dimension numbers (instead of names):
        dimNumsInd = N.array(map(lambda x: isinstance(x,int), dims))
        if dimNumsInd.any() and not dimNumsInd.all():
            raise ValueError("dims must be a 1-D list of dimension indices\
            OR names. Mixing of names and indices is not allowed.\
            Invalid value for dims: %s " % str(dims))

        #If indices are given, convert to dim names:
        if len(dimNumsInd)>0 and dimNumsInd.all():
            dimnames = N.array(self.dims.names)[dims]
        else:
            dimnames = dims

        # If a string is passed in instead of a list or an array, convert:
        dimnames = N.atleast_1d(dimnames)
        if len(N.shape(dimnames)) != 1:
            raise ValueError("dims must be a 1-D list of dimensions.\
            Invalid value for dims: %s " % str(dimnames))

        # We need to work with a copy to ensure that we
        # return the child class as opposed to the parent:
        newDimData = self.copy()

        # Get the dimension names for all dimension and the selected dimensions
        # in a format suitable for generating a Boolean index of the dimensions
        # over which to aggregate:
        dn_all,dn_sel = N.ix_(newDimData.dims.names,dimnames)
        
        # Generate the Boolean aggregation dimension index contingent on dimval:
        if dimval:
            db = N.array(dn_all==dn_sel)
            db = N.atleast_1d(db.any(1))
        else:
            db = N.array(dn_all!=dn_sel)
            db = N.atleast_1d(db.all(1))

        # If there are no dimensions to aggregate over, return a copy of self:
        if not db.any():
            return newDimData
        
        # If dimns is empty, len(db) will be 1 regardless of how many
        # dimensions there are. In this case we need to repeat the single
        # db value to make db the right length:
        if len(db)==1:
            db = db.repeat(len(newDimData.dims.names))

        # Get the names of the dimensions over which to aggregate:
        aggDimNames = N.array(newDimData.dims.names)[db]

        # Apply function to each dimension of the data in turn: We need to do it
        # in reverse order (i.e., starting with the last dimension) to preserve
        # the dimension indices stored in newDimData.dim(dimname):
        newdat = newDimData.data
        for dimname in aggDimNames[::-1]:
            newdat = function(newdat,axis=newDimData.dim(dimname),**kwargs)

        # Update newDimData:
        newDimData.data = newdat
        # The new dims are those not aggregated over:
        newDims = Dims(list(N.array(newDimData.dims.copy().dims)[-db]))
        newDimData.dims = newDims
        newDimData.unit = unit
        # Clean up & return:
        newDimData._reset_data_stats()
        return newDimData



    def margin(self,dim,function,unit=None,**kwargs):
        """
        Return a copy of the data aggregated over all but the
        specified dimension with the passed in function. The function
        needs to take an array and a dimension over which to apply it
        as an input (a suitable candidate is, e.g., numpy.mean). The
        function is applied simultaneously over all other
        dimensions. The unit is set to None, unless otherwise
        specified, because the transformation may change the unit.

        If data is a DimData instance with dimensions A, B, C, and D,
        data.margin('A',numpy.mean) returns a DimData instance
        with dimension A that contains the mean across the B, C, and D
        dimensions.
        """
        # If a string is passed in instead of a list or an array, convert:
        dim = N.atleast_1d(dim)
        if (len(N.shape(dim)) != 1) or (len(dim)!=1):
            raise ValueError("dim must be a single dimension name or index.\
            Invalid value for dim: %s " % str(dim))

        dim = dim[0]

        # If dim is dim name, convert to dim index:
        if isinstance(dim,str):
            dim = self.dim(dim)

        # We need to transpose the data array so that dim is the first
        # dimension. We store the new order of dimensions in totrans:
        totrans = range(len(self.data.shape))
        totrans[0] = dim
        totrans[dim] = 0
        # After the transpose, we need to reshape the array to a 2D
        # array with dimname as the first dimension. We store the
        # new shape in toreshape:
        toreshape = [len(self.dims[dim].data),
                     N.cumprod(N.array(self.data.shape)[totrans[1:]])[-1]]
        # Now we are ready to do the transpose & reshape:
        tmpdata = N.reshape(N.transpose(self.data,totrans),toreshape)
        # Now that zstddata is a 2D array, we can just take the
        # function over the 2nd dimension:
        tmpdata = function(tmpdata,axis=1,**kwargs)
        
        # Now that we have the, we need to create a new DimData instance.
        # Create a new Dims instance:
        newDims = Dims([self.dims[dim].copy()])
        
        # Create a copy of self to make sure we have a child instance:
        newDimData = self.copy()
        newDimData.dims = newDims
        newDimData.data = tmpdata
        newDimData.unit = unit
        # Clean up & return:
        newDimData._reset_data_stats()
        return newDimData



    def get_bins(self,dim,bins,function,unit=None,number_bins=False,dim_unit=None,error_on_nonexact=True,**kwargs):
        """
        Return a copy of the data with dimension dim binned as specified.
        Example usage:
          data.get_bins('time',10,numpy.mean,unit=data.unit,number_bins=False,dim_unit='time bin midpoint')
        Input:
          dim
            The dimension to be binned. Can be name or number.
          bins
            The number of bins (equally spaced, if possible, roughly equally
            spaced if not and error_on_nonexact is False). Alternatively the
            indices where the data should be split into bins can be specified.
            See numpy.split and numpy.array_split for details.
          function
            The function to aggregate over within the bins. Needs to take the
            data as the first argument and an additional axis argument
            (numpy.mean is an example of a valid function).
          unit
            The new unit of the data (the passed in function may change the
            unit so it needs to be explicity specified)
          number_bins
            If True the binned dimension is labeled by bin number. If False
            the new labels are found by binning the old labels and applying
            the function on the old labels. This only works if the labels are
            an acceptable input to function (e.g., numpy.float for numpy.mean).
          dim_unit
            The unit of the binned dimension.
          error_on_nonexact
            Specifies whether roughly equal bin sizes are acceptable when the
            data cannot be evenly split in the specified number of bins.
            Internally, when True, the function numpy.split is used, when False
            the function numpy.array_split is used.
          *kwargs
            Optional key word arguments to be passed on to function.
        Output:
          A new DimData instance in which one of the dimensions is binned as specified.
        """
        # If a string is passed in instead of a list or an array, convert:
        dim = N.atleast_1d(dim)
        if (len(N.shape(dim)) != 1) or (len(dim)!=1):
            raise ValueError("dim must be a single dimension name or index.\
            Invalid value for dim: %s " % str(dim))

        dim = dim[0]

        # If dim is dim name, convert to dim index:
        if isinstance(dim,str):
            dim = self.dim(dim)

        # Determine which function to use for splitting:
        if error_on_nonexact:
            split = N.split
        else:
            split = N.array_split

        # Create the new dimension:
        split_dim = N.array(split(self.dims[dim].data,bins))
        if number_bins:
            new_dim = Dim(self.dims.names[dim],
                          N.arange(len(split_dim)),
                          unit=dim_unit)
        else:
            new_dim = Dim(self.dims.names[dim],
                          function(split_dim,axis=1,**kwargs),
                          unit=dim_unit)
        
        # Create the new data:
        split_dat = N.array(split(self.data,bins,axis=dim))
        new_dat = function(split_dat,axis=dim+1,**kwargs)
        
        # Now the dimensions of the array need be re-arranged in the correct
        # order:
        dim_order = N.arange(len(new_dat.shape))
        dim_order[dim] = 0
        dim_order[0:dim] = N.arange(1,dim+1)
        dim_order[dim+1:len(new_dat.shape)] = N.arange(dim+1,len(new_dat.shape))
        new_dat = N.transpose(new_dat,dim_order)
        
        # Create and return new DimData object:
        new_dims = self.dims.copy()
        new_dims[dim] = new_dim
        newDimData = self.copy()
        newDimData.dims = new_dims
        newDimData.data = new_dat
        newDimData.unit = unit
        newDimData._reset_data_stats()
        return newDimData

