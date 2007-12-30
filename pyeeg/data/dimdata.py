
import re
import numpy as N


##############################
# New dimensioned data classes
##############################

class Dim(object):
    """
    Holds the information describing a dimension of data.
    """
    def __init__(self,name,data,units=None):
        """
        """
        self.name = name
        self.data = N.asarray(data)
        self.units = units

    def copy(self):
        return Dim(self.name,self.data.copy(),self.units)

    def extend(self,other):
        if type(self) != type(other):
            raise "Can only concatenate the same type of data."
        if self.units != other.units:
            raise "Can only concatenate data with the same units."

        self.data = N.concatenate((self.data,other.data),axis=0)
        #return Dim(self.name,N.concatenate((self.data,other.data),axis=0),self.units)
    
    def __str__(self):
        outstr = '%s: %s .. %s %s' % (self.name,
                                      self.data[0],
                                      self.data[-1],
                                      self.units)
        return outstr

    def __repr__(self):
        outstr = 'Dim(%s,\n\t%s,\n\tunits=%s)' % (self.name.__repr__(),
                                                  self.data.__repr__(),
                                                  self.units.__repr__())
        return outstr

    def __getitem__(self, item):
        """
        :Parameters:
            item : ``slice``
                The slice of the data to take.
        
        :Returns: ``numpy.ndarray``
        """
        return Dim(self.name,self.data.copy()[item],self.units)
        #return self.data[item]
        
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

        # make sure the num dims match the shape of the data
        if len(data.shape) != len(self.dims.dims):
            # raise error
            raise ValueError("The length of dims must match the length of the data shape.")

        # set the unit
        self.unit = unit

        self._reset_data_stats()

    def _reset_data_stats(self):
        """
        """
        # describe the data
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.ndim = len(self.shape)

    def copy(self):
        """
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
                # we have a single name, so return the data from it
                return self.dims[res.group()].data
            else:
                # call select
                return self.select(item)
        if isinstance(item,tuple):
            # call select
            return self.select(*item)
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
        data.find('time>kwargs['t']','events.recalled==kwargs['val']',t=0,val=True)

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
        data.data_ind('time>kwargs['t']','events.recalled==kwargs['val']',t=0,val=True)

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
        data.select('time>kwargs['t']','events.recalled==kwargs['val']',t=0,val=True)

        To get a tuple of index arrays for the selected conditions use the find method.
        """
        m_ind,ind = self._select_ind(*args,**kwargs)
        # set up the new data
        newdat = self.data[m_ind]

        # index the dims
        newdims = [dim[ind[d]] for dim,d in zip(self.dims,range(len(ind)))]
        
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
        
        #newdims = self.dims.copy()
        #newdims[dim] = newdims[dim].extend(other.dims[dim])

#         # make the new DimData
#         newDimData = self.copy()
#         newDimData.data = newdat
#         newDimData.dims = Dims(newdims)
#         newDimData._reset_data_stats()
#         return newDimData

    def aggregate(self,dimnames,function,unit=None,dimval=True):
        """
        Return a copy of the data aggregated over the dimensions
        specified in the list dimnames with the passed in
        function. The aggregation is done sequentially over the passed
        in dimensions in the order in which they are passed in. The
        unit is set to None, unless otherwise specified, because the
        transformation may change the unit.
        
        If dimval is False the aggregation is done over every
        dimension EXCEPT those specified in dimnames. In this case the
        data are aggregated over the not specified dimensions in the
        order of the dimension index values.
        """
        # If a string is passed in instead of a list or an array, convert:
        dimnames = N.atleast_1d(dimnames)
        if len(N.shape(dimnames)) != 1:
            raise ValueError("dimnames must be a 1-D list of dimension names.\
            Invalid value for dimnames: %s " % str(dimnames))

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

        # If dimnames is empty, len(db) will be 1 regardless of how many
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
            newdat = function(newdat,newDimData.dim(dimname))

        # Update newDimData:
        newDimData.data = newdat
        # The new dims are those not aggregated over:
        newDims = Dims(list(N.array(newDimData.dims.copy().dims)[-db]))
        newDimData.dims = newDims
        # Clean up & return:
        newDimData._reset_data_stats()
        return newDimData



