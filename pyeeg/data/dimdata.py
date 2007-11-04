
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

        return Dim(self.name,N.concatenate((self.data,other.data),axis=0),self.units)
    
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
        # save the names list and a regexp for it
        self.names = [dim.name for dim in dims]
        regexpNames = '\\b'+'\\b|\\b'.join(self.names)+'\\b'
        self.namesRE = re.compile(regexpNames)

        regexpNameOnly = '(?<!.)\\b' + '\\b(?!.)|(?<!.)\\b'.join(self.names) + '\\b(?!.)'
        self._nameOnlyRE = re.compile(regexpNameOnly)

        # set the dims
        self.dims = dims

    def index(self,name):
        return self.names.index(name)

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
            raise ValueError, "The length of dims must match the length of the data shape."

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

        
    def select(self,*args,**kwargs):
        """
        Return a copy of the data filtered with the select conditions.

        data.select('time>0','events.recalled==True')
        or
        data.select(time=data['time']>0,events=data['events'].recalled==True)
        or 
        data.select('time>kwargs['t']','events.recalled==kwargs['val']',t=0,val=True)
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
        newdat = N.concatenate((self.data,other.data),axis=dim)

        # set the new dims
        newdims = self.dims.copy()
        newdims[dim] = newdims[dim].extend(other.dims[dim])

        # make the new DimData
        newDimData = self.copy()
        newDimData.data = newdat
        newDimData.dims = Dims(newdims)
        newDimData._reset_data_stats()
        return newDimData

