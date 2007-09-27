
# BaseDict code from: 
#   http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/473790
# import cPickle
# import moved to top of file

class BaseDict(dict):
    '''
        A dict allows inputting data as adict.xxx as well as adict['xxx']

        In python obj:
        
            obj.var=x  ---> 'var' be in obj.__dict__ (this is python default)

        In python dict:
        
            dict['var']=x ---> 'var' be in dict.items(this is dict behavior)

        In BaseDict:  

            let bd=BaseDict()
            
            Both bd.var and bd['var'] will save x to bd.items 
            and bd.setDict('var', x) will save x to bd.__dict__ 

            This allows an easier access of the variables.        
  
    '''
    def __init__(self, data=None):
        if data:  dict.__init__(self, data)
        else:     dict.__init__(self)
        dic = self.__dict__
        dic['__ver__']   ='20041208_1'
        dic['__author__']='Runsun Pan'
    
    def __setattr__(self, name, val):
        if name in self.__dict__:  self.__dict__[name]= val        
        else:   self[name] = val
        
    def __getattr__(self, name):
        if name in self.__dict__:  return self.__dict__[name]        
        else:  return self[name] 
           
    def setDict(self, name, val): 
        '''
            setDict(name, val): Assign *val* to the key *name* of __dict__.
         
            :Usage:
            
            >>> bd = BaseDict()
            >>> bd.getDict()['height']   
            Traceback (most recent call last):
            ...
            KeyError: 'height'
            >>> bd.setDict('height', 160)  # setDict 
            {}
            >>> bd.getDict()['height']
            160

            '''
        self.__dict__[name] = val
        return self 

    def getDict(self): 
        ''' 
            Return the internal __dict__.
            
            :Usage:
            
            >>> bd = BaseDict()
            >>> bd.getDict()['height']
            Traceback (most recent call last):
            ...
            KeyError: 'height'
            >>> bd.setDict('height', 160)
            {}
            >>> bd.getDict()['height']
            160
            '''
        return self.__dict__
        
    def setItem(self, name, val): 
        ''' 
            Set the value of dict key *name* to *val*. Note this dict 
            is not the __dict__.

            :Usage:
            
            >>> bd = BaseDict()
            >>> bd
            {}
            >>> bd.setItem('sex', 'male')
            {'sex': 'male'}
            >>> bd['sex'] = 'female'
            >>> bd
            {'sex': 'female'}
            '''
        self[name] = val
        return self
    
    def __getstate__(self): 
        ''' Needed for cPickle in .copy() '''
        return self.__dict__.copy() 

    def __setstate__(self,dict): 
        ''' Needed for cPickle in .copy() '''
        self.__dict__.update(dict)   

    def copy(self):   
        ''' 
            Return a copy. 
            
            :Usage:
            
            >>> bd = BaseDict()
            >>> bd['name']=[1,2,3]
            >>> bd
            {'name': [1, 2, 3]}
            >>> bd2 = bd.copy()
            >>> bd2
            {'name': [1, 2, 3]}
            >>> bd == bd2
            True
            >>> bd is bd2
            False
            >>> bd['name']==bd2['name']
            True
            >>> bd['name'] is bd2['name']
            False
            >>> bd2['name'][0]='aa'
            >>> bd2['height']=60
            >>> bd
            {'name': [1, 2, 3]}
            >>> bd2
            {'name': ['aa', 2, 3], 'height': 60}
                
            '''
        return cPickle.loads(cPickle.dumps(self))



class DataDict(BaseDict):
    """ Dictionary where you can access the values as attributes, but with
    added features for manipulating the data inside.  """
    def removeBuffer(self,fields,axis=-1):
	"""Use the information contained in the data dictionary to remove the
	buffer from the specified fields and reset the time range.  If
	bufLen is 0, no action is performed."""
	# see if remove the anything
	if self.bufLen>0:
	    # make sure it's a list
	    fields = N.asarray(fields)
	    if len(fields.shape)==0:
		fields = [fields]
	    for field in fields:
		# remove the buffer
		self[field] = self[field].take(range(self.bufLen,
						     self[field].shape[axis]-self.bufLen),
					       axis)
	    # set the time range with no buffer
	    self.time = N.linspace(self.OffsetMS,
				   self.OffsetMS+self.DurationMS,
				   self[fields[0]].shape[axis])
	    # reset buffer to indicate it was removed
	    self.bufLen = 0



class InfoArray(N.ndarray):
    def __new__(subtype, data, info=None, dtype=None, copy=True):
        # When data is an InfoArray
        if isinstance(data, InfoArray):
            if not copy and dtype==data.dtype:
                return data.view(subtype)
            else:
                return data.astype(dtype).view(subtype)
        subtype._info = info
        subtype.info = subtype._info
        return N.array(data).view(subtype)

    def __array_finalize__(self,obj):
        if hasattr(obj, "info"):
            # The object already has an info tag: just use it
            self.info = obj.info
        else:
            # The object has no info tag: use the default
            self.info = self._info

    def __repr__(self):
        desc="""\
array(data=
  %(data)s,
      tag=%(tag)s)"""
        return desc % {'data': str(self), 'tag':self.info }

class EegTimeSeries_old(object):
    """
    Holds timeseries data.  

    Keeps track of time dimension, samplerate, buffer, offset, duration, time.
    """
    def __init__(self,data,samplerate,tdim=-1,offsetMS=None,offset=None,bufferMS=None,buffer=None):
        """
        """
        # set the initial values
        self.data = data
        self.samplerate = samplerate
        self.dtype = data.dtype
        self.shape = data.shape
        self.ndim = len(data.shape)
        
        # get the time dimension
        if tdim >= 0:
            # use it
            self.tdim = tdim
        else:
            # turn it into a positive dim
            self.tdim = tdim + self.ndim

        # set the durations
        # set the offset (for events)
        if not offset is None:
            # set offsetMS from offset
            self.offset = offset
            self.offsetMS = float(offset)*1000./self.samplerate
        elif not offsetMS is None:
            # set the offset from the MS
            self.offsetMS = offsetMS
            self.offset = int(N.round(float(offsetMS)*self.samplerate/1000.))
        else:
            # default to no offset
            self.offset = 0
            self.offsetMS = 0
                
        # set the buffer
        if not buffer is None:
            # set bufferMS from buffer
            self.buffer = buffer
            self.bufferMS = float(buffer)*1000./self.samplerate
        elif not bufferMS is None:
            # set the buffer from the MS
            self.bufferMS = bufferMS
            self.buffer = int(N.round(float(bufferMS)*self.samplerate/1000.))
        else:
            # default to no buffer
            self.buffer = 0
            self.bufferMS = 0

        # set the duration (does not include the buffer)
        self.durationMS = (self.shape[self.tdim]-2*self.buffer)*1000./self.samplerate

        # set the time range
        self.trangeMS = N.linspace(self.offsetMS-self.bufferMS,
                                   self.offsetMS+self.durationMS+self.bufferMS,
                                   self.shape[self.tdim])

            
    def __getitem__(self, item):
        """
        :Parameters:
            item : ``slice``
                The slice of the data to take.
        
        :Returns: ``numpy.ndarray``
        """
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

    def removeBuffer(self):
	"""Use the information contained in the time series to remove the
	buffer reset the time range.  If buffer is 0, no action is
	performed."""
	# see if remove the anything
	if self.buffer>0:
            # remove the buffer
            self.data = self.data.take(range(self.buffer,
                                             self.shape[self.tdim]-self.buffer),self.tdim)

            # reset buffer to indicate it was removed
	    self.buffer = 0
            self.bufferMS = 0

            # reset the shape
            self.shape = self.data.shape

	    # set the time range with no buffer
	    self.trangeMS = N.linspace(self.offsetMS-self.bufferMS,
                                       self.offsetMS+self.durationMS+self.bufferMS,
                                       self.shape[self.tdim])



    def filter(self,freqRange,filtType='stop',order=4):
        """
        Filter the data using a Butterworth filter.
        """
        self.data = filter.buttfilt(self.data,freqRange,self.samplerate,filtType,
                                    order,axis=self.tdim)

    def resample(self,resampledRate,window=None):
        """
        Resample the data and reset all the time ranges.  Uses the
        resample function from scipy.  This method seems to be more
        accurate than the decimate method.
        """
        # resample the data
        newLength = N.fix(self.data.shape[self.tdim]*resampledRate/float(self.samplerate))
        self.data = resample(self.data,newLength,axis=self.tdim,window=window)

        # set the new offset and buffer lengths
        self.buffer = int(N.round(float(self.buffer)*resampledRate/float(self.samplerate)))
        self.offset = int(N.round(float(self.offset)*resampledRate/float(self.samplerate)))

        # set the new samplerate
        self.samplerate = resampledRate

        # set the new shape
        self.shape = self.data.shape

        # set the time range with no buffer
        self.trangeMS = N.linspace(self.offsetMS-self.bufferMS,
                                   self.offsetMS+self.durationMS+self.bufferMS,
                                   self.shape[self.tdim])


    def decimate(self,resampledRate, order=None, ftype='iir'):
        """
        Decimate the data and reset the time ranges.
        """

        # set the downfact
        downfact = int(N.round(float(self.samplerate)/resampledRate))

        # do the decimation
        self.data = filter.decimate(self.data,downfact, n=order, ftype=ftype, axis=self.tdim)

        # set the new offset and buffer lengths
        self.buffer = int(N.round(float(self.buffer)*resampledRate/float(self.samplerate)))
        self.offset = int(N.round(float(self.offset)*resampledRate/float(self.samplerate)))

        # set the new samplerate
        self.samplerate = resampledRate

        # set the new shape
        self.shape = self.data.shape

        # set the time range with no buffer
        self.trangeMS = N.linspace(self.offsetMS-self.bufferMS,
                                   self.offsetMS+self.durationMS+self.bufferMS,
                                   self.shape[self.tdim])
                 


# class EventRecord(N.record):
#     """Class to allow for accessing EEG data from a single record."""
#     def getDataMS(self,channel,DurationMS,OffsetMS,BufferMS,resampledRate=None,filtFreq=None,filtType='stop',filtOrder=4,keepBuffer=False):
# 	"""
#         Return the requested range of data for each event by using the
#         proper data retrieval mechanism for each event.

#         The result will be a dictionary with an EEG array of
#         dimensions (events,time) for the data and also some
#         information about the data returned.  """
# 	# get the data
# 	newdat = self['eegsrc'].getDataMS(channel,
# 					  self['eegoffset'],
# 					  DurationMS,
# 					  OffsetMS,
# 					  BufferMS,
# 					  resampledRate,
# 					  filtFreq,
# 					  filtType,
# 					  filtOrder,
# 					  keepBuffer)
# 	return newdat
