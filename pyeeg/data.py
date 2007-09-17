# local imports
import filter

# global imports
import numpy as N

from scipy.io import loadmat
from scipy.signal import resample

import os
import glob
import string
import struct 
import cPickle
import re
import sys

import pdb

# Define exceptions:
class DataException(Exception): pass
class EventsMatFileError(DataException): pass
class FilterStringError(DataException): pass

class DataWrapper(object):
    """
    Base class to provide interface to timeseries data.  
    """
    def getDataMS(self,channels,eventOffsets,DurationMS,OffsetMS,BufferMS,resampledRate=None,filtFreq=None,filtType='stop',filtOrder=4,keepBuffer=False):
        raise RuntimeError("You must use a child class that overloads the getDataMS method.")

class RawBinaryEEG(DataWrapper):
    """
    Interface to data stored in binary format with a separate file for
    each channel.  
    """
    def __init__(self,dataroot,samplerate=None,format='int16',gain=1):
        """Initialize the interface to the data.  You must specify the
        dataroot, which is a string that contains the path to and
        root, up to the channel numbers, where the data are stored."""
        # set up the basic params of the data
        self.dataroot = dataroot
        self.samplerate = samplerate
        self.format = format
        self.gain = gain

        # see if can find them from a params file in dataroot
        self.params = self._getParams(dataroot)

        # set what we can from the params 
        if self.params.has_key('samplerate'):
            self.samplerate = self.params['samplerate']
        if self.params.has_key('format'):
            self.format = self.params['format']
        if self.params.has_key('dataformat'):
            self.format = self.params['dataformat']
        if self.params.has_key('gain'):
            self.gain = self.params['gain']

        # set the nBytes and format str
        if self.format == 'single':
            self.nBytes = 4
            self.fmtStr = 'f'
        elif self.format == 'short' or self.format == 'int16':
            self.nBytes = 2
            self.fmtStr = 'h'
        elif self.format == 'double':
            self.nBytes = 8
            self.fmtStr = 'd'

    def _getParams(self,dataroot):
        """Get parameters of the data from the dataroot."""
        # set default params
        params = {'samplerate':256.03,'gain':1.}

        # first look for dataroot.params file
        paramFile = dataroot + '.params'
        if not os.path.isfile(paramFile):
            # see if it's params.txt
            paramFile = os.path.join(os.path.dirname(dataroot),'params.txt')
            if not os.path.isfile(paramFile):
                #raise "file not found"  # fix this
                return params
        
        # we have a file, so open and process it
        for line in open(paramFile,'r').readlines():
            # get the columns by splitting
            cols = line.strip().split()
            # set the params
            params[cols[0]] = eval(string.join(cols[1:]))
        
        # return the params dict
        return params
        

    def getDataMS(self,channel,eventOffsets,DurationMS,OffsetMS,BufferMS,
                  resampledRate=None,filtFreq=None,filtType='stop',filtOrder=4,keepBuffer=False):
        """
        Return an dictionary containing data for the specified channel
        in the form [events,duration].

        INPUT ARGS:

        channel: Channel to load data from
        eventOffsets: Array of even offsets (in samples) into the data, specifying each event time
        DurationMS: Duration in ms of the data to return.
        OffsetMS: Amount in ms to offset that data around the event.
        BufferMS: Extra buffer to add when doing filtering to avoid edge effects.
        resampledRate: New samplerate to resample the data to after loading.
        filtFreq: Frequency specification for filter (depends on the filter type.
        filtType: Type of filter to run on the data.
        filtOrder: Order of the filter.
        keepBuffer: Whether to keep the buffer when returning the data.
        
        """
        # set event durations from rate
        duration = int(N.fix((DurationMS+(2*BufferMS))*self.samplerate/1000.))
        offset = int(N.fix((OffsetMS-BufferMS)*self.samplerate/1000.))
        buffer = int(N.fix((BufferMS)*self.samplerate/1000.))

        # determine the file
	eegfname = '%s.%03i' % (self.dataroot,channel)
	if os.path.isfile(eegfname):
	    efile = open(eegfname,'rb')
	else:
	    # try unpadded lead
	    eegfname = '%s.%03i' % (self.dataroot,channel)
	    if os.path.isfile(eegfname):
		efile = open(eegfname,'rb')
	    else:
		raise IOError('EEG file not found for channel %i and file root %s\n' 
			      % (channel,self.dataroot))
                
	# loop over events
	eventdata = []
	eventOffsets = N.asarray(eventOffsets)
	if len(eventOffsets.shape)==0:
	    eventOffsets = [eventOffsets]
	for evOffset in eventOffsets:
	    # seek to the position in the file
	    thetime = offset+evOffset
	    efile.seek(self.nBytes*thetime,0)

	    # read the data
	    data = efile.read(int(self.nBytes*duration))

	    # make sure we got some data
	    if len(data) < duration:
		raise IOError('Event with offset %d is outside the bounds of file %s.\n'
			      % (evOffset,eegfname))
                
	    # convert from string to array based on the format
	    # hard codes little endian
	    data = N.array(struct.unpack('<'+str(len(data)/self.nBytes)+self.fmtStr,data))

	    # append it to the events
	    eventdata.append(data)

	# make it an array
	eventdata = N.array(eventdata)

	# turn the data into an EEGArray
        #eventdata = InfoArray(eventdata,info={'samplerate':self.samplerate})

	# filter if desired
	if not filtFreq is None:
	    # filter that data
	    eventdata = filter.buttfilt(eventdata,filtFreq,self.samplerate,filtType,filtOrder)

	# decimate if desired
	samplerate = self.samplerate
	if not resampledRate is None and not resampledRate == self.samplerate:
	    # resample the data
	    newLength = N.fix(eventdata.shape[1]*resampledRate/float(self.samplerate))
	    eventdata = resample(eventdata,newLength,axis=1)
	    
	    # set the new buffer length
	    buffer = int(N.fix(buffer*resampledRate/float(self.samplerate)))
	    
	    # set the new samplerate
	    samplerate = resampledRate

        # remove the buffer and set the time range
	if buffer > 0 and not keepBuffer:
	    # remove the buffer
	    eventdata = eventdata[:,buffer:-buffer]
	    # set the time range with no buffer
	    timeRange = N.linspace(OffsetMS,OffsetMS+DurationMS,eventdata.shape[1])
	    # reset buffer to indicate it was removed
	    buffer = 0
	else:
	    # keep the buffer, but set the time range
	    timeRange = N.linspace(OffsetMS-BufferMS,
				   OffsetMS+DurationMS+BufferMS,
				   eventdata.shape[1])
        # multiply by the gain and return
	eventdata = eventdata*self.gain
	
	# make dictinary of results
	res = {'data': eventdata, 
	       'samplerate': samplerate,
	       'time': timeRange,
	       'OffsetMS': OffsetMS,
	       'DurationMS': DurationMS,
	       'BufferMS': BufferMS,
	       'bufLen': buffer}
        return DataDict(res)

# data array subclass of recarray
class DataArray(N.recarray):
    """ Class that extends the record array so that it's easy to filter
    your records based on various conditions.  """
    def filterIndex(self,filterStr,iterate=False):
        """
        Return the boolean index to filter events based on a filter
        string.  You only need to iterate if there is not a ufunc for
        your test.  For example, if you want to perform a string.find
        on a field in the events.
        """
        
        # see if must iterate or vectorize
        if not iterate:
            # vectorize, so replace fields
            for k in self.dtype.names:
                # prepend "self." to fields in event structure
                # RE makes sure to not replace substrings
                filterStr = re.sub(r'\b'+k+r'\b','self.'+k,filterStr)
                
            # eval to set the boolean indices
            try:
                ind = eval(filterStr)
            except:
                raise FilterStringError, str(sys.exc_info()[0])+"\n"+str(sys.exc_info()[1])+"\nThe filter string is not a valid Python expression!\nExample string:\n\'(var1==0) & (var2<=0)\'"
        else:
            # must iterate over each one to get indices
            for k in self.dtype.names:
                # prepend "self." and append "[i]" to fields in event structure
                # RE makes sure to not replace substrings
                filterStr = re.sub(r'\b'+k+r'\b','self.'+k+'[i]',filterStr)
                
            # apply filter to each item
            try:
                ind = eval(filterStr)
            except:
                raise FilterStringError, str(sys.exc_info()[0])+"\n"+str(sys.exc_info()[1])+"\nThe filter string is not a valid Python expression!\nExample string:\n\'(var1==0) & (var2<=0)\'"

        # return the ind as an array
        return N.asarray(ind)

    def filter(self,filterStr,iterate=False):
        """
        Run filterInd and return its application to the records. 

	Note that this does not make a copy of the data, it only
	slices it."""
        # get the filter index
        ind = self.filterIndex(filterStr,iterate)

        # apply the filter
	return self[ind]

    def removeFields(self,*fieldsToRemove):
        """
        Return a new instance of the array with specified fields
        removed.
        """
        # sequence of arrays and names
        arrays = []
        names = ''

        # loop over fields, keeping if not matching fieldName
        for field in self.dtype.names:
            # don't add the field if in fieldsToRemove list
            if sum(map(lambda x: x==field,fieldsToRemove)) == 0:
                # append the data
                arrays.append(self[field])
                if len(names)>0:
                    # append ,
                    names = names+','
                # append the name
                names = names+field

        # return the new recarray
        return self.__class__(N.rec.fromarrays(arrays,names=names))

    def addFields(self,**fields):
        """ Add fields from the keyword args provided and return a new
        instance.  To add an empty field, pass a dtype as the array.

        addFields(name1=array1, name2=dtype('i4'))
        
        """
        # sequence of arrays and names from starting recarray
        arrays = map(lambda x: self[x], self.dtype.names)
        names = string.join(self.dtype.names,',')
        
        # loop over the kwargs of field
        for name,data in fields.iteritems():
            # see if already there, error if so
            if self.dtype.fields.has_key(name):
                # already exists
                raise AttributeError, 'Field "'+name+'" already exists.'
            
            # append the array and name
            if isinstance(data,N.dtype):
                # add empty array the length of the data
                arrays.append(N.empty(len(self),data))
            else:
                # add the data as an array
                arrays.append(data)

            # add the name
            if len(names)>0:
                # append ,
                names = names+','
            # append the name
            names = names+name

        # return the new recarray
        return self.__class__(N.rec.fromarrays(arrays,names=names))



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

	
class Events(DataArray):
    """Class to hold EEG events.  The record fields must include both
eegsrc and eegoffset so that the class can know how to retrieve data
for each event."""
    def getDataMS(self,channel,DurationMS,OffsetMS,BufferMS,resampledRate=None,filtFreq=None,filtType='stop',filtOrder=4,keepBuffer=False):
        """
        Return the requested range of data for each event by using the
        proper data retrieval mechanism for each event.

        The result will be a dictionary with an EEG array of
        dimensions (events,time) for the data and also some
        information about the data returned.  """
	# get ready to load dat
	eventdata = []
        
	# could be sped up by get unique event sources first
	
	if len(self.shape)==0:
	    events = [self]
	else:
	    events = self
	# loop over events
	for evNo,ev in enumerate(events):
	    # get the eeg
	    newdat = ev['eegsrc'].getDataMS(channel,
					    ev['eegoffset'],
					    DurationMS,
					    OffsetMS,
					    BufferMS,
					    resampledRate,
					    filtFreq,
					    filtType,
					    filtOrder,
					    keepBuffer)

	    # allocate if necessary
	    if len(eventdata) == 0:
		# make ndarray with events by time
		eventdata = N.empty((len(events),newdat['data'].shape[1]),
				    dtype=newdat['data'].dtype)

	    # fill up the eventdata
	    eventdata[evNo,:] = newdat['data'][0,:]

	# set the newdat to hold all the data
	newdat['data'] = eventdata

	# force uniform samplerate, so if no resampledRate is
	# provided, fix to samplerate of first event.

	# return (events, time) ndarray with samplerate
	return newdat

#     # we need to make sure and return an our custom class if it's a
#     # single record, this will ensure we can still call our custom
#     # methods
#     def __getitem__(self, indx):
#         obj = N.ndarray.__getitem__(self, indx)
#         if (isinstance(obj, N.ndarray) and obj.dtype.isbuiltin):
#             return obj.view(N.ndarray)
# 	elif isinstance(obj, N.record):
# 	    # return record as our custom recarray XXX Eventually this
# 	    # should return a EventRecord instance.
# 	    return self.__class__(obj)
#         return obj

	
def createEventsFromMatFile(matfile):
    """Create an events data array from an events structure saved in a
    Matlab mat file."""
    # load the mat file
    mat = loadmat(matfile)

    if 'events' not in mat.keys():
        raise EventsMatFileError, "\nError processing the Matlab file: %s\nThis file must contain an events structure with the name \"events\" (case sensitive)!\n(All other content of the file is ignored.)" % matfile 
    
    # get num events
    numEvents = len(mat['events'])

    # determine the fieldnames and formats
    fields = mat['events'][0]._fieldnames
    
    # create list with array for each field
    data = []
    hasEEGInfo = False
    for f,field in enumerate(fields):
	# handle special cases
	if field == 'eegfile':
	    # we have eeg info
	    hasEEGInfo = True

	    # get unique files
	    eegfiles = N.unique(map(lambda x: str(x.eegfile),mat['events']))
	    
	    # make dictionary of data wrapers for the eeg files
	    efile_dict = {}
	    for eegfile in eegfiles:
		efile_dict[eegfile] = RawBinaryEEG(eegfile)

	    # Handle when the eegfile field is blank
	    efile_dict[''] = None
	
	    # set the eegfile to the correct data wrapper
	    newdat = N.array(map(lambda x: efile_dict[str(x.__getattribute__(field))],
				 mat['events']))
			
	    # change field name to eegsrc
	    fields[f] = 'eegsrc'
	else:
	    # get the data in normal fashion
	    newdat = N.array(map(lambda x: x.__getattribute__(field),mat['events']))

	# append the data
	data.append(newdat)

    # allocate for new array
    newrec = N.rec.fromarrays(data,names=fields)

    # see if process into DataArray or Events
    if hasEEGInfo:
	#newrec = Events(newrec)
	newrec = newrec.view(Events)
    else:
	#newrec = DataArray(newrec)
	newrec = newrec.view(DataArray)
    return newrec


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


class EegTimeSeries(object):
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
                 
