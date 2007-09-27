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
        # get the samplesize in ms
        samplesize = 1000./self.samplerate
        # get the number of buffer samples
        buffer = int(N.ceil(BufferMS/samplesize))
        # calculate the offset samples that contains the desired offsetMS
        offset = int(N.ceil((N.abs(OffsetMS)-samplesize*.5)/samplesize)*N.sign(OffsetMS))

        # finally get the duration necessary to cover the desired span
        duration = int(N.ceil((DurationMS+OffsetMS - samplesize*.5)/samplesize)) - offset + 1
        
        # add in the buffer
        duration += 2*buffer
        offset -= buffer

#         # calculate the duration samples that contain the desired ending point
#         buffer = int(N.ceil(BufferMS*self.samplerate/1000.))
#         duration = int(N.ceil(DurationMS*self.samplerate/1000.)) + 2*buffer
#         offset = int(N.ceil(OffsetMS*self.samplerate/1000.)) + buffer

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

        # calc the time range in MS
        sampStart = offset*samplesize
        sampEnd = sampStart + (duration-1)*samplesize
        timeRange = N.linspace(sampStart,sampEnd,duration)

	# make it a timeseries
        dims = [Dim('eventOffsets', eventOffsets, 'samples'),
                Dim('time',timeRange,'ms')]
        eventdata = EegTimeSeries(N.array(eventdata),
                                  dims,
                                  self.samplerate,
                                  tdim=-1,
                                  buffer=buffer)

	# filter if desired
	if not filtFreq is None:
	    # filter that data
            eventdata.filter(filtFreq,filtType=filtType,order=filtOrder)

	# resample if desired
	samplerate = self.samplerate
	if not resampledRate is None and not resampledRate == eventdata.samplerate:
	    # resample the data
            eventdata.resample(resampledRate)

        # remove the buffer and set the time range
	if eventdata.buffer > 0 and not keepBuffer:
	    # remove the buffer
            eventdata.removeBuffer()

        # multiply by the gain and return
	eventdata.data = eventdata.data*self.gain
	
        return eventdata

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
        
        if len(self.shape)==0:
	    events = [self]
	else:
	    events = self

        # speed up by getting unique event sources first
        usources = N.unique1d(events['eegsrc'])

        # loop over unique sources
        for src in usources:
            # get the eventOffsets from that source
            ind = events['eegsrc']==src
            evOffsets = events['eegoffset'][ind]
                                      
            # get the timeseries for those events
            newdat = src.getDataMS(channel,
                                   evOffsets,
                                   DurationMS,
                                   OffsetMS,
                                   BufferMS,
                                   resampledRate,
                                   filtFreq,
                                   filtType,
                                   filtOrder,
                                   keepBuffer)

            # see if concatenate


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
        self.names = [dim.name for dim in dims]
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
        newDimData = self.copy()
        newDimData.data = newdat
        newDimData.dims = newdims
        newDimData._reset_data_stats()
        return newDimData

    def extend(self,other,dim):
        """
        Method to concatenate two DimData instances over a given dimension.
        """
        # set the new dat
        newdat = N.concatenate((self.data,other.data),axis=dim)

        # set the new dims
        newdims = [dim for dim in self.dims]
        newdims[dim] = newdims[dim].extend(other.dims[dim])

        # make the new DimData
        newDimData = self.copy()
        newDimData.data = newdat
        newDimData.dims = newdims
        newDimData._reset_data_stats()
        return newDimData


class EegTimeSeries(DimData):
    """
    Class to hold EEG timeseries data.
    """
    def __init__(self,data,dims,samplerate,unit=None,tdim=-1,buffer=None):
        """
        """
        # call the base class init
        DimData.__init__(self,data,dims,unit)
        
        # set the timeseries-specific information
        self.samplerate = samplerate
        
        # get the time dimension
        if tdim >= 0:
            # use it
            self.tdim = tdim
        else:
            # turn it into a positive dim
            self.tdim = tdim + self.ndim
        
        # set the buffer information
        self.buffer = buffer

    def copy(self):
        """
        """
        newdata = self.data.copy()
        newdims = self.dims.copy()
        return EegTimeSeries(newdata,newdims,self.samplerate,
                             unit=self.unit,tdim=self.tdim,buffer=self.buffer)

    def removeBuffer(self):
	"""Use the information contained in the time series to remove the
	buffer reset the time range.  If buffer is 0, no action is
	performed."""
	# see if remove the anything
	if self.buffer>0:
            # remove the buffer from the data
            self.data = self.data.take(range(self.buffer,
                                             self.shape[self.tdim]-self.buffer),self.tdim)

            # remove the buffer from the tdim
            self.dims[self.tdim] = self.dims[self.tdim][self.buffer:self.shape[self.tdim]-self.buffer]

            # reset buffer to indicate it was removed
	    self.buffer = 0

            # reset the shape
            self.shape = self.data.shape

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
        # resample the data, getting new time range
        timeRange = self.dims[self.tdim].data
        newLength = int(N.round(self.data.shape[self.tdim]*resampledRate/float(self.samplerate)))
        self.data,newTimeRange = resample(self.data,newLength,t=timeRange,axis=self.tdim,window=window)

#         # resample the tdim
#         # calc the time range in MS
#         timeRange = self.dims[self.tdim].data
#         samplesize = N.abs(timeRange[0]-timeRange[1])
#         newsamplesize = samplesize*self.samplerate/resampledRate
#         adjustment = (newsamplesize - samplesize)/2.
#         sampStart = timeRange[0] + adjustment
#         sampEnd = timeRange[-1] - adjustment
#         newTimeRange = N.linspace(sampStart,sampEnd,newLength)

        # set the time dimension
        self.dims[self.tdim].data = newTimeRange

        # set the new buffer lengths
        self.buffer = int(N.round(float(self.buffer)*resampledRate/float(self.samplerate)))

        # set the new samplerate
        self.samplerate = resampledRate

        # set the new shape
        self.shape = self.data.shape


        
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
                 
