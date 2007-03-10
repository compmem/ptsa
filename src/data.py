
# necessary global imports
import numpy as N

from scipy.io import loadmat
from scipy.signal import resample

import os
import glob
import string
import struct 

import filter

import pdb

class DataWrapper:
    """
    Base class to provide interface to timeseries data.
    """
    def getdataMS(self,channels,eventOffsets,DurationMS,OffsetMS,BufferMS,resampledRate=None,filtFreq=None,filtType='stop',filtOrder=4,keepBuffer=False):
        pass

class RawBinaryEEG(DataWrapper):
    """
    Interface to data stored in binary format with a separate file for
    each channel.
    """
    def __init__(self,dataroot,samplerate=None,format='int16',gain=1):
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
        

    def getDataMS(self,channel,eventOffsets,DurationMS,OffsetMS,BufferMS,resampledRate=None,filtFreq=None,filtType='stop',filtOrder=4,keepBuffer=False):
        """
        Return an EEGArray of data for the specified channel,events,and durations.
        """
        # set event durations from rate
        duration = N.fix((DurationMS+(2*BufferMS))*self.samplerate/1000)
        offset = N.fix((OffsetMS-BufferMS)*self.samplerate/1000)
        buffer = N.fix((BufferMS)*self.samplerate/1000)

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
	    buffer = N.fix(buffer*resampledRate/float(self.samplerate))
	    
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
				   OffsetMS+DurationMS+2*BufferMS,
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
        return res

# data array subclass of recarray
class DataArray(N.recarray):
    def __new__(subtype, data, dtype=None, copy=True):
        # When data is an DataArray
        if isinstance(data, DataArray):
            if not copy and dtype==data.dtype:
                return data.view(subtype)
            else:
                return data.astype(dtype).view(subtype)
        return N.array(data).view(subtype)

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
                filterStr = filterStr.replace(k,'self.'+k)

            # eval to set the boolean indices
            ind = eval(filterStr)
        else:
            # must iterate over each one to get indices
            for k in self.dtype.names:
                filterStr = filterStr.replace(k,'self.'+k+'[i]')

            # apply filter to each item
            ind = [eval(filterStr) for i in xrange(self.len())]

        # return the ind as an array
        return N.asarray(ind)

    def filter(self,filterStr,iterate=False):
        """
        Run filterInd and return its application to the events.
        """
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

class Events(DataArray):
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

	# return (events, time) InfoArray with samplerate
	#return InfoArray(eventdata,info={'samplerate':eventdata[0].info['samplerate']})
	
	# return (events, time) ndarray with samplerate
	return newdat
		
def createEventsFromMatFile(matfile):
    """Create an events data array from an events structure saved in a
    Matlab mat file."""
    # load the mat file
    mat = loadmat(matfile)

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
	newrec = Events(newrec)
    else:
	newrec = DataArray(newrec)
    return newrec


def testcase():
    # hypothetical test case

    # load events
    ev = createEventsFromMatFile('events.mat')
    
    # split out two conditions
    rev = ev.filter('recalled==1')
    nev = ev.filter('recalled==0')

    # do sample erp by getting raw eeg and doing an average for a
    # single channel

    # get power for the events for a range of freqs
    freqs = range(2,81,2)
    DurationMS = 2500
    OffsetMS = -500
    BufferMS = 1000
    # should give me data struct with: power,phase,time,freqs
    
    # plot difference in mean power


