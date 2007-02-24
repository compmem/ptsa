
# necessary global imports
import numpy as N

from scipy.io import loadmat

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
    def getdataMS(self,channels,eventOffsets,DurationMS,OffsetMS,BufferMS,resampledRate=None,filtFreq=None,filtType='stop',filtOrder=4):
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
        

    def getdataMS(self,channels,eventOffsets,DurationMS,OffsetMS,BufferMS,resampledRate=None,filtFreq=None,filtType='stop',filtOrder=4):
        """
        Return an EEGArray of data for the specified channels,events,and durations.
        """
        # set event durations from rate
        duration = N.fix((DurationMS+(2*BufferMS))*self.samplerate/1000);
        offset = N.fix((OffsetMS-BufferMS)*self.samplerate/1000);
        buffer = N.fix((BufferMS)*self.samplerate/1000);

        # loop over channels
        chandata = []
        for chan in channels:
            # determine the file
            eegfname = '%s.%03i' % (self.dataroot,chan)
            if os.path.isfile(eegfname):
                efile = open(eegfname,'rb')
            else:
                # try unpadded lead
                eegfname = '%s.%03i' % (self.dataroot,chan)
                if os.path.isfile(eegfname):
                    efile = open(eegfname,'rb')
                else:
                    raise IOError('EEG file not found for channel %i and file root %s\n' 
                                  % (chan,self.dataroot))
                
            # loop over events
            eventdata = []
            for evOffset in eventOffsets:
                # seek to the position in the file
                thetime = offset+evOffset
                efile.seek(self.nBytes*thetime,0)

                # read the data
                data = efile.read(int(self.nBytes*duration))
                
                # convert from string to array based on the format
                # hard codes little endian
                data = N.array(struct.unpack('<'+str(len(data)/self.nBytes)+self.fmtStr,data))

                # filter if desired
                if not filtFreq is None:
                    # filter that data
                    data = filter.buttfilt(data,filtFreq,self.samplerate,filtType,filtOrder)

                # decimate if desired

                # append it to the events
                eventdata.append(data)

            # append the event data
            chandata.append(eventdata)

        # turn the data into an EEGArray
        chandata = InfoArray(chandata,info={'samplerate':self.samplerate})

        # remove the buffer
	if buffer > 0:
	    chandata = chandata[:,:,buffer:-buffer]

        # multiply by the gain and return
        return chandata*self.gain

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


# data array subclass of recarray
class DataArray(N.recarray):
    def __new__(subtype,obj):
        self = obj.view(subtype)
        return self
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
    def getDataMS(self,channels,DurationMS,OffsetMS,BufferMS,resampledRate,filtFreq=None,filtType='stop',filtOrder=4):
        """
        Return the requested range of data for each event by using the
        proper data retrieval mechanism for each event.

        The result will be an EEG array of dimensions (channels,events,time).
        """
        pass

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
    for field in fields:
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
	
	    # set the eegfile to the correct data wrapper
	    newdat = N.array(map(lambda x: efile_dict[str(x.__getattribute__(field))],
				     mat['events']))
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
