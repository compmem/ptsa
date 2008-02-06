
# local imports
from datawrapper import DataWrapper
from events import Events,EegEvents
from dimdata import Dim,Dims
from timeseries import TimeSeries

# global imports
import numpy as N
import string
import struct
import os
from scipy.io import loadmat

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
        

    def get_event_data(self,channel,eventInfo,dur,offset,buf,resampledRate=None,
                       filtFreq=None,filtType='stop',filtOrder=4,keepBuffer=False):
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
        samplesize = 1./self.samplerate
        # get the number of buffer samples
        buf_samp = int(N.ceil(buf/samplesize))
        # calculate the offset samples that contains the desired offsetMS
        offset_samp = int(N.ceil((N.abs(offset)-samplesize*.5)/samplesize)*N.sign(offset))

        # finally get the duration necessary to cover the desired span
        dur_samp = int(N.ceil((dur+offset - samplesize*.5)/samplesize)) - offset_samp + 1
        
        # add in the buffer
        dur_samp += 2*buf_samp
        offset_samp -= buf_samp

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
        # get the eventOffsets
        if isinstance(eventInfo,EegEvents):
            eventOffsets = eventInfo['eegoffset']
        else:
            eventOffsets = eventInfo
        eventOffsets = N.asarray(eventOffsets)
	if len(eventOffsets.shape)==0:
	    eventOffsets = [eventOffsets]
	for evOffset in eventOffsets:
	    # seek to the position in the file
	    thetime = offset_samp+evOffset
	    efile.seek(self.nBytes*thetime,0)

	    # read the data
	    data = efile.read(int(self.nBytes*dur_samp))

	    # make sure we got some data
	    if len(data) < dur_samp:
		raise IOError('Event with offset %d is outside the bounds of file %s.\n'
			      % (evOffset,eegfname))
                
	    # convert from string to array based on the format
	    # hard codes little endian
	    data = N.array(struct.unpack('<'+str(len(data)/self.nBytes)+self.fmtStr,data))

	    # append it to the events
	    eventdata.append(data)

        # calc the time range in MS
        sampStart = offset_samp*samplesize
        sampEnd = sampStart + (dur_samp-1)*samplesize
        timeRange = N.linspace(sampStart,sampEnd,dur_samp)

	# make it a timeseries
        if isinstance(eventInfo,EegEvents):
            dims = [Dim('event', eventInfo.data, 'event'),
                    Dim('time',timeRange,'ms')]
        else:
            dims = [Dim('eventOffsets', eventOffsets, 'samples'),
                    Dim('time',timeRange,'ms')]
        eventdata = TimeSeries(N.array(eventdata),
                               dims,
                               self.samplerate,
                               tdim=-1,
                               buf_samp=buf_samp)

	# filter if desired
	if not(filtFreq is None):
	    # filter that data
            eventdata.filter(filtFreq,filtType=filtType,order=filtOrder)

	# resample if desired
	samplerate = self.samplerate
	if not(resampledRate is None) and not(resampledRate == eventdata.samplerate):
	    # resample the data
            eventdata.resample(resampledRate)

        # remove the buffer and set the time range
	if eventdata.buf_samp > 0 and not(keepBuffer):
	    # remove the buffer
            eventdata.removeBuf()

        # multiply by the gain and return
	eventdata.data = eventdata.data*self.gain
	
        return eventdata


def createEventsFromMatFile(matfile):
    """Create an events data array from an events structure saved in a
    Matlab mat file."""
    # load the mat file
    mat = loadmat(matfile)

    if 'events' not in mat.keys():
        raise "\nError processing the Matlab file: %s\nThis file must contain an events structure with the name \"events\" (case sensitive)!\n(All other content of the file is ignored.)" % matfile 
    
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
	newrec = EegEvents(newrec)
    else:
	newrec = Events(newrec)

    return newrec


