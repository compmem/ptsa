#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

# local imports
from basewrapper import BaseWrapper
from events import Events
from dimarray import Dim
from ptsa.data import TimeSeries

# global imports
import numpy as np
import string
import struct
import os
from scipy.io import loadmat

from glob import glob

class RawBinWrapper(BaseWrapper):
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
        self._samplerate = samplerate
        self.format = format
        self.gain = gain

        # see if can find them from a params file in dataroot
        self.params = self._get_params(dataroot)

        # set what we can from the params 
        if self.params.has_key('samplerate'):
            self._samplerate = self.params['samplerate']
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

    def _get_samplerate(self, channel=None):
        # Same samplerate for all channels:
        return self._samplerate

    def _get_nsamples(self,channel=None):
        # get the dimensions of the data
        # must open a valid channel and seek to the end
        eegfiles = glob(self.dataroot+'.*[0-9]')
        efile = open(eegfiles[0],'rb')
        efile.seek(0,2)
        nsamples = efile.tell()/self.nBytes
        if (efile.tell()%self.nBytes) != 0:
            raise ValueError('File length does not correspond to data format!')
        return nsamples

    def _get_channels(self):
        # get the dimensions of the data
        # must loop through directory identifying valid channels
        channel_files = glob(self.dataroot+'.*[0-9]')
        channels = [cf.split('.')[-1] for cf in channel_files]
        return channels

    def _get_annotations(self):
        # no annotations for raw data
        annot = None
        return annot

    def _get_params(self,dataroot):
        """Get parameters of the data from the dataroot."""
        # # set default params
        # params = {'samplerate':256.03,'gain':1.}
        params = {}

        # first look for dataroot.params file
        param_file = dataroot + '.params'
        if not os.path.isfile(param_file):
            # see if it's params.txt
            param_file = os.path.join(os.path.dirname(dataroot),'params.txt')
            if not os.path.isfile(param_file):
                raise IOError(
                    'No params file found in '+str(dataroot)+
                    '. Params files must be in the same directory '+
                    'as the EEG data and must be named \".params\" '+
                    'or \"params.txt\".')
                # return params
        
        # we have a file, so open and process it
        for line in open(param_file,'r').readlines():
            # get the columns by splitting
            cols = line.strip().split()
            # set the params
            params[cols[0]] = eval(string.join(cols[1:]))

        if (not params.has_key('samplerate')) or (not params.has_key('gain')):
            raise ValueError('Params file must contain samplerate and gain!\n'+
                             'The following fields were supplied:\n'+
                             str(params.keys()))
        # return the params dict
        return params
        

    def _load_data(self,channels,event_offsets,dur_samp,offset_samp):
        """
        """
        
        # allocate for data
        eventdata = np.empty((len(channels),len(event_offsets),dur_samp),
                             dtype=np.float)*np.nan

        # loop over channels
        for c,channel in enumerate(channels):
            # determine the file
            # eegfname = '%s.%03i' % (self.dataroot,channel)
            if isinstance(channel,str):
                eegfname = self.dataroot+'.'+channel
            elif int(channel)==channel: # isinstance distinguished
                                        # between np.int and np.int32
                eegfname = self.dataroot+'.'+str(channel)
            else:
                raise ValueError('Channel must be int or string, received '+
                                 str(type(channel))+'; value: '+str(channel))
            if os.path.isfile(eegfname):
                efile = open(eegfname,'rb')
            else:
                raise IOError('EEG file not found for channel '+str(channel)+
                              ' and file root '+self.dataroot)

            # loop over events
            for e,ev_offset in enumerate(event_offsets):
                # seek to the position in the file
                ssamp = offset_samp+ev_offset
                efile.seek(self.nBytes*ssamp,0)

                # read the data
                data = efile.read(int(self.nBytes*dur_samp))

                # convert from string to array based on the format
                # hard codes little endian
                data = np.array(struct.unpack('<'+str(len(data)/self.nBytes)+
                                              self.fmtStr,data))

                # make sure we got some data
                if len(data) < dur_samp:
                    raise IOError('Event with offset '+str(ev_offset)+
                                  ' is outside the bounds of file '+str(eegfname))

                # append it to the events
                eventdata[c,e,:] = data

        # multiply by the gain
	eventdata *= self.gain
        return eventdata

    ########################################
    def get_event_data(self,channels,events,
                       start_time,end_time,buffer_time=0.0,
                       resampled_rate=None,
                       filt_freq=None,filt_type='stop',filt_order=4,
                       keep_buffer=False,
                       loop_axis=None,num_mp_procs=0,eoffset='eoffset'):
        """
        Return an TimeSeries containing data for the specified channel
        in the form [events,duration].

        Parameters
        ----------
        channels: {int} or {dict}
            Channels from which to load data.
        events: {array_like} or {recarray}
            Array/list of event offsets (in seconds) into the data,
            specifying each event onset time.
        start_time: {float}
            Start of epoch to retrieve (in time-unit of the data).
        end_time: {float}
            End of epoch to retrieve (in time-unit of the data).
        buffer_time: {float},optional
            Extra buffer to add on either side of the event in order
            to avoid edge effects when filtering (in time unit of the
            data).
        resampled_rate: {float},optional
            New samplerate to resample the data to after loading.
        filt_freq: {array_like},optional
            The range of frequencies to filter (depends on the filter
            type.)
        filt_type = {scipy.signal.band_dict.keys()},optional
            Filter type.
        filt_order = {int},optional
            The order of the filter.
        keep_buffer: {boolean},optional
            Whether to keep the buffer when returning the data.
        """

        # translate back to dur and offset
        dur = end_time - start_time
        offset = start_time
        buf = buffer_time

        # get the event offsets
        if ((not (hasattr(events,'dtype') or hasattr(events,'columns'))) or
            (hasattr(events,'dtype') and events.dtype.names is None)):
            # they just passed in a list
            event_offsets = events
        elif ((hasattr(events, 'dtype') and (eoffset in events.dtype.names)) or
              (hasattr(events, 'columns') and (eoffset in events.columns))):
            event_offsets = events[eoffset]
        else:
            raise ValueError(eoffset+' must be a valid fieldname '+
                             'specifying the offset for the data.')
        
        # Sanity checks:
        if(dur<0):
            raise ValueError('Duration must not be negative! '+
                             'Specified duration: '+str(dur))
        if(np.min(event_offsets)<0):
            raise ValueError('Event offsets must not be negative!')

        # make sure the events are an actual array
        event_offsets = np.asarray(event_offsets)
        
        # set event durations from rate
        # get the samplesize
        samplesize = 1./self.samplerate

        # get the number of buffer samples
        buf_samp = int(np.ceil(buf/samplesize))

        # calculate the offset samples that contains the desired offset
        offset_samp = int(np.ceil((np.abs(offset)-samplesize*.5)/samplesize)*
                          np.sign(offset))

        # finally get the duration necessary to cover the desired span
        #dur_samp = int(np.ceil((dur - samplesize*.5)/samplesize))
        dur_samp = (int(np.ceil((dur+offset - samplesize*.5)/samplesize)) -
                    offset_samp + 1)
        
        # add in the buffer
        dur_samp += 2*buf_samp
        offset_samp -= buf_samp

        # check that we have all the data we need before every event:
        if(np.min(event_offsets+offset_samp)<0):
            bad_evs = ((event_offsets+offset_samp)<0)
            raise ValueError('The specified values for offset and buffer '+
                             'require more data than is available before '+
                             str(np.sum(bad_evs))+' of all '+
                             str(len(bad_evs))+' events.')

        # process the channels
        if isinstance(channels, dict):
            # turn into indices
            ch_info = self.channels
            key = channels.keys()[0]
            channels = [np.nonzero(ch_info[key]==c)[0][0] for c in channels[key]]
        elif isinstance(channels, str):
            # find that channel by name
            channels = np.nonzero(self.channels['name']==channels)[0][0]
        if channels is None or len(np.atleast_1d(channels))==0:
            channels = np.arange(self.nchannels)
        channels = np.atleast_1d(channels)
        channels.sort()

        # load the timeseries (this must be implemented by subclasses)
        eventdata = self._load_data(channels,event_offsets,dur_samp,offset_samp)

        # calc the time range
        # get the samplesize
        samp_start = offset_samp*samplesize
        samp_end = samp_start + (dur_samp-1)*samplesize
        time_range = np.linspace(samp_start,samp_end,dur_samp)

        # make it a timeseries
        dims = [Dim(self.channels[channels],'channels'),  # can index into channels
                Dim(events,'events'),
                Dim(time_range,'time')]
        eventdata = TimeSeries(np.asarray(eventdata),
                               'time',
                               self.samplerate,dims=dims)

	# filter if desired
	if not(filt_freq is None):
	    # filter that data
            eventdata = eventdata.filtered(filt_freq,
                                           filt_type=filt_type,
                                           order=filt_order)

	# resample if desired
	if (not(resampled_rate is None) and
            not(resampled_rate == eventdata.samplerate)):
	    # resample the data
            eventdata = eventdata.resampled(resampled_rate,
                                            loop_axis=loop_axis,
                                            num_mp_procs=num_mp_procs)

        # remove the buffer and set the time range
	if buf > 0 and not(keep_buffer):
	    # remove the buffer
            eventdata = eventdata.remove_buffer(buf)

        # return the timeseries
        return eventdata


    

def createEventsFromMatFile(matfile):
    """Create an events data array with data wrapper information from
    an events structure saved in a Matlab mat file."""
    # load the mat file
    mat = loadmat(matfile)

    if 'events' not in mat.keys():
        raise "\nError processing the Matlab file: %s\n" + \
              "This file must contain an events structure" + \
              "with the name \"events\" (case sensitive)!\n" +\
              "(All other content of the file is ignored.)" % matfile 

    # get the events
    events = mat['events'][0]
    
    # get num events
    numEvents = len(events)

    # determine the fieldnames and formats
    fields = events[0]._fieldnames

    def loadfield(events,field,dtype=None):
        data = []
        for x in events:
            dat = getattr(x,field)
            if len(dat) == 0:
                data.append(None)
            else:
                data.append(dtype(dat[0]))
        return data
    
    # create list with array for each field
    data = []
    for f,field in enumerate(fields):
	# handle special cases
	if field == 'eegfile':
	    # get unique files
	    #eegfiles = np.unique(map(lambda x: str(x.eegfile),events))
	    #eegfiles = np.unique([str(x.eegfile[0]) for x in events])
            eegfiles = np.unique(loadfield(events,field))
            eegfiles = eegfiles[eegfiles!=None]
	    
	    # make dictionary of data wrapers for the unique eeg files
	    efile_dict = {}
	    for eegfile in eegfiles:
		efile_dict[eegfile] = RawBinWrapper(eegfile)

	    # Handle when the eegfile field is blank
	    efile_dict[''] = None
	
	    # set the eegfile to the correct data wrapper
            
	    # newdat = np.array(
            #     map(lambda x: efile_dict[str(x.__getattribute__(field))],
            #         events))
            newdat = np.array([efile_dict[str(x.__getattribute__(field)[0])]
                               for x in events])
			
	    # change field name to esrc
	    fields[f] = 'esrc'
	elif field == 'eegoffset':
            # change the field name
            fields[f] = 'eoffset'
            newdat = np.array([x.__getattribute__(field)[0]
                               for x in events])
	else:
	    # get the data in normal fashion
	    # newdat = np.array(
            #     map(lambda x: x.__getattribute__(field),events))
            newdat = np.array([x.__getattribute__(field)[0]
                               for x in events])

	# append the data
	data.append(newdat)

    # allocate for new array
    newrec = np.rec.fromarrays(data,names=fields).view(Events)

    return newrec


