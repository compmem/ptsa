#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

# local imports
#from events import Events,TsEvents
from timeseries import TimeSeries,Dim

# global imports
import numpy as np

class BaseWrapper(object):
    """
    Base class to provide interface to data.  Child classes will
    provide methods to return data, samplerate, nsamples, nchannels,
    annotations, and channel information.  The annotations and channel
    information will typically be recarrays.
    """

    # required methods that the child class must define.
    def _get_samplerate(self,channel=None):
        """
        Returns sample rate for a dataset or given channel.

        Parameters
        ----------
        channel : {None,int,str}
            If specified, an integer or (if appropriate for a given data format)
            a string label specifying the channel.
        
        Returns
        -------
        samplerate : {float}
            Samplerate for the specified channel.
        """
        raise NotImplementedError
    
    def _get_nsamples(self,channel=None):
        """
        Returns the number of samples for a dataset or given channel.

        Parameters
        ----------
        channel : {None,int,str}
            If specified, an integer or (if appropriate for a given data format)
            a string label specifying the channel.
        
        Returns
        -------
        nsamples : {float}
            Number of samples for the dataset or specified channel.
        """
        raise NotImplementedError
    
    def _get_nchannels(self):
        """
        Returns the number of channels in a dataset.
        
        Returns
        -------
        nchannels : {int}
            Number of channels.
        """
        raise NotImplementedError
    
    def _get_annotations(self):
        """
        Returns the annotations associated with the dataset.

        Returns
        -------
        annotations : {array-like}
            Annotations
        """
        raise NotImplementedError
        
    def _set_annotations(self, annotations):
        """
        Set the annotations associated with the dataset.

        """
        raise NotImplementedError
        
    def _get_channel_info(self):
        """
        Returns the channel info associated with the dataset.

        Returns
        -------
        channel_info : {array-like}
            Channel information (e.g., names, locations, etc...)
        """
        # generate recarray of channel info based on nchannels
        return np.rec.fromarrays(zip(*[(i+1,'Ch%d'%(i+1)) 
                                       for i in range(self.nchannels)]),
                                 names='number,name')
        #raise NotImplementedError
        
    def _set_channel_info(self, channel_info):
        """
        Set the channel info associated with the dataset.

        """
        raise NotImplementedError
        
    def _load_data(self,channels,event_offsets,dur_samp,offset_samp):
        """
        Method for loading data that each child wrapper class must
        implement.

        Parameters
        ----------
        channels : {list,int,str}
            Channel to load. Either integer number or (if appropriate
            for a given data format) a string label.
        event_offsets : {array_like}
            List of offsets into the data (in samples) marking event
            onsets.
        dur_samp : {int}
            Duration in samples of each event.
        offset_samp : {int}
            Offset (in samples) from the event onset from where to
            extract the duration of the event.

        Returns
        -------
        data : {ndarray}
            Array of data for the specified channel in the form
            [events, duration].
        """
        raise NotImplementedError

    def append_data(self, data):
        """
        """
        raise NotImplementedError
        
    def set_channel_data(self, channel, data):
        """
        """
        raise NotImplementedError
            
    def get_event_data(self,channels,events,
                       start_time,end_time,buffer_time=0.0,
                       resampled_rate=None,
                       filt_freq=None,filt_type='stop',filt_order=4,
                       keep_buffer=False,
                       loop_axis=None,num_mp_procs=0,eoffset='eoffset',
                       eoffset_in_time=True):
        """
        Return an TimeSeries containing data for the specified channel
        in the form [events,duration].

        Parameters
        ----------
        channels: {int} or {dict}
            Channels from which to load data.
        events: {array_like} or {recarray}
            Array/list of event offsets (in time or samples as
            specified by eoffset_in_time; in time by default) into
            the data, specifying each event onset time.
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
        eoffset_in_time: {boolean},optional        
            If True, the unit of the event offsets is taken to be
            time (unit of the data), otherwise samples.
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

        # make sure the events are an actual array:
        event_offsets = np.asarray(event_offsets)
        if eoffset_in_time:
            # convert to samples
            event_offsets = np.atleast_1d(np.int64(
                np.round(event_offsets*self.samplerate)))
        
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

    def get_all_data(self, channels=None):
        """
        Return a TimeSeries containing all the data.
        """
        if channels is None:
            channels = np.arange(self.nchannels)
        dur_samp = self.nsamples
        data = self._load_data(channels,[0],dur_samp,0)
        # remove events dimension
        data = data[:,0,:]

        # turn it into a TimeSeries
        # get the samplesize
        samplesize = 1./self.samplerate

        # set timerange
        samp_start = 0*samplesize
        samp_end = samp_start + (dur_samp-1)*samplesize
        time_range = np.linspace(samp_start,samp_end,dur_samp)

	# make it a timeseries
        dims = [Dim(self.channels[channels],'channels'),
                Dim(time_range,'time')]
        data = TimeSeries(np.asarray(data),
                          'time',
                          self.samplerate,dims=dims)

        return data
    
    # class properties
    samplerate = property(lambda self: self._get_samplerate())
    nsamples = property(lambda self: self._get_nsamples())
    nchannels = property(lambda self: self._get_nchannels())
    annotations = property(lambda self: self._get_annotations(),
                           lambda self,annot: self._set_annotations(annot))
    channel_info = property(lambda self: self._get_channel_info(),
                            lambda self,chan_info: self._set_channel_info(chan_info))
    channels = property(lambda self: self._get_channel_info(),
                        lambda self,chan_info: self._set_channel_info(chan_info))
    data = property(lambda self: self.get_all_data())
