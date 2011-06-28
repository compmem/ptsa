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
    Base class to provide interface to data.  
    """
    # required methods that the child class must define.
    def get_samplerate(self,channel):
        """
        Returns sample rate for a given channel.

        Parameters
        ----------
        channel : {int,str}
            Either integer or (if appropriate for a given data format)
            a string label specifying the channel.
        
        Returns
        -------
        samplerate : {float}
            Samplerate for the specified channel.
        """
        raise NotImplementedError
    
    def _load_data(self,channel,event_offsets,dur_samp,offset_samp):
        """
        Method for loading data that each child wrapper class must
        implement.

        Parameters
        ----------
        channel : {int,str}
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
    
    def _load_all_data(self,channel,dur_chunk=7372800):
        """
        Method for loading all data in a given channel that each child
        wrapper class must implement.

        Parameters
        ----------
        channel : {int,str}
            Channel to load. Either integer number or (if appropriate
            for a given data format) a string label.
        dur_chunk: {int},optional
            Size of data chunks to read in in samples. This
            can be any positive integer up to the maximum value
            allowed for an int. Down the line a buffer array of this
            size gets allocated.

        Returns
        -------
        data : {ndarray}
            Array of all data for the specified channel in the form
            [duration].
        """
        raise NotImplementedError
    
    def get_event_data(self,channel,event_offsets,
                       dur,offset,buf,
                       resampled_rate=None,
                       filt_freq=None,filt_type='stop',filt_order=4,
                       keep_buffer=False):
        """
        Return an TimeSeries containing data for the specified channel
        in the form [events,duration].

        Parameters
        ----------
        channel: {int}
            Channel from which to load data.
        event_offsets: {array_like}
            Array/list of event offsets (in samples) into the data,
            specifying each event onset time.
        dur: {float}
            Duration of the data to return (in time-unit of the data).
        offset: {float}
            Amount (in time-unit of data) to offset the data around
            the event.
        buf: {float}
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

        # Sanity checks:
        if(dur<0):
            raise ValueError('Duration must not be negative! '+
                             'Specified duration: '+str(dur))
        if(np.min(event_offsets)<0):
            raise ValueError('Event offsets must not be negative!')

        # set event durations from rate
        # get the samplesize
        samplesize = 1./self.get_samplerate(channel)

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

        # load the timeseries (this must be implemented by subclasses)
        eventdata = self._load_data(channel,event_offsets,dur_samp,offset_samp)

        # calc the time range
        # get the samplesize
        samp_start = offset_samp*samplesize
        samp_end = samp_start + (dur_samp-1)*samplesize
        time_range = np.linspace(samp_start,samp_end,dur_samp)

	# make it a timeseries
        dims = [Dim(event_offsets,'event_offsets'),
                Dim(time_range,'time')]
        eventdata = TimeSeries(np.asarray(eventdata),
                               'time',
                               self.get_samplerate(channel),dims=dims)

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
            eventdata = eventdata.resampled(resampled_rate)

        # remove the buffer and set the time range
	if buf > 0 and not(keep_buffer):
	    # remove the buffer
            eventdata = eventdata.remove_buffer(buf)

        # return the timeseries
        return eventdata
