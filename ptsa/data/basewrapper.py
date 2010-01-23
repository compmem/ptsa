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
    # required properties that the child class must set.
    samplerate = None
    
    def _load_data(self,channel,event_offsets,dur_samp,offset_samp):
        """
        Method for loading data that each child wrapper class must
        implement.

        Parameters
        ----------
        channel : {int}
            Channel to load.
        event_offsets : {array_like}
            List of offsets into the data (in samples) marking event
            onsets.
        dur_samp : {int}
            Duration in samples of each event.
        offset_samp : {int}
            Offset from the event onset from where to extract the
            duration of the event.

        Returns
        -------
        data : {ndarray}
            Array of data for the specified channel in the form
            [events, duration].
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
        
        # set event durations from rate
        # get the samplesize
        samplesize = 1./self.samplerate
        # get the number of buffer samples
        buf_samp = int(np.ceil(buf/samplesize))
        # calculate the offset samples that contains the desired offsetMS
        offset_samp = int(np.ceil((np.abs(offset)-samplesize*.5)/samplesize)*
                          np.sign(offset))

        # finally get the duration necessary to cover the desired span
        dur_samp = (int(np.ceil((dur+offset - samplesize*.5)/samplesize)) -
                    offset_samp + 1)
        
        # add in the buffer
        dur_samp += 2*buf_samp
        offset_samp -= buf_samp

        # load the timeseries (this must be implemented by subclasses)
        eventdata = self._load_data(channel,event_offsets,dur_samp,offset_samp)

        # calc the time range
        # get the samplesize
        sampStart = offset_samp*samplesize
        sampEnd = sampStart + (dur_samp-1)*samplesize
        timeRange = np.linspace(sampStart,sampEnd,dur_samp)

	# make it a timeseries
        # if isinstance(eventInfo,TsEvents):
        #     dims = [Dim('event', eventInfo.data, 'event'),
        #             Dim('time',timeRange)]
        # else:
        #     dims = [Dim('event_offsets', event_offsets, 'samples'),
        #             Dim('time',timeRange)]            
        dims = [Dim(event_offsets,'event_offsets'),
                Dim(timeRange,'time')]
        eventdata = TimeSeries(np.asarray(eventdata),
                               'time',
                               self.samplerate,dims=dims)

	# filter if desired
	if not(filt_freq is None):
	    # filter that data
            eventdata = eventdata.filter(filt_freq,
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
