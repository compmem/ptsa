#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

class DataWrapper(object):
    """
    Base class to provide interface to timeseries data.  
    """
    def _load_data(self,channel,eventOffsets,dur_samp,offset_samp):
        raise NotImplementedError
    
    def get_event_data(self,channels,eventOffsets,
                       dur,offset,buf,
                       resampledRate=None,
                       filtFreq=None,filtType='stop',filtOrder=4,
                       keepBuffer=False):
        """
        Return an dictionary containing data for the specified channel
        in the form [events,duration].

        Parameters
        ----------
        channel: Channel to load data from
        eventOffsets: Array of event offsets (in samples) into the data,
                      specifying each event time
        Duration: Duration in ms of the data to return.
        Offset: Amount in ms to offset that data around the event.
        Buffer: Extra buffer to add when doing filtering to avoid edge effects.
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
        buf_samp = int(np.ceil(buf/samplesize))
        # calculate the offset samples that contains the desired offsetMS
        offset_samp = int(np.ceil((np.abs(offset)-samplesize*.5)/samplesize)*np.sign(offset))

        # finally get the duration necessary to cover the desired span
        dur_samp = int(np.ceil((dur+offset - samplesize*.5)/samplesize)) - offset_samp + 1
        
        # add in the buffer
        dur_samp += 2*buf_samp
        offset_samp -= buf_samp

        # load the timeseries (this must be implemented by subclasses)
        eventdata = self._load_data(channel,eventOffsets,dur_samp,offset_samp)

        # calc the time range
        sampStart = offset_samp*samplesize
        sampEnd = sampStart + (dur_samp-1)*samplesize
        timeRange = np.linspace(sampStart,sampEnd,dur_samp)

	# make it a timeseries
        # if isinstance(eventInfo,TsEvents):
        #     dims = [Dim('event', eventInfo.data, 'event'),
        #             Dim('time',timeRange)]
        # else:
        #     dims = [Dim('eventOffsets', eventOffsets, 'samples'),
        #             Dim('time',timeRange)]            
        dims = [Dim(eventOffsets,'eventOffsets'),
                Dim(timeRange,'time')]
        eventdata = TimeSeries(np.asarray(eventdata),
                               dims,
                               tdim='time',
                               self.samplerate)

        return eventdata


	# filter if desired
	if not(filtFreq is None):
	    # filter that data
            eventdata = eventdata.filter(filtFreq,filtType=filtType,order=filtOrder)

	# resample if desired
	if not(resampledRate is None) and \
               not(resampledRate == eventdata.samplerate):
	    # resample the data
            eventdata = eventdata.resampled(resampledRate)

        # remove the buffer and set the time range
	if buf > 0 and not(keepBuffer):
	    # remove the buffer
            eventdata = eventdata.removeBuf(buf)

        # return the timeseries
        return eventdata
