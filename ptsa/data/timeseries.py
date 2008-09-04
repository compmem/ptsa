#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from dimarray import Dim,DimArray
from ptsa import filt

from scipy.signal import resample
import numpy as np


__docformat__ = 'restructuredtext'


class TimeSeries(DimArray):
    """Class to hold continuous timeseries data.  In addition to
    having all the basic DimArray properties, it keeps track of the
    time dimension and its sample rate.  It also provides methods for
    manipulating the time dimension, such as resampling and filtering
    the data."""

    _required_attrs = {'dims':np.ndarray,
                       'tdim':str,
                       'samplerate':int}
    taxis = property(lambda self:
                     self.get_axis(self.tdim),
                     doc="Numeric time axis (read only).")

    def __new__(cls, data, dims, tdim, samplerate,
                dtype=None, copy=False, **kwargs):
        """
        Needs docstring.
        """
        # set the kwargs to have tdim, samplerate
        kwargs['tdim'] = tdim
        kwargs['samplerate'] = samplerate
        # make new DimArray with timeseries attributes
        ts = DimArray(data, dims, dtype=dtype, copy=copy, **kwargs)
        # convert to TimeSeries and return:
        return ts.view(cls)

    def remove_buffer(self, duration):
	"""Remove the desired buffer duration (in seconds) and reset
	the time range.  You can provide a tuple for the duration
	argument if you would like to remove different numbers of
	samples from the beginning and end of the time series."""
	# see if we need to remove anything
        duration = np.atleast_1d(duration)
        if len(duration) != 2:
            duration = duration.repeat(2)
        num_samp = np.round(float(self.samplerate) * duration)
	if np.any(num_samp>0):
            # remove the buf from the data
            return self.take(range(int(num_samp[0]),
                                   self.shape[self.taxis]-int(num_samp[1])),
                             self.taxis)

    def filtered(self,freq_range,filt_type='stop',order=4):
        """
        Filter the data using a Butterworth filter and return a new
        TimeSeries instance.
        """
        filtered_array = filt.buttfilt(self,freq_range,self.samplerate,filt_type,
                                       order,axis=self.taxis)
        attrs = self._attrs.copy()
        for k in self._required_attrs.keys():
            attrs.pop(k,None)
        return TimeSeries(filtered_array,self.dims.copy(),
                          self.tdim, self.samplerate, **attrs)

    def resampled(self,resampled_rate,window=None):
        """
        Resample the data and reset all the time ranges.  Uses the
        resample function from scipy.  This method seems to be more
        accurate than the decimate method.
        """
        # resample the data, getting new time range
        time_range = self[self.tdim]
        new_length = int(np.round(len(time_range)*resampled_rate/float(self.samplerate)))
        newdat,new_time_range = resample(self, new_length, t=time_range,
                                     axis=self.taxis, window=window)

        # set the time dimension
        newdims = self.dims.copy()
        attrs = self.dims[self.taxis]._attrs.copy()
        for k in self.dims[self.taxis]._required_attrs.keys():
            attrs.pop(k,None)
        newdims[self.taxis] = Dim(new_time_range,
                                  self.dims[self.taxis].name,
                                  **attrs)

        attrs = self._attrs.copy()
        for k in self._required_attrs.keys():
            attrs.pop(k,None)
        return TimeSeries(newdat, newdims,
                          self.tdim, resampled_rate, **attrs)

