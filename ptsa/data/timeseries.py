#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from dimarray import Dim,DimArray,AttrArray
from ptsa import filt
from ptsa.helper import next_pow2, pad_to_next_pow2

from scipy.signal import resample
import numpy as np
import sys

try:
    import multiprocessing as mp
    has_mp = True
except ImportError:
    has_mp = False

__docformat__ = 'restructuredtext'


class TimeSeries(DimArray):
    """
    TimeSeries(data, tdim, samplerate, *args, **kwargs)

    A subclass of DimArray to hold timeseries data (i.e. data with a
    time dimension and associated sample rate).  It also provides
    methods for manipulating the time dimension, such as resampling
    and filtering the data.
    
    Parameters
    ----------
    data : {array_like}
        The time series data.
    tdim : {str}
        The name of the time dimension.
    samplerate : {float}
        The sample rate of the time dimension. Constrained to be of
        type float (any passed in value is converted to a float).
    *args : {*args},optional
        Additinal custom attributes
    **kwargs : {**kwargs},optional
        Additional custom keyword attributes.

    Note
    ----
    Useful additional (keyword) attributes include dims, dtype, and
    copy (see DimArray docstring for details).

    See also
    --------
    DimArray

    Examples
    --------
    >>> import numpy as np
    >>> import dimarray as da
    >>> import ptsa.data.timeseries as ts
    >>> observations = da.Dim(['a','b','c'],'obs')
    >>> time = da.Dim(np.arange(4),'time')
    >>> data = ts.TimeSeries(np.random.rand(3,4),'time',samplerate=1,
                             dims=[observations,time])
    >>> data
    TimeSeries([[ 0.51244513,  0.39930142,  0.63501339,  0.67071605],
       [ 0.46962664,  0.51071395,  0.46748319,  0.78265951],
       [ 0.85515317,  0.10996395,  0.41642481,  0.50561768]])

    >>> data.samplerate
    1.0
    >>> data.tdim
    'time'
    """

    _required_attrs = {'dims':np.ndarray,
                       'tdim':str,
                       'samplerate':float}
    taxis = property(lambda self:
                     self.get_axis(self.tdim),
                     doc="Numeric time axis (read only).")

    # def __new__(cls, data, dims, tdim, samplerate,
    def __new__(cls, data, tdim, samplerate, *args, **kwargs):
        # make new DimArray with timeseries attributes
        ts = DimArray(data, *args, **kwargs)
        # ensure that tdim is a valid dimension name:
        if not(tdim in ts.dim_names):
            raise ValueError(
                'Provided time dimension name (tdim) is invalid!\n'+
                'Provided value: '+ str(tdim)+'\nAvailable dimensions: '+
                str(ts.dim_names))
        ts.tdim = tdim
        # ensure that sample rate is a float:
        samplerate = float(samplerate)
        # ensure that sample rate is postive:
        if samplerate <= 0:
            raise ValueError(
                'Samplerate must be positive! Provided value: '+
                str(samplerate))
        ts.samplerate = samplerate
        
        # convert to TimeSeries and return:
        return ts.view(cls)
    
    def __setattr__(self, name, value):
        # ensure that tdim is a valid dimension name:
        if name == 'tdim':
            if not(value in self.dim_names):
                raise ValueError(
                    'Provided time dimension name (tdim) is invalid!\n'+
                    'Provided value: '+ str(value)+'\nAvailable dimensions: '+
                    str(self.dim_names))
        # ensure that sample rate is a postive float:
        elif name == 'samplerate':
            value = float(value)
            if value <= 0:
                raise ValueError(
                    'Samplerate must be positive! Provided value: '+
                    str(value))
        DimArray.__setattr__(self,name,value)

    def _ret_func(self, ret, axis):
        """
        Return function output for functions that take an axis
        argument after adjusting dims properly.
        """
        if axis is None:
            # just return what we got
            return ret.view(AttrArray)
        else:
            # see if is time axis
            return_as_dimarray = False
            if self.get_axis(axis) == self.taxis:
                return_as_dimarray = True
            # pop the dim
            ret.dims = ret.dims[np.arange(len(ret.dims))!=axis]
        if return_as_dimarray:
            # The function removed the time dimension, so we return a
            # DimArray instead of a TimeSeries
            return ret.view(DimArray)
        else:
            return ret.view(self.__class__)
    
        
    def remove_buffer(self, duration):
	"""
        Remove the desired buffer duration (in seconds) and reset the
	time range.

        Parameter
        ---------
        duration : {int,float,({int,float},{int,float})}
            The duration to be removed. The units depend on the samplerate:
            E.g., if samplerate is specified in Hz (i.e., samples per second),
            the duration needs to be specified in seconds and if samplerate is
            specified in kHz (i.e., samples per millisecond), the duration needs
            to be specified in milliseconds.
            A single number causes the specified duration to be removed from the
            beginning and end. A 2-tuple can be passed in to specify different
            durations to be removed from the beginning and the end respectively.
            
        Returns
        -------
        ts : {TimeSeries}
            A TimeSeries instance with the requested durations removed from the
            beginning and/or end.
        """
	# see if we need to remove anything
        duration = np.atleast_1d(duration)
        if len(duration) != 2:
            duration = duration.repeat(2)
        num_samp = np.round(self.samplerate * duration)
        # ensure that the number of samples are >= 0:
	if np.any(num_samp<0):
            raise ValueError('Duration must not be negative!'+
                             'Provided values: '+str(duration))
        # remove the buffer from the data
        return self.take(range(int(num_samp[0]),
                               self.shape[self.taxis]-int(num_samp[1])),
                         self.taxis)

    def filtered(self,freq_range,filt_type='stop',order=4):
        """
        Filter the data using a Butterworth filter and return a new
        TimeSeries instance.

        Parameters
        ----------
        freq_range : {array_like}
            The range of frequencies to filter.
        filt_type = {scipy.signal.band_dict.keys()},optional
            Filter type.
        order = {int}
            The order of the filter.

        Returns
        -------
        ts : {TimeSeries}
            A TimeSeries instance with the filtered data.
        """

        filtered_array = filt.buttfilt(np.asarray(self),
                                       freq_range,self.samplerate,filt_type,
                                       order,axis=self.taxis)
        attrs = self._attrs.copy()
        for k in self._required_attrs.keys():
            attrs.pop(k,None)
        return TimeSeries(filtered_array,self.tdim, self.samplerate,
                          dims=self.dims.copy(), **attrs)

    def resampled(self, resampled_rate, window=None,
                  loop_axis=None, num_mp_procs=0, pad_to_pow2=False):
        """
        Resample the data and reset all the time ranges.

        Uses the resample function from scipy.  This method seems to
        be more accurate than the decimate method.

        Parameters
        ----------
        resampled_rate : {float}
            New sample rate to resample to.
        window : {None,str,float,tuple}, optional
            See scipy.signal.resample for details
        loop_axis: {None,str,int}, optional
            Sometimes it might be faster to loop over an axis.
        num_mp_procs: int, optional
            Whether to try and use multiprocessing to loop over axis.
            0 means no multiprocessing
            >0 specifies num procs to use
            None means yes, and use all possible procs
        pad_to_pow2: bool, optional
            Pad along the time dimension to the next power of 2 so
            that the resampling is much faster (experimental).

        Returns
        -------
        ts : {TimeSeries}
            A TimeSeries instance with the resampled data.

        See Also
        --------
        scipy.signal.resample
        """
        # resample the data, getting new time range
        time_range = self[self.tdim]
        new_length = int(np.round(len(time_range)*resampled_rate/self.samplerate))

        if pad_to_pow2:
            padded_length = 2**next_pow2(len(time_range))
            padded_new_length = int(np.round(padded_length*resampled_rate/self.samplerate))
            time_range = np.hstack([time_range, 
                                    (np.arange(1,padded_length-len(time_range)+1)*np.diff(time_range[-2:]))+time_range[-1]])

        if loop_axis is None:
            # just do standard method on all data at once
            if pad_to_pow2:
                newdat,new_time_range = resample(pad_to_next_pow2(np.asarray(self),axis=self.taxis), 
                                                 padded_new_length, t=time_range,
                                                 axis=self.taxis, window=window)
            else:
                newdat,new_time_range = resample(np.asarray(self),
                                                 new_length, t=time_range,
                                                 axis=self.taxis, window=window)

        else:
            # loop over specified axis
            # get the loop axis name and length
            loop_dim = self.get_dim_name(loop_axis)
            loop_dim_len = len(self[loop_dim])
            # specify empty boolean index
            ind = np.zeros(loop_dim_len,dtype=np.bool)
            newdat = []
            if has_mp and num_mp_procs != 0:
                po = mp.Pool(num_mp_procs)

            for i in range(loop_dim_len):
                ind[i] = True
                dat = self.select(**{loop_dim:ind})
                taxis = dat.taxis
                if has_mp and num_mp_procs != 0:
                    # start async proc
                    if pad_to_pow2:
                        dat = pad_to_next_pow2(np.asarray(dat), axis=dat.taxis)
                        newdat.append(po.apply_async(resample,
                                                     (np.asarray(dat), padded_new_length, time_range,
                                                      taxis, window)))
                    else:
                        newdat.append(po.apply_async(resample,
                                                     (np.asarray(dat), new_length, time_range,
                                                      taxis, window)))
                else:
                    # just call on that dataset
                    sys.stdout.write('%d '%i)
                    sys.stdout.flush()
                    if pad_to_pow2:
                        dat = pad_to_next_pow2(np.asarray(dat), axis=dat.taxis)
                        ndat,new_time_range = resample(np.asarray(dat), padded_new_length, t=time_range,
                                                       axis=taxis, window=window)
                    else:
                        ndat,new_time_range = resample(np.asarray(dat), new_length, t=time_range,
                                                       axis=taxis, window=window)
                    newdat.append(ndat)
                ind[i] = False
            if has_mp and num_mp_procs != 0:
                # aggregate mp results
                po.close()
                #po.join()
                out = []
                for i in range(len(newdat)):
                    sys.stdout.write('%d '%i)
                    sys.stdout.flush()
                    out.append(newdat[i].get())
                #out = [newdat[i].get() for i in range(len(newdat))]
                newdat = [out[i][0] for i in range(len(out))]
                new_time_range = out[i][1]

            # concatenate the new data
            newdat = np.concatenate(newdat,axis=self.get_axis(loop_axis))

            sys.stdout.write('\n')
            sys.stdout.flush()

        # remove pad if we padded it
        if pad_to_pow2:
            newdat = newdat.take(range(new_length),axis=self.taxis)
            new_time_range = new_time_range[:new_length]

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
        return TimeSeries(newdat, self.tdim, resampled_rate,
                          dims=newdims, **attrs)

    def baseline_corrected(self, base_range):
        """

        Return a baseline corrected timeseries by subtracting the
        average value in the baseline range from all other time points
        for each dimension.

        Parameters
        ----------
        base_range: {tuple}
            Tuple specifying the start and end time range (inclusive) 
            for the baseline.

        Returns
        -------
        ts : {TimeSeries}
            A TimeSeries instance with the baseline corrected data.

        """
        # get the average of baseline range
        baseline = self['time >= %f'%base_range[0],'time <= %f'%base_range[1]].mean('time')

        # replicate over the time dimension
        baseline = baseline.add_dim(self['time']).transpose(self.dim_names)

        # subtract the baseline
        new_dat = self - baseline

        # return a new timeseries
        attrs = self._attrs.copy()
        for k in self._required_attrs.keys():
            attrs.pop(k,None)
        return TimeSeries(new_dat,self.tdim, self.samplerate,
                          dims=self.dims.copy(), **attrs)
        
