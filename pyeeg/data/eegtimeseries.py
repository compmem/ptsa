
from dimdata import DimData
from pyeeg import filter

from scipy.signal import resample
import numpy as N

class EegTimeSeries(DimData):
    """
    Class to hold EEG timeseries data.
    """
    def __init__(self,data,dims,samplerate,unit=None,tdim=-1,buffer=0):
        """
        """
        # call the base class init
        DimData.__init__(self,data,dims,unit)
        
        # set the timeseries-specific information
        self.samplerate = samplerate
        
        # get the time dimension
        if tdim >= 0:
            # use it
            self.tdim = tdim
        else:
            # turn it into a positive dim
            self.tdim = tdim + self.ndim
        
        # set the buffer information
        self.buffer = buffer

    def copy(self):
        """
        """
        newdata = self.data.copy()
        newdims = self.dims.copy()
        return EegTimeSeries(newdata,newdims,self.samplerate,
                             unit=self.unit,tdim=self.tdim,buffer=self.buffer)

    def removeBuffer(self):
	"""Use the information contained in the time series to remove the
	buffer reset the time range.  If buffer is 0, no action is
	performed."""
	# see if remove the anything
	if self.buffer>0:
            # remove the buffer from the data
            self.data = self.data.take(range(self.buffer,
                                             self.shape[self.tdim]-self.buffer),self.tdim)

            # remove the buffer from the tdim
            self.dims[self.tdim] = self.dims[self.tdim][self.buffer:self.shape[self.tdim]-self.buffer]

            # reset buffer to indicate it was removed
	    self.buffer = 0

            # reset the shape
            self.shape = self.data.shape

    def filter(self,freqRange,filtType='stop',order=4):
        """
        Filter the data using a Butterworth filter.
        """
        self.data = filter.buttfilt(self.data,freqRange,self.samplerate,filtType,
                                    order,axis=self.tdim)

    def resample(self,resampledRate,window=None):
        """
        Resample the data and reset all the time ranges.  Uses the
        resample function from scipy.  This method seems to be more
        accurate than the decimate method.
        """
        # resample the data, getting new time range
        timeRange = self.dims[self.tdim].data
        newLength = int(N.round(self.data.shape[self.tdim]*resampledRate/float(self.samplerate)))
        self.data,newTimeRange = resample(self.data,newLength,t=timeRange,axis=self.tdim,window=window)

#         # resample the tdim
#         # calc the time range in MS
#         timeRange = self.dims[self.tdim].data
#         samplesize = N.abs(timeRange[0]-timeRange[1])
#         newsamplesize = samplesize*self.samplerate/resampledRate
#         adjustment = (newsamplesize - samplesize)/2.
#         sampStart = timeRange[0] + adjustment
#         sampEnd = timeRange[-1] - adjustment
#         newTimeRange = N.linspace(sampStart,sampEnd,newLength)

        # set the time dimension
        self.dims[self.tdim].data = newTimeRange

        # set the new buffer lengths
        self.buffer = int(N.round(float(self.buffer)*resampledRate/float(self.samplerate)))

        # set the new samplerate
        self.samplerate = resampledRate

        # set the new shape
        self.shape = self.data.shape

