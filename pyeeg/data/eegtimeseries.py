
from dimdata import DimData
from pyeeg import filter

from scipy.signal import resample
import numpy as N

class EegTimeSeries(DimData):
    """
    Class to hold EEG timeseries data.
    """
    def __init__(self,data,dims,samplerate,unit=None,tdim=-1,buf=0):
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
            self.tdim = self.ndim - 1
        
        # set the buf information
        self.buf = buf

    def copy(self):
        """
        """
        newdata = self.data.copy()
        newdims = self.dims.copy()
        return EegTimeSeries(newdata,newdims,self.samplerate,
                             unit=self.unit,tdim=self.tdim,buf=self.buf)

    def removeBuf(self):
	"""Use the information contained in the time series to remove the
	buf reset the time range.  If buf is 0, no action is
	performed."""
	# see if we need to remove anything
	if self.buf>0:
            # remove the buf from the data
            self.data = self.data.take(range(self.buf,
                                             self.shape[self.tdim]-self.buf),self.tdim)

            # remove the buf from the tdim
            self.dims[self.tdim] = self.dims[self.tdim].select(slice(self.buf,self.shape[self.tdim]-self.buf))

            # reset buf to indicate it was removed
	    self.buf = 0

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

        # set the new buf lengths
        self.buf = int(N.round(float(self.buf)*resampledRate/float(self.samplerate)))

        # set the new samplerate
        self.samplerate = resampledRate

        # set the new shape
        self.shape = self.data.shape

