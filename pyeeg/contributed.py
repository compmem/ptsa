import numpy as N
from scipy import unwrap
import sys

#from filter import decimate
#from helper import reshapeTo2D,reshapeFrom2D
from pyeeg.data import TimeSeries,Dim,Dims,DimData
from pyeeg import wavelet
import scipy.stats as stats



import pdb

def tsZtransPow(freqs,tseries,zTrans=True,log=True,width=5,resample=None,
                keepBuffer=False,verbose=False,toReturn='both',freqDimName='freq'):
    """
    Calculate z-transformed power (and optionally phase) on a
    TimeSeries, returning new TimeSeries instances.
    """
    if (toReturn != 'both') and (toReturn != 'pow'):
        raise ValueError("toReturn must be \'pow\'or \'both\' to specify\
        whether power only, or power and phase are returned. Only power is\
        z-tranformed; if only phase and/or untransformed power is of interest,\
        the function tsPhasePow() should be called directly. Invalid value for\
        toReturn: %s" % toReturn)

    # Get the power (and optionally phase) for tseries:
    if toReturn == 'both':
        phaseAll,powerAll = wavelet.tsPhasePow(freqs=freqs,tseries=tseries,width=width,
                                       resample=resample,keepBuffer=keepBuffer,
                                       verbose=verbose,toReturn=toReturn,
                                       freqDimName=freqDimName)
    else:
        powerAll = wavelet.tsPhasePow(freqs=freqs,tseries=tseries,width=width,
                              resample=resample,keepBuffer=keepBuffer,
                              verbose=verbose,toReturn=toReturn,
                              freqDimName=freqDimName)

    if log: # Ensure power is positive and log10 transform:
        powerAll.data[powerAll.data<=0] = N.finfo(powerAll.data.dtype).eps
        powerAll.data = N.log10(powerAll.data)

    # Get zmean and zstd (DimData objects with a frequency dimension each):
    if isinstance(zTrans,tuple): # zmean and zstd are passed as zTrans
        if ((len(zTrans) != 2) or (not isinstance(zTrans[0],DimData)) or
            (not isinstance(zTrans[1],DimData)) or (zTrans[0].ndim!=1) or
            (zTrans[1].ndim!=1) or (zTrans[0].dims.names[0]!=freqDimName) or
            (zTrans[1].dims.names[0]!=freqDimName) or
            (zTrans[0][freqDimName]!=powerAll[freqDimName]).any() or 
            (zTrans[1][freqDimName]!=powerAll[freqDimName]).any()):
            raise ValueError("The ztrans tuple needs to conform to the\
            following format: (zmean,zstd). Where zmean and zstd are both\
            instances of DimData each with a single frequency dimension.\
            The name of the dimension must be as specified in freqDimName and\
            the same frequency values as those in tseries must be used.\
            Invalid value: %s" % str(zTrans))
        elif zTrans[1].data.min() <= 0:
            raise ValueError("The zstd must be postive: zTrans[1].data.min() =\
            %f" % zTrans[1].data.min())
        zmean = zTrans[0]
        zstd = zTrans[1]
    else: # zmean and zstd must be calculated
        if isinstance(zTrans,TimeSeries):
            # Get the power for the provided baseline time series:
            zpow = wavelet.tsPhasePow(freqs=freqs,tseries=zTrans,width=width,
                              resample=resample,keepBuffer=False,verbose=verbose,
                              toReturn='pow',freqDimName=freqDimName)
            if log:
                zpow.data[zpow.data<=0] = N.finfo(zpow.data.dtype).eps
                zpow.data = N.log10(zpow.data)
        else:
            # Copy the power for the entire time series:
            zpow = powerAll.copy()
            zpow.removeBuffer()
        # Now calculate zmean and zstd from zpow:
        # (using stats.std will calculate the unbiased std)
        if log:
            zmean = zpow.margin(freqDimName,stats.mean,unit="mean log10 power")
            zstd = zpow.margin(freqDimName,stats.std,unit="std of log10 power")
        else:
            zmean = zpow.margin(freqDimName,stats.mean,unit="mean power")
            zstd = zpow.margin(freqDimName,stats.std,unit="std of power")

    # For the transformation {zmean,zstd}.data need to have a compatible shape.
    # Calculate the dimensions with which to reshape (all 1 except for the
    # frequency dimension):
    reshapedims = N.ones(len(powerAll.shape))
    reshapedims[powerAll.dim(freqDimName)] = -1

    # z transform using reshapedims to make the arrays compatible:
    powerAll.data = powerAll.data - zmean.data.reshape(reshapedims)
    powerAll.data = powerAll.data / zstd.data.reshape(reshapedims)
    
    if toReturn == 'both':
        return phaseAll,powerAll,(zmean,zstd)
    else:
        return powerAll,(zmean,zstd)
        
