import numpy as N
from scipy import unwrap
import sys

from filter import decimate
from helper import reshapeTo2D,reshapeFrom2D
from pyeeg.data import EegTimeSeries,Dim,Dims,DimData

import pdb

def morlet(freq,t,width):
    """ Generate a Morlet wavelet for specified frequncy for times t.

    The wavelet will be normalized so the total energy is 1.  width
    defines the ``width'' of the wavelet in cycles.  A value >= 5 is
    suggested.

    """
    sf = float(freq)/float(width)
    st = 1./(2*N.pi*sf)
    A = 1./N.sqrt(st*N.sqrt(N.pi))
    y = A*N.exp(-N.power(t,2)/(2*N.power(st,2)))*N.exp(2j*N.pi*freq*t)
    return y

def phasePow1d(freq,dat,samplerate,width):
    """ Calculate phase and power for a single freq and 1d signal.

    """
    # set the parameters for the wavelet
    dt = 1./float(samplerate)
    sf = float(freq)/float(width)
    st = 1./(2*N.pi*sf)
    
    # get the morlet wavelet for the proper time range
    t=N.arange(-3.5*st,3.5*st,dt)
    m = morlet(freq,t,width)

    # make sure we are not trying to get a too low a freq
    # for now it is up to them
    #if len(t) > len(dat):
	#raise

    # convolve the wavelet and the signal
    y = N.convolve(m,dat,'full')

    # cut off the extra
    y = y[N.ceil(len(m)/2.)-1:len(y)-N.floor(len(m)/2.)];

    # get the power
    power = N.power(N.abs(y),2)

    # find where the power is zero
    ind = power==0
        
    # normalize the phase estimates to length one
    y[ind] = 1.
    y = y/N.abs(y)
    y[ind] = 0
        
    # get the phase
    phase = N.angle(y)

    return phase,power

def phasePow2d(freq,dat,samplerate,width):
    """ Calculate phase and power for a single freq and 2d signal of shape
    (events,time).

    This will be slightly faster than phasePow1d for multiple events
    because it only calculates the Morlet wavelet once.  """
    # set the parameters for the wavelet
    dt = 1./float(samplerate)
    sf = float(freq)/float(width)
    st = 1./(2*N.pi*sf)
    
    # get the morlet wavelet for the proper time range
    t=N.arange(-3.5*st,3.5*st,dt)
    m = morlet(freq,t,width)

    # make sure is array
    dat = N.asarray(dat)

    # allocate for the necessary space
    wCoef = N.empty(dat.shape,N.complex64)

    for ev,evDat in enumerate(dat):
	# convolve the wavelet and the signal
	y = N.convolve(m,evDat,'full')

	# cut off the extra
	y = y[N.ceil(len(m)/2.)-1:len(y)-N.floor(len(m)/2.)];

	# insert the data
	wCoef[ev] = y

    # get the power
    power = N.power(N.abs(wCoef),2)

    # find where the power is zero
    ind = power==0
        
    # normalize the phase estimates to length one
    wCoef[ind] = 1.
    wCoef = wCoef/N.abs(wCoef)
    wCoef[ind] = 0
        
    # get the phase
    phase = N.angle(wCoef)

    return phase,power

def tsPhasePow(freqs,tseries,width=5,resample=None,keepBuffer=False,
               verbose=False,toReturn='both',freqDimName='freq'):
    """
    Calculate phase and/or power on an EegTimeSeries, returning new
    EegTimeSeries instances.
    """
    if (toReturn != 'both') and (toReturn != 'pow') and (toReturn != 'phase'):
        raise ValueError("toReturn must be \'pow\', \'phase\', or \'both\' to\
        specify whether power, phase, or both should be  returned. Invalid\
        value for toReturn: %s " % toReturn)
    
    # first get the phase and power as desired
    res = calcPhasePow(freqs,tseries.data,tseries.samplerate,axis=tseries.tdim,
                       width=width,verbose=verbose,toReturn=toReturn)

    # handle the dims
    tsdims = tseries.dims.copy()

    # add in frequency dimension
    freqDim = Dim(freqDimName,freqs,'Hz')
    tsdims.insert(0,freqDim)
    
    # turn them into timeseries
    if toReturn == 'pow' or toReturn == 'both':
        # turn into a timeseries
        powerAll = EegTimeSeries(res,tsdims,
                                 tseries.samplerate,unit='XXX get pow unit',
                                 tdim=-1,buffer=tseries.buffer)
        powerAll.data[powerAll.data<=0] = N.finfo(powerAll.data.dtype).eps
        # see if resample
        if resample:
            # must take log before the resample
            powerAll.data = N.log10(powerAll.data)
            powerAll.resample(resample)
            powerAll.data = N.power(10,powerAll.data)
        # see if remove buffer
        if not keepBuffer:
            powerAll.removeBuffer()
    
    if toReturn == 'phase' or toReturn == 'both':
        # get the phase matrix
        phaseAll = EegTimeSeries(res,tsdims,
                                 tseries.samplerate,unit='radians',
                                 tdim=-1,buffer=tseries.buffer)
        if resample:
            # must unwrap before resampling
            phaseAll.data = N.unwrap(phaseAll.data)
            phaseAll.resample(resample)
            phaseAll.data = N.mod(phaseAll.data+N.pi,2*N.pi)-N.pi;            
        # see if remove buffer
        if not keepBuffer:
            phaseAll.removeBuffer()
    
    # see what to return
    if toReturn == 'pow':
        return powerAll
    elif toReturn == 'phase':
        return phaseAll
    elif toReturn == 'both':
        return phaseAll,powerAll
        
    

def calcPhasePow(freqs,dat,samplerate,axis=-1,width=5,verbose=False,toReturn='both'):
    """Calculate phase and power over time with a Morlet wavelet.

    You can optionally pass in downsample, which is the samplerate to
    decimate to following the power/phase calculation. 

    As always, it is best to pass in extra signal (a buffer) on either
    side of the signal of interest because power calculations and
    decimation have edge effects."""

    if toReturn != 'both' and toReturn != 'pow' and toReturn != 'phase':
        raise ValueError("toReturn must be \'pow\', \'phase\', or \'both\' to specify whether power, phase, or both are returned. Invalid value: %s " % toReturn)
    
    # reshape the data to 2D with time on the 2nd dimension
    origshape = dat.shape
    eegdat = reshapeTo2D(dat,axis)

    # allocate
    phaseAll = []
    powerAll = []

    # loop over freqs
    freqs = N.asarray(freqs)
    if len(freqs.shape)==0:
	freqs = N.array([freqs])
    if verbose:
	sys.stdout.write('Calculating wavelet phase/power...\n')
	sys.stdout.write('Freqs (%g to %g): ' % (N.min(freqs),N.max(freqs)))
    for f,freq in enumerate(freqs):
	if verbose:
	    sys.stdout.write('%g ' % (freq))
	    sys.stdout.flush()
	# get the phase and power for that freq
	phase,power = phasePow2d(freq,eegdat,samplerate,width)
        
        # reshape back do original data shape
	if toReturn == 'phase' or toReturn == 'both':
	    phase = reshapeFrom2D(phase,axis,origshape)
	if toReturn == 'pow' or toReturn == 'both':
	    power = reshapeFrom2D(power,axis,origshape)

	# see if allocate
	if len(phaseAll) == 0 and len(powerAll) == 0:
	    if toReturn == 'phase' or toReturn == 'both':
		phaseAll = N.empty(N.concatenate(([len(freqs)],phase.shape)),
				   dtype=phase.dtype)
	    if toReturn == 'pow' or toReturn == 'both':
		powerAll = N.empty(N.concatenate(([len(freqs)],power.shape)),
				   dtype=power.dtype)
        # insert into all
	if toReturn == 'phase' or toReturn == 'both':
	    phaseAll[f] = phase
	if toReturn == 'pow' or toReturn == 'both':
	    powerAll[f] = power

    if verbose:
	sys.stdout.write('\n')

    if toReturn == 'pow':
        return powerAll
    elif toReturn == 'phase':
        return phaseAll
    elif toReturn == 'both':
        return phaseAll,powerAll



#     # convert negative axis to positive axis
#     rnk = len(origshape)
    
#     # set time axis, used for decimation
#     taxis = axis
#     if taxis < 0: 
# 	taxis = taxis + rnk
	
#     # add one b/c of new freq dim at beginning
#     taxis = taxis + 1

#     # see if decimate
#     samplerate = dat.samplerate
#     timeRange = dat.time
#     buffer = dat.bufLen
#     if not downsample is None and downsample != samplerate:
# 	if verbose:
# 	    sys.stdout.write('Decimating...')
# 	    sys.stdout.flush()
#         # set the decimation ratio
#         dmate = int(N.round(samplerate/downsample))

# 	if not phaseOnly:
# 	    # must log transform powerAll before decimating
# 	    powerAll[powerAll<=0] = N.finfo(powerAll.dtype).eps
# 	    powerAll = N.log10(powerAll)
# 	    powerAll = decimate(powerAll,dmate,axis=taxis);
# 	    powerAll = N.power(10,powerAll)

# 	if not powOnly:
# 	    # decimate the unwraped phase, then wrap it back
# 	    phaseAll = N.mod(decimate(N.unwrap(phaseAll),dmate)+N.pi,2*N.pi)-N.pi;

# 	# redo the time and reset the samplerate
# 	samplerate = downsample
# 	if dat.bufLen > 0:
# 	    # redo using the buffer
# 	    timeRange = N.linspace(dat.OffsetMS-dat.BufferMS,
# 				   dat.OffsetMS+dat.DurationMS+dat.BufferMS,
# 				   phaseAll.shape[taxis])
# 	    # reset the buffer
# 	    buffer = int(N.fix((dat.BufferMS)*samplerate/1000.))
# 	else:
# 	    # redo with no buffer
# 	    timeRange = N.linspace(dat.OffsetMS,
# 				   dat.OffsetMS+dat.DurationMS,
# 				   phaseAll.shape[taxis])
# 	if verbose:
# 	    sys.stdout.write('Done!\n')
# 	    sys.stdout.flush()

#     # make dictinary of results
#     res = {'freqs': freqs,
# 	   'width': width,
# 	   'samplerate': samplerate,
# 	   'time': timeRange,
# 	   'OffsetMS': dat.OffsetMS,
# 	   'DurationMS': dat.DurationMS,
# 	   'BufferMS': dat.BufferMS,
# 	   'bufLen': buffer}
#     if not powOnly:
# 	res['phase'] = phaseAll
#     if not phaseOnly:
# 	res['power'] = powerAll
	   
#     #res = DataDict(res) 
#     # XXX Replace with EegTimeSeries XXX

#     # see if remove the buffer
#     if not keepBuffer:
# 	toRemove = []
# 	if not powOnly:
# 	    toRemove.append('phase')
# 	if not phaseOnly:
# 	    toRemove.append('power')
# 	res.removeBuffer(toRemove,axis=taxis)
    
#     # return the results
#     return res

def tsZtransPow(freqs,tseries,zTrans=True,width=5,resample=None,keepBuffer=False,
                verbose=False,toReturn='both',freqDimName='freq'):
    """
    Calculate z-transformed power (and optionally phase) on an
    EegTimeSeries, returning new EegTimeSeries instances.
    """
    if (toReturn != 'both') and (toReturn != 'pow'):
        raise ValueError("toReturn must be \'pow\'or \'both\' to specify\
        whether power only, or power and phase are returned. Only power is\
        z-tranformed; if only phase and/or untransformed power is of interest,\
        the function tsPhasePow() should be called directly. Invalid value for\
        toReturn: %s" % toReturn)

    # Get the power (and optionally phase) for tseries:
    if toReturn == 'both':
        phaseAll,powerAll = tsPhasePow(freqs=freqs,tseries=tseries,width=width,
                                       resample=resample,keepBuffer=keepBuffer,
                                       verbose=verbose,toReturn=toReturn,
                                       freqDimName=freqDimName)
    else:
        powerAll = tsPhasePow(freqs=freqs,tseries=tseries,width=width,
                              resample=resample,keepBuffer=keepBuffer,
                              verbose=verbose,toReturn=toReturn,
                              freqDimName=freqDimName)

    # Ensure power is positive and log10 transform:
    powerAll.data[powerAll.data<=0] = N.finfo(powerAll.data.dtype).eps
    powerAll.data = N.log10(powerAll.data)

    # Get zmean and zstd (DimData objects with a frequency dimension each):
    if isinstance(zTrans,tuple): # zmean and zstd are passed as zTrans
        if ((len(zTrans) != 2) or (not isinstance(zTrans[0],DimData)) or
            (not isinstance(zTrans[1],DimData)) or (zTrans[0].ndim!=1) or
            (zTrans[1].ndim!=1) or (zTrans[0].dims.names[0]!=freqDimName) or
            (zTrans[1].dims.names[0]!=freqDimName) or
            (zTrans[0][freqDimName].data!=powerAll[freqDimName].data) or 
            (zTrans[1][freqDimName]!=powerAll[freqDimName].data)):
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
        if isinstance(zTrans,EegTimeSeries):
            # Get the power for the provided baseline time series:
            zpow = tsPhasePow(freqs=freqs,tseries=zTrans,width=width,
                              resample=resample,keepBuffer=False,verbose=verbose,
                              toReturn='pow',freqDimName=freqDimName)
            zpow.data[zpow.data<=0] = N.finfo(zpow.data.dtype).eps
            zpow.data = N.log10(zpow.data)
        else:
            # Copy the power for the entire time series:
            zpow = powerAll.copy()
            zpow.removeBuffer()
        # Now calculate zmean and zstd from zpow:
        zmean = zpow.margin(freqDimName,N.mean,unit="mean log10 power")
        zstd = zpow.margin(freqDimName,N.std,unit="std of log10 power")

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
        
        
