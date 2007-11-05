import numpy as N
from scipy import unwrap
import sys

from filter import decimate
from helper import reshapeTo2D,reshapeFrom2D
from data import EegTimeSeries,Dim,Dims

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
               verbose=False,phaseOnly=False,powOnly=False):
    """
    Calculate phase and/or power on an EegTimeSeries, returning new
    EegTimeSeries instances.
    """
    # first get the phase and power as desired
    res = calcPhasePow(freqs,tseries.data,axis=tseries.tdim,width=width,
                       verbose=verbose,phaseOnly=phaseOnly,powOnly=powOnly)

    # handle the dims
    tsdims = tseries.dims.copy()

    # add in frequency dimension
    freqDim = Dim('freq',freqs,'Hz')
    tsdims.insert(0,freqDim)
    
    # turn them into timeseries
    if not phaseOnly:
        # turn into a timeseries
        powerAll = EegTimeSeries(res,tsdims,
                                 tseries.samplerate,unit='XXX get pow unit',
                                 tdim=-1,buffer=tseries.buffer)
        # see if resample
        if resample:
            # must take log before the resample
            powerAll.data[powerAll.data<=0] = N.finfo.powerAll.data.dtype).eps
            powerAll.data = N.log10(powerAll.data)
            powerAll.resample(resample)
            powerAll.data = N.power(10,powerAll.data)
        # see if remove buffer
        if not keepBuffer:
            powerAll.removeBuffer()

    if not powOnly:
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
    if powOnly:
        return powerAll
    elif phaseOnly:
        return phaseAll
    else:
        return phaseAll,powerAll
        
    

def calcPhasePow(freqs,dat,axis=-1,width=5,verbose=False,phaseOnly=False,powOnly=False):
    """Calculate phase and power over time with a Morlet wavelet.

    You can optionally pass in downsample, which is the samplerate to
    decimate to following the power/phase calculation. 

    As always, it is best to pass in extra signal (a buffer) on either
    side of the signal of interest because power calculations and
    decimation have edge effects."""
    
    # reshape the data to 2D with time on the 2nd dimension
    origshape = dat
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
	phase,power = phasePow2d(freq,eegdat,dat.samplerate,width)
        
        # reshape back do original data shape
	if not powOnly:
	    phase = reshapeFrom2D(phase,axis,origshape)
	if not phaseOnly:
	    power = reshapeFrom2D(power,axis,origshape)

	# see if allocate
	if len(phaseAll) == 0:
	    if not powOnly:
		phaseAll = N.empty(N.concatenate(([len(freqs)],phase.shape)),
				   dtype=phase.dtype)
	    if not phaseOnly:
		powerAll = N.empty(N.concatenate(([len(freqs)],power.shape)),
				   dtype=power.dtype)
        # insert into all
	if not powOnly:
	    phaseAll[f] = phase
	if not phaseOnly:
	    powerAll[f] = power

    if verbose:
	sys.stdout.write('\n')

    if powOnly:
        return powerAll
    elif phaseOnly:
        return phaseAll
    else:
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

