#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as N
from scipy import unwrap
import sys

from filt import decimate
from helper import reshapeTo2D,reshapeFrom2D
from ptsa.data import TimeSeries,Dim,Dims,DimData
import scipy.stats as stats

import pdb

# def morlet(freq,t,width):
#     """ Generate a Morlet wavelet for specified frequncy for times t.

#     The wavelet will be normalized so the total energy is 1.  width
#     defines the ``width'' of the wavelet in cycles.  A value >= 5 is
#     suggested.

#     """
#     sf = float(freq)/float(width)
#     st = 1./(2*N.pi*sf)
#     A = 1./N.sqrt(st*N.sqrt(N.pi))
#     y = A*N.exp(-N.power(t,2)/(2*N.power(st,2)))*N.exp(2j*N.pi*freq*t)
#     return y


def getNormWavelet(freq,samplerate,wavelet=ptsa.fixed_scipy.wavelets.morlet,**kwargs):
    # generate wavelet based on frequency & sample rate
    # normalize wavlet so the total energy is 1
    pass
    


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
    Calculate phase and/or power on an TimeSeries, returning new
    TimeSeries instances.
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
        powerAll = TimeSeries(res,tsdims,
                              tseries.samplerate,unit='XXX get pow unit',
                              tdim=-1,buf_samp=tseries.buf_samp)
        powerAll.data[powerAll.data<=0] = N.finfo(powerAll.data.dtype).eps
        # see if resample
        if resample:
            # must take log before the resample
            powerAll.data = N.log10(powerAll.data)
            powerAll.resample(resample)
            powerAll.data = N.power(10,powerAll.data)
        # see if remove buffer
        if not keepBuffer:
            powerAll.removeBuf()
    
    if toReturn == 'phase' or toReturn == 'both':
        # get the phase matrix
        phaseAll = TimeSeries(res,tsdims,
                              tseries.samplerate,unit='radians',
                              tdim=-1,buf_samp=tseries.buf_samp)
        if resample:
            # must unwrap before resampling
            phaseAll.data = N.unwrap(phaseAll.data)
            phaseAll.resample(resample)
            phaseAll.data = N.mod(phaseAll.data+N.pi,2*N.pi)-N.pi;            
        # see if remove buffer
        if not keepBuffer:
            phaseAll.removeBuf()
    
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


