#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import sys
import numpy as N
from scipy import unwrap
import scipy.stats as stats
from scipy.fftpack import fft,ifft
from scipy.signal.signaltools import _centered as centered

from ptsa.filt import decimate
from ptsa.helper import reshapeTo2D,reshapeFrom2D,nextPow2
from ptsa.data import TimeSeries,Dim,Dims,DimData
from ptsa.fixed_scipy import morlet

def morlet_multi(freqs, widths, samplerate,
                 sampling_window=7, complete=True):
    """
    Calculate Morlet wavelets with the total energy normalized to 1.
    
    Calls the scipy.signal.wavelet.morlet() function to generate
    Morlet wavelets with the specified frequencies, samplerate, and
    widths (in cycles); see the docstring for the scipy morlet function
    for details. These wavelets are normalized before they are returned.
    
    Parameters
    ----------
    freqs : {int, float, array_like of ints or floats}
        The frequencies of the Morlet wavelets
    widths : {int, float, array_like of ints or floats}
        The width(s) of the wavelets in cycles. If only one width is passed
        in, all wavelets have the same width. If len(widths)==len(freqs),
        each frequency is paired with a corresponding width. If
        1<len(widths)<len(freqs), len(freqs) must be evenly divisible by
        len(widths) (i.e., len(freqs)%len(widths)==0). In this case widths
        are repeated such that (1/len(widths))*len(freq) neigboring wavelets
        have the same width -- e.g., if len(widths)==2, the the first and
        second half of the wavelets have widths of widths[0] and width[1]
        respectively, and if len(widths)==3 the first, middle, and last
        third of wavelets have widths of widths[0], widths[1], and widths[2]
        respectively.
    samplerate : {float}
        The sample rate of the signal (e.g., 200 Hz)
    sampling_window : {float}
        How much of the wavelet is sampled. As sampling_window increases,
        the number of samples increases and thus the samples near the edge
        approach zero increasingly closely. The number of samples are
        determined from the wavelet(s) with the largest standard deviation
        in the time domain. All other wavelets are therefore guaranteed to
        approach zero better at the edges. A value >= 7 is recommended.
    complete : {bool}
        Whether to generate a complete or standard approximation to
        the complete version of a Morlet wavelet. Complete should be True,
        especially for low (<=5) values of width. See
        scipy.signal.wavelet.morlet() for details.
    
    Returns
    -------
    A 2-D (frequencies * samples) array of Morlet wavelets.
    
    Notes
    -----
    The in scipy versions <= 0.6.0, the scipy.signal.wavelet.morlet()
    code contains a bug. Until it is fixed in a stable release, this
    code calls a local fixed version of the scipy function.
    
    Examples
    --------
    >>> wavelet = morlet_multi(10,5,200)
    >>> wavelet.shape
    (1, 112)
    >>> wavelet = morlet_multi([10,20,30],5,200)
    >>> wavelet.shape
    (3, 112)
    >>> wavelet = morlet_multi([10,20,30],[5,6,7],200)
    >>> wavelet.shape
    (3, 112)
    """

    # ensure the proper dimensions
    freqs = N.atleast_1d(freqs)
    widths = N.atleast_1d(widths)

    # make len(widths)==len(freqs):
    widths = widths.repeat(len(freqs)/len(widths))
    if len(widths) != len(freqs):
        raise ValueError("Freqs and widths are not compatible: len(freqs) must be "+
                         "evenly divisible by len(widths).\n"+
                         "len(freqs) = "+str(len(freqs))+
                         "\nlen(widths) = "+str(len(widths)))
    
    # std. devs. in the time domain:
    st = widths/(2*N.pi*freqs)
    
    # determine number of samples needed based on wavelet with maximum
    # standard deviation in time domain
    samples = N.ceil(N.max(st)*samplerate*sampling_window)
    
    # determine the scale of the wavelet (cf.
    # scipy.signal.wavelets.morlet docstring):
    scale = (freqs*samples)/(2.*widths*samplerate)
    
    wavelets = N.array([morlet(samples,w=widths[i],s=scale[i],
                               complete=complete)
                        for i in xrange(len(scale))])
    energy = N.sqrt(N.sum(N.power(N.abs(wavelets),2.),axis=1)/samplerate)
    norm_factors = N.vstack([1./energy]*samples).T
    return wavelets*norm_factors


def fconv_multi(in1, in2, mode='full'):
    """Convolve multiple 1-dimensional arrays using FFT. See convolve
    for additional details.

    N.B. The time dimension MUST be the first dimension.  The FFT are
    calculated for each column of each input signals and then are
    combined as if you looped over the 1D signals in the first input
    and for each of those signals you looped over the the 1D signals
    from the second input.  Time then becomes the second axis on the
    return value while each pairwise combination makes up the first
    axis.
    """
    
    # ensure proper number of dimensions
    in1 = N.atleast_2d(in1)
    in2 = N.atleast_2d(in2)

    # get the number of samples for each input
    s1 = in1.shape[0]
    s2 = in2.shape[0]

    # get the number of signals in each input
    num1 = in1.shape[1]
    num2 = in2.shape[1]
    
    # see if we will be returning a complex result
    complex_result = (N.issubdtype(in1.dtype, N.complex) or
                      N.issubdtype(in2.dtype, N.complex))

    # determine the size based on the next power of 2
    actual_size = s1+s2-1
    size = N.power(2,nextPow2(actual_size))

    # perform the fft of each column of in1 and in2
    in1_fft = N.empty((num1,size),dtype=N.complex128)
    for i in xrange(num1):
        in1_fft[i,:] = fft(in1[:,i],size)
    in2_fft = N.empty((num2,size),dtype=N.complex128)
    for i in xrange(num2):
        in2_fft[i,:] = fft(in2[:,i],size)

    # duplicate the signals and multiply before taking the inverse
    ret = ifft(in1_fft.repeat(num2,axis=0) * \
               N.vstack([in2_fft]*num1))

    # delete to save memory
    del in1_fft, in2_fft

    # strip of extra space if necessary
    ret = ret[:,:actual_size]

    # determine if complex, keeping only real if not
    if not complex_result:
        ret = ret.real

    # now only keep the requested portion
    if mode == "full":
        return ret
    elif mode == "same":
        if s1 > s2:
            osize = s1
        else:
            osize = s2
        return centered(ret,osize)
    elif mode == "valid":
        return centered(ret,N.abs(s2-s1)+1)


def phase_pow_multi(freqs, dat, samplerate, axis=-1,
                    widths=5, verbose=False, toReturn='both'):
    # base this code on calcPhasePow from below
    pass


##################
# Old wavelet code
##################

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


