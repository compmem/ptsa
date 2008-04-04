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

from ptsa.filt import decimate
from ptsa.helper import reshapeTo2D,reshapeFrom2D,nextPow2,centered
from ptsa.data import TimeSeries,Dim,Dims,DimData
from ptsa.fixed_scipy import morlet as morlet_wavelet

def morlet_multi(freqs, widths, samplerates,
                 sampling_windows=7, complete=True):
    """
    Calculate Morlet wavelets with the total energy normalized to 1.
    
    Calls the scipy.signal.wavelet.morlet() function to generate
    Morlet wavelets with the specified frequencies, samplerates, and
    widths (in cycles); see the docstring for the scipy morlet function
    for details. These wavelets are normalized before they are returned.
    
    Parameters
    ----------
    freqs : {float, array_like of floats}
        The frequencies of the Morlet wavelets.
    widths : {float, array_like floats}
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
    samplerates : {float, array_like floats}
        The sample rate(s) of the signal (e.g., 200 Hz).
    sampling_windows : {float, array_like of floates},optional
        How much of the wavelets is sampled. As sampling_window increases,
        the number of samples increases and thus the samples near the edge
        approach zero increasingly closely. If desired different values can
        be specified for different wavelets (the syntax for multiple sampling
        windows is the same as for widths). One value >= 7 is recommended.
    complete : {bool},optional
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
    samplerates = N.atleast_1d(samplerates)
    sampling_windows = N.atleast_1d(sampling_windows)

    # check input:
    if len(freqs) < 1:
        raise ValueError("At least one frequency must be specified!")
    if len(widths) < 1 or len(freqs)%len(widths) != 0:
        raise ValueError("Freqs and widths are not compatible: len(freqs) must "+
                         "be evenly divisible by len(widths).\n"+
                         "len(freqs) = "+str(len(freqs))+"\nlen(widths) = "+
                         str(len(widths)))
    if len(samplerates) < 1 or len(freqs)%len(samplerates) != 0:
        raise ValueError("Freqs and samplerates are not compatible:"+
                         "len(freqs) must be evenly divisible by"+
                         "len(samplerates).\nlen(freqs) = "+str(len(freqs))+
                         "\nlen(samplerates) = "+str(len(samplerates)))
    if len(sampling_windows) < 1 or len(freqs)%len(sampling_windows) != 0:
        raise ValueError("Freqs and sampling_windows are not compatible:"+
                         "len(freqs) must be evenly divisible by"+
                         "len(sampling_windows).\nlen(freqs) = "+str(len(freqs))+
                         "\nlen(sampling_windows) = "+str(len(sampling_windows)))
     
    
    # make len(widths)==len(freqs):
    widths = widths.repeat(len(freqs)/len(widths))
    
    # make len(samplerates)==len(freqs):
    samplerates = samplerates.repeat(len(freqs)/len(samplerates))

    # make len(sampling_windows)==len(freqs):
    sampling_windows = sampling_windows.repeat(len(freqs)/len(sampling_windows))
   
    # std. devs. in the time domain:
    st = widths/(2*N.pi*freqs)
    
    # determine number of samples needed:
    samples = N.ceil(st*samplerates*sampling_windows)
    
    # each scale depends on frequency, samples, width, and samplerates:
    scales = (freqs*samples)/(2.*widths*samplerates)
    
    # generate list of unnormalized wavelets:
    wavelets = [morlet_wavelet(samples[i],w=widths[i],s=scales[i],
                               complete=complete)
                for i in xrange(len(scales))]
    
    # generate list of energies for the wavelets:
    energies = [N.sqrt(N.sum(N.power(N.abs(wavelets[i]),2.))/samplerates[i])
                for i in xrange(len(scales))]
    
    # normalize the wavelets by dividing each one by its energy:
    norm_wavelets = [wavelets[i]/energies[i]
                     for i in xrange(len(scales))]
    
    return norm_wavelets


def phase_pow_multi(freqs, dat, samplerates, widths=5, toReturn='both',
                    time_axis=-1, freq_axis=0, conv_dtype=N.complex64, **kwargs):
    """
    Calculate phase and power with wavelets across multiple events.

    Calls the morlet_multi() and fconv_multi() functions to convolve
    dat with Morlet wavelets.  Phase and power over time across all
    events are calculated from the results. Time/samples should
    include a buffer before onsets and after offsets of the events of
    interest to avoid edge effects.

    Parameters
    ----------
    freqs : {int, float, array_like of ints or floats}
        The frequencies of the Morlet wavelets.
    dat : {array_like}
        The data to determine the phase and power of. Time/samples must be
        last dimension and should include a buffer to avoid edge effects.
    samplerates : {float, array_like of floats}
        The sample rate(s) of the signal (e.g., 200 Hz).
    widths : {float, array_like of floats}
        The width(s) of the wavelets in cycles. See docstring of
        morlet_multi() for details.
    toReturn : {'both','power','phase'}, optional
        Specify whether to return power, phase, or both.
    time_axis : {int},optional
        Index of the time/samples dimension in dat.
        Should be in {-1,0,len(dat.shape)}
    freq_axis : {int},optional
        Index of the frequency dimension in the returned array(s).
        Should be in {0, time_axis, time_axis+1,len(dat.shape)}.
    conv_dtype : {numpy.complex*},optional
        Data type for the convolution array. Using a larger dtype
        (e.g., numpy.complex128) can increase processing time.
        This value influences the dtype of the output array. In case of
        numpy.complex64 the dtype of the output array is numpy.float32.
        Higher complex dtypes produce higher float dtypes in the output.
    **kwargs : {**kwargs},optional
        Additional key word arguments to be passed on to morlet_multi().
    
    Returns
    -------
    Array(s) of phase and/or power values as specified in toReturn. The
    returned array(s) has/have one more dimension than dat. The added
    dimension is for the frequencies and is inserted at freq_axis.
    """

    # ensure proper dimensionality (needed for len call later):
    freqs = N.atleast_1d(freqs)
    
    # check input values:
    if toReturn != 'both' and toReturn != 'power' and toReturn != 'phase':
        raise ValueError("toReturn must be \'power\', \'phase\', or \'both\' to "+
                         "specify whether power, phase, or both are to be "+
                         "returned. Invalid value: %s " % toReturn)

    if not N.issubdtype(conv_dtype,N.complex):
        raise ValueError("conv_dtype must be a complex data type!\n"+
                         "Invalid value: "+str(conv_dtype))

    # generate list of wavelets:
    wavelets = morlet_multi(freqs,widths,samplerates,**kwargs)
        
    # make sure we have at least as many data samples as wavelet samples
    if (N.max([len(i) for i in wavelets]) >  dat.shape[time_axis]):
        raise ValueError("The number of data samples is insufficient compared "+
                         "to the number of wavelet samples. Try increasing "+
                         "data samples by using a (longer) buffer.\n data "+
                         "samples: "+str(dat.shape[time_axis])+"\nmax wavelet "+
                         "samples: "+str(N.max([len(i) for i in wavelets])))
    
    # reshape the data to 2D with time on the 2nd dimension
    origshape = dat.shape
    eegdat = reshapeTo2D(dat,time_axis)

    # for efficiency pre-generate empty array for convolution:
    wavCoef = N.empty((eegdat.shape[time_axis-1]*len(freqs),
                       eegdat.shape[time_axis]),dtype=conv_dtype)
    
    # populate this array with the convolutions:
    i=0
    for wav in wavelets:
        for evDat in dat:
            wavCoef[i]=N.convolve(wav,evDat,'same')
            i+=1
    
    # Determine shape for ouput arrays with added frequency dimension:
    newshape = list(origshape)
    # freqs must be first for reshapeFrom2D to work
    newshape.insert(freq_axis,len(freqs))
    newshape = tuple(newshape)
    
    if toReturn == 'power' or toReturn == 'both':
        # calculate power:
        power = N.power(N.abs(wavCoef),2)
        # reshape to new shape:
        power = reshapeFrom2D(power,time_axis,newshape)
    
    if toReturn == 'phase' or toReturn == 'both':
        # normalize the phase estimates to length one taking care of
        # instances where they are zero:
        norm_factor = N.abs(wavCoef)
        ind = norm_factor == 0
        norm_factor[ind] = 1.
        wavCoef = wavCoef/norm_factor
        wavCoef[ind] = 0
        # calculate phase:
        phase = N.angle(wavCoef)
        # reshape to new shape
        phase = reshapeFrom2D(phase,time_axis,newshape)
    
    if toReturn == 'power':
        return power
    elif toReturn == 'phase':
        return phase
    elif toReturn == 'both':
        return phase,power



##################
# Old wavelet code
##################

def morlet(freq,t,width):
    """Generate a Morlet wavelet for specified frequncy for times t.
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
    #wCoef = N.empty(dat.shape,N.complex192)

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


