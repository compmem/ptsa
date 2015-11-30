#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import sys
import numpy as np
# from scipy import unwrap
# import scipy.stats as stats
from scipy.fftpack import fft,ifft
# import scipy.signal
# import scipy.ndimage
# from ptsa.filt import decimate
from ptsa.helper import reshape_to_2d,reshape_from_2d,centered,next_pow2
from ptsa.data import TimeSeries,Dim
from ptsa.fixed_scipy import morlet as morlet_wavelet

import pywt
import math

# try:
#     import multiprocessing as mp
#     has_mp = True
# except ImportError:
#     has_mp = False


def swt(data, wavelet, level=None):
    """
    Stationary Wavelet Transform

    This version is 2 orders of magnitude faster than the one in pywt
    even though it uses pywt for all the calculations.
    
      Input parameters: 

        data
          One-dimensional data to transform
        wavelet
          Either the name of a wavelet or a Wavelet object
        level
          Number of levels

    """
    if level is None:
        level = pywt.swt_max_level(len(data))
    num_levels = level
    idata = data.copy()
    res = []
    for j in range(1,num_levels+1): 
        step_size = int(math.pow(2, j-1))
        last_index = step_size
        # allocate
        cA = np.empty_like(data)
        cD = np.empty_like(data)
        for first in xrange(last_index): # 0 to last_index - 1
            # Getting the indices that we will transform 
            indices = np.arange(first, len(cD), step_size)

            # select the even indices
            even_indices = indices[0::2] 
            # select the odd indices
            odd_indices = indices[1::2] 
            
            # get the even
            (cA1,cD1) = pywt.dwt(idata[indices], wavelet, 'per')
            cA[even_indices] = cA1
            cD[even_indices] = cD1

            # then the odd
            (cA1,cD1) = pywt.dwt(np.roll(idata[indices],-1), wavelet, 'per')
            cA[odd_indices] = cA1
            cD[odd_indices] = cD1

        # set the data for the next loop
        idata = cA

        # prepend the result
        res.insert(0,(cA,cD))

    return res


def iswt(coefficients, wavelet):
    """
    Inverse Stationary Wavelet Transform
    
      Input parameters: 

        coefficients
          approx and detail coefficients, arranged in level value 
          exactly as output from swt:
          e.g. [(cA1, cD1), (cA2, cD2), ..., (cAn, cDn)]

        wavelet
          Either the name of a wavelet or a Wavelet object

    """
    output = coefficients[0][0].copy() # Avoid modification of input data

    #num_levels, equivalent to the decomposition level, n
    num_levels = len(coefficients)
    for j in range(num_levels,0,-1): 
        step_size = int(math.pow(2, j-1))
        last_index = step_size
        _, cD = coefficients[num_levels - j]
        for first in xrange(last_index): # 0 to last_index - 1

            # Getting the indices that we will transform 
            indices = np.arange(first, len(cD), step_size)

            # select the even indices
            even_indices = indices[0::2] 
            # select the odd indices
            odd_indices = indices[1::2] 

            # perform the inverse dwt on the selected indices,
            # making sure to use periodic boundary conditions
            x1 = pywt.idwt(output[even_indices], cD[even_indices], wavelet, 'per') 
            x2 = pywt.idwt(output[odd_indices], cD[odd_indices], wavelet, 'per') 

            # perform a circular shift right
            x2 = np.roll(x2, 1)

            # average and insert into the correct indices
            output[indices] = (x1 + x2)/2.  

    return output


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
    freqs = np.atleast_1d(freqs)
    widths = np.atleast_1d(widths)
    samplerates = np.atleast_1d(samplerates).astype(np.float64)
    sampling_windows = np.atleast_1d(sampling_windows)

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
    st = widths/(2*np.pi*freqs)
    
    # determine number of samples needed:
    samples = np.ceil(st*samplerates*sampling_windows)
    
    # each scale depends on frequency, samples, width, and samplerates:
    scales = (freqs*samples)/(2.*widths*samplerates)
    
    # generate list of unnormalized wavelets:
    wavelets = [morlet_wavelet(samples[i],w=widths[i],s=scales[i],
                               complete=complete)
                for i in xrange(len(scales))]
    
    # generate list of energies for the wavelets:
    energies = [np.sqrt(np.sum(np.power(np.abs(wavelets[i]),2.))/samplerates[i])
                for i in xrange(len(scales))]
    
    # normalize the wavelets by dividing each one by its energy:
    norm_wavelets = [wavelets[i]/energies[i]
                     for i in xrange(len(scales))]
    
    return norm_wavelets


def convolve_wave(wav,eegdat):
    wave_coef = []
    for ev_dat in eegdat:
        wave_coef.append(np.convolve(wav,ev_dat,'same'))
    return wave_coef


def fconv_multi(in1, in2, mode='full'):
    """
    Convolve multiple 1-dimensional arrays using FFT.

    Calls scipy.signal.fft on every row in in1 and in2, multiplies
    every possible pairwise combination of the transformed rows, and
    returns an inverse fft (by calling scipy.signal.ifft) of the
    result. Therefore the output array has as many rows as the product
    of the number of rows in in1 and in2 (the number of colums depend
    on the mode).
    
    Parameters
    ----------
    in1 : {array_like}
        First input array. Must be arranged such that each row is a
        1-D array with data to convolve.
    in2 : {array_like}
        Second input array. Must be arranged such that each row is a
        1-D array with data to convolve.
    mode : {'full','valid','same'},optional
        Specifies the size of the output. See the docstring for
        scipy.signal.convolve() for details.
    
    Returns
    -------
    Array with in1.shape[0]*in2.shape[0] rows with the convolution of
    the 1-D signals in the rows of in1 and in2.
    """    
    # ensure proper number of dimensions
    in1 = np.atleast_2d(in1)
    in2 = np.atleast_2d(in2)

    # get the number of signals and samples in each input
    num1,s1 = in1.shape
    num2,s2 = in2.shape
    
    # see if we will be returning a complex result
    complex_result = (np.issubdtype(in1.dtype, np.complex) or
                      np.issubdtype(in2.dtype, np.complex))

    # determine the size based on the next power of 2
    actual_size = s1+s2-1
    size = np.power(2,next_pow2(actual_size))

    # perform the fft of each row of in1 and in2:
    #in1_fft = np.empty((num1,size),dtype=np.complex128)
    in1_fft = np.empty((num1,size),dtype=np.complex)
    for i in xrange(num1):
        in1_fft[i] = fft(in1[i],size)
    #in2_fft = np.empty((num2,size),dtype=np.complex128)
    in2_fft = np.empty((num2,size),dtype=np.complex)
    for i in xrange(num2):
        in2_fft[i] = fft(in2[i],size)
    
    # duplicate the signals and multiply before taking the inverse
    in1_fft = in1_fft.repeat(num2,axis=0)
    in1_fft *= np.vstack([in2_fft]*num1)
    ret = ifft(in1_fft)
#     ret = ifft(in1_fft.repeat(num2,axis=0) * \
#                np.vstack([in2_fft]*num1))
    
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
        return centered(ret,(num1*num2,osize))
    elif mode == "valid":
        return centered(ret,(num1*num2,np.abs(s2-s1)+1))



def phase_pow_multi(freqs, dat,  samplerates=None, widths=5,
                    to_return='both', time_axis=-1,
                    conv_dtype=np.complex64, freq_name='freqs',
                    **kwargs):
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
        The data to determine the phase and power of. Sample rate(s)
        and time dimension must be specified as attributes of dat or
        in the key word arguments.  The time dimension should include
        a buffer to avoid edge effects.
    samplerates : {float, array_like of floats}, optional
        The sample rate(s) of the signal. Must be specified if dat is
        not a TimeSeries instance. If dat is a TimeSeries instance,
        any value specified here will be replaced by the value stored
        in the samplerate attribute.
    widths : {float, array_like of floats},optional
        The width(s) of the wavelets in cycles. See docstring of
        morlet_multi() for details.
    to_return : {'both','power','phase'}, optional
        Specify whether to return power, phase, or both.        
    time_axis : {int},optional
        Index of the time/samples dimension in dat. Must be specified
        if dat is not a TimeSeries instance. If dat is a TimeSeries
        instance any value specified here will be replaced by the
        value specified in the tdim attribute.
    conv_dtype : {numpy.complex*},optional
        Data type for the convolution array. Using a larger dtype
        (e.g., numpy.complex128) can increase processing time.
        This value influences the dtype of the output array. In case of
        numpy.complex64 the dtype of the output array is numpy.float32.
        Higher complex dtypes produce higher float dtypes in the output.
    freq_name : {string},optional
        Name of frequency dimension of the returned TimeSeries object
        (only used if dat is a TimeSeries instance).
    **kwargs : {**kwargs},optional
        Additional key word arguments to be passed on to morlet_multi().
    
    Returns
    -------
    Array(s) of phase and/or power values as specified in to_return. The
    returned array(s) has/have one more dimension than dat. The added
    dimension is for the frequencies and is inserted as the first
    dimension.
    """

    dat_is_ts = False # is dat a TimeSeries instance?
    if isinstance(dat,TimeSeries):
        samplerates = dat.samplerate
        time_axis = dat.get_axis(dat.tdim)
        dat_is_ts = True
    elif samplerates is None:
        raise ValueError('Samplerate must be specified unless you provide a TimeSeries!')

    # convert the time_axis to positive index
    if time_axis < 0: 
        time_axis += len(dat.shape)
    
    # ensure proper dimensionality (needed for len call later):
    freqs = np.atleast_1d(freqs)
    
    # check input values:
    if to_return != 'both' and to_return != 'power' and to_return != 'phase':
        raise ValueError("to_return must be \'power\', \'phase\', or \'both\' to "+
                         "specify whether power, phase, or both are to be "+
                         "returned. Invalid value: %s " % to_return)

    if not np.issubdtype(conv_dtype,np.complex):
        raise ValueError("conv_dtype must be a complex data type!\n"+
                         "Invalid value: "+str(conv_dtype))

    # generate list of wavelets:
    wavelets = morlet_multi(freqs,widths,samplerates,**kwargs)
        
    # make sure we have at least as many data samples as wavelet samples
    if (np.max([len(i) for i in wavelets]) >  dat.shape[time_axis]):
        raise ValueError("The number of data samples is insufficient compared "+
                         "to the number of wavelet samples. Try increasing "+
                         "data samples by using a (longer) buffer.\n data "+
                         "samples: "+str(dat.shape[time_axis])+"\nmax wavelet "+
                         "samples: "+str(np.max([len(i) for i in wavelets])))
    
    # reshape the data to 2D with time on the 2nd dimension
    origshape = dat.shape
    eegdat = reshape_to_2d(dat, time_axis) #.view(np.ndarray)

    # for efficiency pre-generate empty array for convolution:
    wav_coef = np.empty((eegdat.shape[0]*len(freqs),
                         eegdat.shape[1]),dtype=conv_dtype)
    
    # populate this array with the convolutions:
    i=0
    step = len(eegdat)
    for wav in wavelets:
        wc = fconv_multi(wav,eegdat,'same')
        wav_coef[i:i+step] = wc
        i+=step
        # for ev_dat in eegdat:
        #     wav_coef[i]=np.convolve(wav,ev_dat,'same')
        #     #wav_coef[i]=scipy.signal.fftconvolve(ev_dat,wav,'same')
        #     i+=1
    
    # Determine shape for ouput arrays with added frequency dimension:
    newshape = list(origshape)
    # freqs must be first for reshape_from_2d to work
    newshape.insert(0,len(freqs))
    newshape = tuple(newshape)
    # must add to the time axis, too
    time_axis += 1
    if dat_is_ts:
        freq_dim = Dim(freqs,freq_name)
        dims_with_freq = np.empty(len(dat.dims)+1,dat.dims.dtype)
        dims_with_freq[0] = freq_dim
        dims_with_freq[1:] = dat.dims[:]
        
    if to_return == 'power' or to_return == 'both':
        # calculate power (wav_coef values are complex, so taking the
        # absolute value is necessary before taking the power):
        power = np.abs(wav_coef)**2
        # reshape to new shape:
        power = reshape_from_2d(power,time_axis,newshape)
        if dat_is_ts:
            power = TimeSeries(power, tdim=dat.tdim,
                               samplerate=dat.samplerate,
                               dims=dims_with_freq)
            
    
    if to_return == 'phase' or to_return == 'both':
        # normalize the phase estimates to length one taking care of
        # instances where they are zero:
        norm_factor = np.abs(wav_coef)
        ind = norm_factor == 0
        norm_factor[ind] = 1.
        wav_coef = wav_coef/norm_factor
        # wav_coef contains complex numbers, so we need to set these
        # to 0 when the absolute value 0.
        wav_coef[ind] = 0
        # calculate phase:
        phase = np.angle(wav_coef)
        # reshape to new shape
        phase = reshape_from_2d(phase,time_axis,newshape)
        if dat_is_ts:
            phase = TimeSeries(phase, tdim=dat.tdim,
                               samplerate=dat.samplerate,
                               dims=dims_with_freq)

    
    if to_return == 'power':
        return power
    elif to_return == 'phase':
        return phase
    elif to_return == 'both':
        return phase,power



##################
# Old wavelet code
##################

def phase_pow_multi_old(freqs, dat, samplerates, widths=5, to_return='both',
                        time_axis=-1, freq_axis=0, conv_dtype=np.complex64, **kwargs):
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
    to_return : {'both','power','phase'}, optional
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
    Array(s) of phase and/or power values as specified in to_return. The
    returned array(s) has/have one more dimension than dat. The added
    dimension is for the frequencies and is inserted at freq_axis.
    """

    # ensure proper dimensionality (needed for len call later):
    freqs = np.atleast_1d(freqs)
    
    # check input values:
    if to_return != 'both' and to_return != 'power' and to_return != 'phase':
        raise ValueError("to_return must be \'power\', \'phase\', or \'both\' to "+
                         "specify whether power, phase, or both are to be "+
                         "returned. Invalid value: %s " % to_return)

    if not np.issubdtype(conv_dtype,np.complex):
        raise ValueError("conv_dtype must be a complex data type!\n"+
                         "Invalid value: "+str(conv_dtype))

    # generate list of wavelets:
    wavelets = morlet_multi(freqs,widths,samplerates,**kwargs)
        
    # make sure we have at least as many data samples as wavelet samples
    if (np.max([len(i) for i in wavelets]) >  dat.shape[time_axis]):
        raise ValueError("The number of data samples is insufficient compared "+
                         "to the number of wavelet samples. Try increasing "+
                         "data samples by using a (longer) buffer.\n data "+
                         "samples: "+str(dat.shape[time_axis])+"\nmax wavelet "+
                         "samples: "+str(np.max([len(i) for i in wavelets])))
    
    # reshape the data to 2D with time on the 2nd dimension
    origshape = dat.shape
    eegdat = reshape_to_2d(dat,time_axis)

    # for efficiency pre-generate empty array for convolution:
    wavCoef = np.empty((eegdat.shape[time_axis-1]*len(freqs),
                       eegdat.shape[time_axis]),dtype=conv_dtype)
    
    # populate this array with the convolutions:
    i=0
    for wav in wavelets:
        for evDat in dat:
            wavCoef[i]=np.convolve(wav,evDat,'same')
            i+=1
    
    # Determine shape for ouput arrays with added frequency dimension:
    newshape = list(origshape)
    # freqs must be first for reshapeFrom2D to work
    newshape.insert(freq_axis,len(freqs))
    newshape = tuple(newshape)
    
    if to_return == 'power' or to_return == 'both':
        # calculate power:
        power = np.power(np.abs(wavCoef),2)
        # reshape to new shape:
        power = reshape_from_2d(power,time_axis,newshape)
    
    if to_return == 'phase' or to_return == 'both':
        # normalize the phase estimates to length one taking care of
        # instances where they are zero:
        norm_factor = np.abs(wavCoef)
        ind = norm_factor == 0
        norm_factor[ind] = 1.
        wavCoef = wavCoef/norm_factor
        wavCoef[ind] = 0
        # calculate phase:
        phase = np.angle(wavCoef)
        # reshape to new shape
        phase = reshape_from_2d(phase,time_axis,newshape)
    
    if to_return == 'power':
        return power
    elif to_return == 'phase':
        return phase
    elif to_return == 'both':
        return phase,power







def morlet(freq,t,width):
    """Generate a Morlet wavelet for specified frequncy for times t.
    The wavelet will be normalized so the total energy is 1.  width
    defines the ``width'' of the wavelet in cycles.  A value >= 5 is
    suggested.
    """
    sf = float(freq)/float(width)
    st = 1./(2*np.pi*sf)
    A = 1./np.sqrt(st*np.sqrt(np.pi))
    y = A*np.exp(-np.power(t,2)/(2*np.power(st,2)))*np.exp(2j*np.pi*freq*t)
    return y


def phasePow1d(freq,dat,samplerate,width):
    """ Calculate phase and power for a single freq and 1d signal.

    """
    # set the parameters for the wavelet
    dt = 1./float(samplerate)
    sf = float(freq)/float(width)
    st = 1./(2*np.pi*sf)
    
    # get the morlet wavelet for the proper time range
    t=np.arange(-3.5*st,3.5*st,dt)
    m = morlet(freq,t,width)

    # make sure we are not trying to get a too low a freq
    # for now it is up to them
    #if len(t) > len(dat):
	#raise

    # convolve the wavelet and the signal
    y = np.convolve(m,dat,'full')

    # cut off the extra
    y = y[np.ceil(len(m)/2.)-1:len(y)-np.floor(len(m)/2.)];

    # get the power
    power = np.power(np.abs(y),2)

    # find where the power is zero
    ind = power==0
        
    # normalize the phase estimates to length one
    y[ind] = 1.
    y = y/np.abs(y)
    y[ind] = 0
        
    # get the phase
    phase = np.angle(y)

    return phase,power

def phasePow2d(freq,dat,samplerate,width):
    """ Calculate phase and power for a single freq and 2d signal of shape
    (events,time).

    This will be slightly faster than phasePow1d for multiple events
    because it only calculates the Morlet wavelet once.  """
    # set the parameters for the wavelet
    dt = 1./float(samplerate)
    sf = float(freq)/float(width)
    st = 1./(2*np.pi*sf)
    
    # get the morlet wavelet for the proper time range
    t=np.arange(-3.5*st,3.5*st,dt)
    m = morlet(freq,t,width)

    # make sure is array
    dat = np.asarray(dat)

    # allocate for the necessary space
    wCoef = np.empty(dat.shape,np.complex64)
    #wCoef = np.empty(dat.shape,np.complex192)

    for ev,evDat in enumerate(dat):
	# convolve the wavelet and the signal
	y = np.convolve(m,evDat,'full')

	# cut off the extra
	y = y[np.ceil(len(m)/2.)-1:len(y)-np.floor(len(m)/2.)];

	# insert the data
	wCoef[ev] = y

    # get the power
    power = np.power(np.abs(wCoef),2)

    # find where the power is zero
    ind = power==0
        
    # normalize the phase estimates to length one
    wCoef[ind] = 1.
    wCoef = wCoef/np.abs(wCoef)
    wCoef[ind] = 0
        
    # get the phase
    phase = np.angle(wCoef)

    return phase,power

def tsPhasePow(freqs,tseries,width=5,resample=None,keepBuffer=False,
               verbose=False,to_return='both',freqDimName='freq'):
    """
    Calculate phase and/or power on an TimeSeries, returning new
    TimeSeries instances.
    """
    if (to_return != 'both') and (to_return != 'pow') and (to_return != 'phase'):
        raise ValueError("to_return must be \'pow\', \'phase\', or \'both\' to\
        specify whether power, phase, or both should be  returned. Invalid\
        value for to_return: %s " % to_return)
    
    # first get the phase and power as desired
    res = calcPhasePow(freqs,tseries.data,tseries.samplerate,axis=tseries.tdim,
                       width=width,verbose=verbose,to_return=to_return)

    # handle the dims
    tsdims = tseries.dims.copy()

    # add in frequency dimension
    freqDim = Dim(freqDimName,freqs,'Hz')
    tsdims.insert(0,freqDim)
    
    # turn them into timeseries
    if to_return == 'pow' or to_return == 'both':
        # turn into a timeseries
        powerAll = TimeSeries(res,tsdims,
                              tseries.samplerate,unit='XXX get pow unit',
                              tdim=-1,buf_samp=tseries.buf_samp)
        powerAll.data[powerAll.data<=0] = np.finfo(powerAll.data.dtype).eps
        # see if resample
        if resample:
            # must take log before the resample
            powerAll.data = np.log10(powerAll.data)
            powerAll.resample(resample)
            powerAll.data = np.power(10,powerAll.data)
        # see if remove buffer
        if not keepBuffer:
            powerAll.removeBuf()
    
    if to_return == 'phase' or to_return == 'both':
        # get the phase matrix
        phaseAll = TimeSeries(res,tsdims,
                              tseries.samplerate,unit='radians',
                              tdim=-1,buf_samp=tseries.buf_samp)
        if resample:
            # must unwrap before resampling
            phaseAll.data = np.unwrap(phaseAll.data)
            phaseAll.resample(resample)
            phaseAll.data = np.mod(phaseAll.data+np.pi,2*np.pi)-np.pi;            
        # see if remove buffer
        if not keepBuffer:
            phaseAll.removeBuf()
    
    # see what to return
    if to_return == 'pow':
        return powerAll
    elif to_return == 'phase':
        return phaseAll
    elif to_return == 'both':
        return phaseAll,powerAll
        
    

def calcPhasePow(freqs,dat,samplerate,axis=-1,width=5,verbose=False,to_return='both'):
    """Calculate phase and power over time with a Morlet wavelet.

    You can optionally pass in downsample, which is the samplerate to
    decimate to following the power/phase calculation. 

    As always, it is best to pass in extra signal (a buffer) on either
    side of the signal of interest because power calculations and
    decimation have edge effects."""

    if to_return != 'both' and to_return != 'pow' and to_return != 'phase':
        raise ValueError("to_return must be \'pow\', \'phase\', or \'both\' to specify whether power, phase, or both are returned. Invalid value: %s " % to_return)
    
    # reshape the data to 2D with time on the 2nd dimension
    origshape = dat.shape
    eegdat = reshape_to_2d(dat,axis)

    # allocate
    phaseAll = []
    powerAll = []

    # loop over freqs
    freqs = np.asarray(freqs)
    if len(freqs.shape)==0:
	freqs = np.array([freqs])
    if verbose:
	sys.stdout.write('Calculating wavelet phase/power...\n')
	sys.stdout.write('Freqs (%g to %g): ' % (np.min(freqs),np.max(freqs)))
    for f,freq in enumerate(freqs):
	if verbose:
	    sys.stdout.write('%g ' % (freq))
	    sys.stdout.flush()
	# get the phase and power for that freq
	phase,power = phasePow2d(freq,eegdat,samplerate,width)
        
        # reshape back do original data shape
	if to_return == 'phase' or to_return == 'both':
	    phase = reshape_from_2d(phase,axis,origshape)
	if to_return == 'pow' or to_return == 'both':
	    power = reshape_from_2d(power,axis,origshape)

	# see if allocate
	if len(phaseAll) == 0 and len(powerAll) == 0:
	    if to_return == 'phase' or to_return == 'both':
		phaseAll = np.empty(np.concatenate(([len(freqs)],phase.shape)),
				   dtype=phase.dtype)
	    if to_return == 'pow' or to_return == 'both':
		powerAll = np.empty(np.concatenate(([len(freqs)],power.shape)),
				   dtype=power.dtype)
        # insert into all
	if to_return == 'phase' or to_return == 'both':
	    phaseAll[f] = phase
	if to_return == 'pow' or to_return == 'both':
	    powerAll[f] = power

    if verbose:
	sys.stdout.write('\n')

    if to_return == 'pow':
        return powerAll
    elif to_return == 'phase':
        return phaseAll
    elif to_return == 'both':
        return phaseAll,powerAll


