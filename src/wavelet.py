from numpy import *
from scipy import unwrap
from filter import decimate

def morlet(freq,t,width):
    """ Generate a Morlet wavelet for specified frequncy for times t.

    The wavelet will be normalized so the total energy is 1.  width
    defines the ``width'' of the wavelet in cycles.  A value >= 5 is
    suggested.

    """
    sf = freq/float(width)
    st = 1/(2*pi*sf)
    A = 1/sqrt(st*sqrt(pi))
    y = A*exp(-pow(t,2)/(2*pow(st,2)))*exp(2j*pi*freq*t)
    return y

def phasePow1d(freq,dat,samplerate,width):
    """ Calculate phase and power for a single freq and 1d signal.

    """
    # set the parameters for the wavelet
    dt = 1./samplerate
    sf = freq/float(width)
    st = 1./(2*pi*sf)
    
    # get the morlet wavelet for the proper time range
    t=arange(-3.5*st,3.5*st,dt)
    m = morlet(freq,t,width)

    # convolve the wavelet and the signal
    y = convolve(m,dat,1)
    
    # get the power
    power = pow(abs(y),2)

    # find where the power is zero
    ind = power==0
        
    # normalize the phase estimates to length one
    y[ind] = 1.
    y = y/abs(y)
    y[ind] = 0
        
    # get the phase
    phase = angle(y)

    return phase,power

def tfPhasePow(freqs,dat,samplerate,width=5,downsample=None):
    """ Calculate phase and power over time with a Morlet wavelet.

    You can optionally pass in downsample, which is the samplerate to
    decimate to following the power/phase calculation. """
    # make sure dat is an array
    dat = asarray(dat)

    # allocate for the phase and power
    # appending as a list and turning it into an array saves memory use
    phase = []
    power = []

    # see if shape is 1d, otherwise loop over first dimension calling
    # recursively
    if len(dat.shape) > 1:    
        # has more dimensions, loop over first dimension
        for i in xrange(dat.shape[0]):
            tPhase,tPower = tfPhasePow(freqs,dat[i],samplerate,width)
            phase.append(tPhase)
            power.append(tPower)

    else:
        # hase one dimension
        # calculate the phase and pow for each freq
        for freq in freqs:
            # get the phase and pow
            tPhase,tPower = phasePow1d(freq,dat,samplerate,width)

            # append it to the result
            phase.append(tPhase)
            power.append(tPower)

    # turn into array
    phase = asarray(phase)
    power = asarray(power)        
    
    # see if decimate
    if not downsample is None and downsample != samplerate:
        # set the decimation ratio
        dmate = int(round(samplerate/downsample))

        # must log transform power before decimating
        power[power<=0] = finfo(power.dtype).eps
        power = log10(power)
        power = decimate(power,dmate);
        power = pow(10,power)

        # decimate the unwraped phase, then wrap it back
        phase= mod(decimate(unwrap(phase),dmate)+pi,2*pi)-pi;


    # return the power and phase
    return phase,power
