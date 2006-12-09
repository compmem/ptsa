from numpy import *

def morlet(freq,t,width):
    """ Generate a Morlet wavelet for specified frequncy for times t.

    The wavelet will be normalized so the total energy is 1.  width
    defines the ``width'' of the wavelet in cycles.  A value >= 5 is
    suggested.

    """
    sf = freq/width
    st = 1/(2*pi*sf)
    A = 1/sqrt(st*sqrt(pi))
    y = A*exp(-pow(t,2)/(2*pow(st,2)))*exp(2j*pi*freq*t)
    return y

def calcPhasePow(freq,s,sampleRate,width):
    """ Calculate phase and power over time with a Morlet wavelet.

    """
    dt = 1./Fs
    sf = freq/width
    st = 1./(2*pi*sf)

    # get the morlet wavelet for the proper time range
    t=arange(-3.5*st,3.5*st,dt)
    m = morlet(freq,t,width)

    # convolve the wavelet and the signal
    y = convolve(m,s,1)
    
    # get the power
    power = pow(abs(y),2)

    # get the phase
    #phase = angle(

    # return the power and phase
    return power
