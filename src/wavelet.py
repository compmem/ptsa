from numpy import *
from scipy import unwrap
from filter import decimate
from helper import reshapeTo2D,reshapeFrom2D

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
    y = convolve(m,dat,'same')
    
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



def tfPhasePow(freqs,dat,samplerate,axis=-1,width=5,downsample=None):
    """Calculate phase and power over time with a Morlet wavelet.

    You can optionally pass in downsample, which is the samplerate to
    decimate to following the power/phase calculation. 

    As always, it is best to pass in extra signal (a buffer) on either
    side of the signal of interest because power calculations and
    decimation have edge effects."""
    # make sure dat is an array
    dat = asarray(dat)

    # reshape the data to 2D with time on the 2nd dimension
    origshape = dat.shape
    dat = reshapeTo2D(dat,axis)

    # allocate
    phaseAll = []
    powerAll = []

    # loop over freqs
    for freq in freqs:
        # allocate for data
        phase = empty(dat.shape,single)
        power = empty(dat.shape,single)
    
        # loop over chunks of time
        for i in xrange(dat.shape[0]):
            # get the phase and pow
            phase[i],power[i] = phasePow1d(freq,dat[i],samplerate,width)
        
        # reshape back do original data shape
        phase = reshapeFrom2D(phase,axis,origshape)
        power = reshapeFrom2D(power,axis,origshape)

        # append to all
        phaseAll.append(phase)
        powerAll.append(power)

    # turn into array
    phaseAll = asarray(phaseAll)
    powerAll = asarray(powerAll)        

    # see if decimate
    if not downsample is None and downsample != samplerate:
        # convert negative axis to positive axis
        rnk = len(origshape)

        # set time axis, used for decimation
        taxis = axis
        if taxis < 0: 
            taxis = taxis + rnk

        # add one b/c of new freq dim at beginning
        taxis = taxis + 1

        # set the decimation ratio
        dmate = int(round(samplerate/downsample))

        # must log transform powerAll before decimating
        powerAll[powerAll<=0] = finfo(powerAll.dtype).eps
        powerAll = log10(powerAll)
        powerAll = decimate(powerAll,dmate,axis=taxis);
        powerAll = pow(10,powerAll)

        # decimate the unwraped phase, then wrap it back
        phaseAll = mod(decimate(unwrap(phaseAll),dmate)+pi,2*pi)-pi;


    # return the power and phase
    return phaseAll,powerAll
