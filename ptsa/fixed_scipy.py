################################################################################
################################################################################
###
### scipy.signal.wavelets.morlet
###
################################################################################
################################################################################

from scipy import linspace, pi, exp, zeros

def morlet(M, w=5.0, s=1.0, complete=True):
    """Complex Morlet wavelet.

    Parameters
    ----------
    M : int
        Length of the wavelet.
    w : float
        Omega0
    s : float
        Scaling factor, windowed from -s*2*pi to +s*2*pi.
    complete : bool
        Whether to use the complete or the standard version.

    Notes:
    ------
    The standard version:
        pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))

        This commonly used wavelet is often referred to simply as the
        Morlet wavelet.  Note that, this simplified version can cause
        admissibility problems at low values of w.

    The complete version:
        pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))

        The complete version of the Morlet wavelet, with a correction
        term to improve admissibility. For w greater than 5, the
        correction term is negligible.

    Note that the energy of the return wavelet is not normalised
    according to s.

    The fundamental frequency of this wavelet in Hz is given
    by f = 2*s*w*r / M where r is the sampling rate.

    """
    x = linspace(-s*2*pi,s*2*pi,M)
    output = exp(1j*w*x)
    
    if complete:
        output -= exp(-0.5*(w**2))
    
    output *= exp(-0.5*(x**2)) * pi**(-0.25)
    
    return output

###
### scipy.signal.wavelets.morlet()
###
################################################################################
################################################################################
