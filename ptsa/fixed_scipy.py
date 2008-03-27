################################################################################
################################################################################
###
### scipy.signal.wavelets
###
################################################################################
################################################################################

import numpy as sb
from numpy.dual import eig
from scipy.misc import comb
from scipy import linspace, pi, exp, zeros

################################################################################
# scipy.signal.wavelets.morlet

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

#
################################################################################


################################################################################
################################################################################
###
### scipy.signal.signaltools
###
################################################################################
################################################################################

import types
import scipy.signal.sigtools
from scipy import special, linalg
from scipy.fftpack import fft, ifft, ifftshift, fft2, ifft2
from numpy import polyadd, polymul, polydiv, polysub, \
     roots, poly, polyval, polyder, cast, asarray, isscalar, atleast_1d, \
     ones, sin, linspace, real, extract, real_if_close, zeros, array, arange, \
     where, sqrt, rank, newaxis, argmax, product, cos, pi, exp, \
     ravel, size, less_equal, sum, r_, iscomplexobj, take, \
     argsort, allclose, expand_dims, unique, prod, sort, reshape, c_, \
     transpose, dot, any, minimum, maximum, mean, cosh, arccosh, \
     arccos, concatenate
import numpy
from scipy.fftpack import fftn, ifftn, fft
from scipy.misc import factorial


################################################################################
# scipy.signal.signaltools.fftconvolve

def fftconvolve(in1, in2, mode="full"):
    """Convolve two N-dimensional arrays using FFT. See convolve.
    """
    s1 = array(in1.shape)
    s2 = array(in2.shape)
    complex_result = (numpy.issubdtype(in1.dtype, numpy.complex) or
                      numpy.issubdtype(in2.dtype, numpy.complex))
    size = s1+s2-1
    IN1 = fftn(in1,size)
    IN1 *= fftn(in2,size)
    ret = ifftn(IN1)
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        if product(s1,axis=0) > product(s2,axis=0):
            osize = s1
        else:
            osize = s2
        return _centered(ret,osize)
    elif mode == "valid":
        return _centered(ret,abs(s2-s1)+1)

#
################################################################################
