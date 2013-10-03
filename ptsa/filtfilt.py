#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

# Pulls In filtfilt from the SciPy future

from scipy import linalg
from scipy.signal import lfilter
import numpy as np
from _arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext


def lfilter_zi(b, a):
    """
Compute an initial state `zi` for the lfilter function that corresponds
to the steady state of the step response.

A typical use of this function is to set the initial state so that the
output of the filter starts at the same value as the first element of
the signal to be filtered.

Parameters
----------
b, a : array_like (1-D)
The IIR filter coefficients. See `scipy.signal.lfilter` for more
information.

Returns
-------
zi : 1-D ndarray
The initial state for the filter.

Notes
-----
A linear filter with order m has a state space representation (A, B, C, D),
for which the output y of the filter can be expressed as::

z(n+1) = A*z(n) + B*x(n)
y(n) = C*z(n) + D*x(n)

where z(n) is a vector of length m, A has shape (m, m), B has shape
(m, 1), C has shape (1, m) and D has shape (1, 1) (assuming x(n) is
a scalar). lfilter_zi solves::

zi = A*zi + B

In other words, it finds the initial condition for which the response
to an input of all ones is a constant.

Given the filter coefficients `a` and `b`, the state space matrices
for the transposed direct form II implementation of the linear filter,
which is the implementation used by scipy.signal.lfilter, are::

A = scipy.linalg.companion(a).T
B = b[1:] - a[1:]*b[0]

assuming `a[0]` is 1.0; if `a[0]` is not 1, `a` and `b` are first
divided by a[0].

Examples
--------
The following code creates a lowpass Butterworth filter. Then it
applies that filter to an array whose values are all 1.0; the
output is also all 1.0, as expected for a lowpass filter. If the
`zi` argument of `lfilter` had not been given, the output would have
shown the transient signal.

>>> from numpy import array, ones
>>> from scipy.signal import lfilter, lfilter_zi, butter
>>> b, a = butter(5, 0.25)
>>> zi = lfilter_zi(b, a)
>>> y, zo = lfilter(b, a, ones(10), zi=zi)
>>> y
array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

Another example:

>>> x = array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
>>> y, zf = lfilter(b, a, x, zi=zi*x[0])
>>> y
array([ 0.5 , 0.5 , 0.5 , 0.49836039, 0.48610528,
0.44399389, 0.35505241])

Note that the `zi` argument to `lfilter` was computed using
`lfilter_zi` and scaled by `x[0]`. Then the output `y` has no
transient until the input drops from 0.5 to 0.0.

"""

    # FIXME: Can this function be replaced with an appropriate
    # use of lfiltic? For example, when b,a = butter(N,Wn),
    # lfiltic(b, a, y=numpy.ones_like(a), x=numpy.ones_like(b)).
    #

    # We could use scipy.signal.normalize, but it uses warnings in
    # cases where a ValueError is more appropriate, and it allows
    # b to be 2D.
    b = np.atleast_1d(b)
    if b.ndim != 1:
        raise ValueError("Numerator b must be rank 1.")
    a = np.atleast_1d(a)
    if a.ndim != 1:
        raise ValueError("Denominator a must be rank 1.")

    while len(a) > 1 and a[0] == 0.0:
        a = a[1:]
    if a.size < 1:
        raise ValueError("There must be at least one nonzero `a` coefficient.")

    if a[0] != 1.0:
        # Normalize the coefficients so a[0] == 1.
        a = a / a[0]
        b = b / a[0]

    n = max(len(a), len(b))

    # Pad a or b with zeros so they are the same length.
    if len(a) < n:
        a = np.r_[a, np.zeros(n - len(a))]
    elif len(b) < n:
        b = np.r_[b, np.zeros(n - len(b))]

    IminusA = np.eye(n - 1) - linalg.companion(a).T
    B = b[1:] - a[1:] * b[0]
    # Solve zi = A*zi + B
    zi = np.linalg.solve(IminusA, B)

    # For future reference: we could also use the following
    # explicit formulas to solve the linear system:
    #
    # zi = np.zeros(n - 1)
    # zi[0] = B.sum() / IminusA[:,0].sum()
    # asum = 1.0
    # csum = 0.0
    # for k in range(1,n-1):
    # asum += a[k]
    # csum += b[k] - a[k]*b[0]
    # zi[k] = asum*zi[0] - csum

    return zi


def filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None):
    """A forward-backward filter.

This function applies a linear filter twice, once forward
and once backwards. The combined filter has linear phase.

Before applying the filter, the function can pad the data along the
given axis in one of three ways: odd, even or constant. The odd
and even extensions have the corresponding symmetry about the end point
of the data. The constant extension extends the data with the values
at end points. On both the forward and backwards passes, the
initial condition of the filter is found by using lfilter_zi and
scaling it by the end point of the extended data.

Parameters
----------
b : array_like, 1-D
The numerator coefficient vector of the filter.
a : array_like, 1-D
The denominator coefficient vector of the filter. If a[0]
is not 1, then both a and b are normalized by a[0].
x : array_like
The array of data to be filtered.
axis : int, optional
The axis of `x` to which the filter is applied.
Default is -1.
padtype : str or None, optional
Must be 'odd', 'even', 'constant', or None. This determines the
type of extension to use for the padded signal to which the filter
is applied. If `padtype` is None, no padding is used. The default
is 'odd'.
padlen : int or None, optional
The number of elements by which to extend `x` at both ends of
`axis` before applying the filter. This value must be less than
`x.shape[axis]-1`. `padlen=0` implies no padding.
The default value is 3*max(len(a),len(b)).

Returns
-------
y : ndarray
The filtered output, an array of type numpy.float64 with the same
shape as `x`.

See Also
--------
lfilter_zi
lfilter

Examples
--------
First we create a one second signal that is the sum of two pure sine
waves, with frequencies 5 Hz and 250 Hz, sampled at 2000 Hz.

>>> t = np.linspace(0, 1.0, 2001)
>>> xlow = np.sin(2 * np.pi * 5 * t)
>>> xhigh = np.sin(2 * np.pi * 250 * t)
>>> x = xlow + xhigh

Now create a lowpass Butterworth filter with a cutoff of 0.125 times
the Nyquist rate, or 125 Hz, and apply it to x with filtfilt. The
result should be approximately xlow, with no phase shift.

>>> from scipy.signal import butter
>>> b, a = butter(8, 0.125)
>>> y = filtfilt(b, a, x, padlen=150)
>>> np.abs(y - xlow).max()
9.1086182074789912e-06

We get a fairly clean result for this artificial example because
the odd extension is exact, and with the moderately long padding,
the filter's transients have dissipated by the time the actual data
is reached. In general, transient effects at the edges are
unavoidable.
"""

    if padtype not in ['even', 'odd', 'constant', None]:
        raise ValueError(("Unknown value '%s' given to padtype. padtype must "
                         "be 'even', 'odd', 'constant', or None.") %
                            padtype)

    b = np.asarray(b)
    a = np.asarray(a)
    x = np.asarray(x)

    ntaps = max(len(a), len(b))

    if padtype is None:
        padlen = 0

    if padlen is None:
        # Original padding; preserved for backwards compatibility.
        edge = ntaps * 3
    else:
        edge = padlen

    # x's 'axis' dimension must be bigger than edge.
    if x.shape[axis] <= edge:
        raise ValueError("The length of the input vector x must be at least "
                         "padlen, which is %d." % edge)

    if padtype is not None and edge > 0:
        # Make an extension of length `edge` at each
        # end of the input array.
        if padtype == 'even':
            ext = even_ext(x, edge, axis=axis)
        elif padtype == 'odd':
            ext = odd_ext(x, edge, axis=axis)
        else:
            ext = const_ext(x, edge, axis=axis)
    else:
        ext = x

    # Get the steady state of the filter's step response.
    zi = lfilter_zi(b, a)

    # Reshape zi and create x0 so that zi*x0 broadcasts
    # to the correct value for the 'zi' keyword argument
    # to lfilter.
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = np.reshape(zi, zi_shape)
    x0 = axis_slice(ext, stop=1, axis=axis)

    # Forward filter.
    (y, zf) = lfilter(b, a, ext, zi=zi * x0)

    # Backward filter.
    # Create y0 so zi*y0 broadcasts appropriately.
    y0 = axis_slice(y, start=-1, axis=axis)
    (y, zf) = lfilter(b, a, axis_reverse(y, axis=axis), zi=zi * y0)

    # Reverse y.
    y = axis_reverse(y, axis=axis)

    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)

    return y
