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
from scipy.signal import hilbert

from ptsa.data.timeseries import TimeSeries,Dim
from ptsa.helper import next_pow2

freq_bands = [('delta', [2.0,4.0]),
              ('theta', [4.0,8.0]),
              ('alpha', [9.0,14.0]),
              ('beta', [16.0,26.0]),
              ('gamma_1', [28.0,42.0]),
              ('gamma_2', [44.0,100.0])]
def hilbert_pow(dat_ts, bands=None, pad_to_pow2=False, verbose=True):
    """
    """
    # set default freq bands
    if bands is None:
        bands = freq_bands

    # proc padding
    taxis = dat_ts.get_axis(dat_ts.tdim)
    npts_orig = dat_ts.shape[taxis]
    if pad_to_pow2:
        npts = 2**next_pow2(npts_orig)
    else:
        npts = npts_orig

    # calc the hilbert power
    if verbose:
        sys.stdout.write('Hilbert Bands: ')
        sys.stdout.flush()
    pow = None
    for band in bands:
        if verbose:
            sys.stdout.write('%s '%band[0])
            sys.stdout.flush()
        p = TimeSeries(np.abs(hilbert(dat_ts.filtered(band[1], 
                                                      filt_type='pass'),
                                      N=npts, axis=taxis).take(np.arange(npts_orig),
                                                               axis=taxis)), 
                       tdim=dat_ts.tdim, samplerate=dat_ts.samplerate, 
                       dims=dat_ts.dims.copy()).add_dim(Dim([band[0]],'freqs'))
        if pow is None:
            pow = p
        else:
            pow = pow.extend(p, 'freqs')

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()
    return pow
