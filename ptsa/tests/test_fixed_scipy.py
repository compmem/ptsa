#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as N

#from scipy.signal import wavelets
from numpy.testing import NumpyTest, NumpyTestCase, assert_equal, assert_array_less
import ptsa.fixed_scipy as wavelets

class TestWavelets(NumpyTestCase):
    def test_morlet(self):

        x = wavelets.morlet(50,4.1,complete=True)
        y = wavelets.morlet(50,4.1,complete=False)
        # Test if complete and incomplete wavelet have same lengths:
        assert_equal(len(x),len(y))
        # Test if complete wavelet is less than incomplete wavelet:
        assert_array_less(x,y)
        
        x = wavelets.morlet(10,50,complete=False)
        y = wavelets.morlet(10,50,complete=True)
        # For large widths complete and incomplete wavelets should be
        # identical:
        assert_equal(x,y)

        print "bla bla"
