#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
import re
from numpy.testing import * #NumpyTest, NumpyTestCase

from ptsa.wavelet import *
from ptsa.data import TimeSeries,Dim



class test_morlet_multi(TestCase):
    def test_morlet_multi(self):
        # make sure errors are raised when not called with enough or
        # the right kind of arguments:
        self.assertRaises(TypeError,morlet_multi)
        self.assertRaises(ValueError,morlet_multi,[],[],200)
        self.assertRaises(ValueError,morlet_multi,[1],[],200)
        self.assertRaises(ValueError,morlet_multi,[],[1],200)
        self.assertRaises(ValueError,morlet_multi,[1],[1],[])
        self.assertRaises(ValueError,morlet_multi,[1,2],[1,2,3],200)
        self.assertRaises(ValueError,morlet_multi,[1,2,3],[1,2],200)
        self.assertRaises(ValueError,morlet_multi,[1],[1,2],200)
        self.assertRaises(ValueError,morlet_multi,[1],[1],200,[1,2])
        self.assertRaises(ValueError,morlet_multi,[1,2,3],[1],200,[1,2])
        
        x = morlet_multi(2,4,200,complete=True)
        y = morlet_multi(2,4,200,complete=False)
        # Make sure we got one wavelet in each case:
        self.assertEqual(len(x),1)
        self.assertEqual(len(y),1)
        # Test if complete and incomplete wavelet have same lengths:
        self.assertEqual(len(x[0]),len(y[0]))
        # Test if complete wavelet is less than incomplete wavelet:
        assert_array_less(x[0],y[0])

        x = morlet_multi([2,2,2],5,100)
        # len(freqs) wavelets are expected:
        self.assertEqual(len(x),3)
        # when the same frequencies and widths are given, the same
        # wavelets should result:
        assert_array_equal(x[0],x[1])
        assert_array_equal(x[0],x[2])

        
        x = morlet_multi([2,4,6],5,100)
        # len(freqs) wavelets are expected:
        self.assertEqual(len(x),3)
        # length of wavelet should be longer for lower frequencies
        # (all else being equal)
        self.assertTrue(len(x[0])>len(x[1]) and len(x[1])>len(x[2]))

        x = morlet_multi([2,2,2],[3,6,9],100)
        # len(freqs) wavelets are expected:
        self.assertEqual(len(x),3)
        # length of wavelet should be longer for greater widths
        # (all else being equal)
        self.assertTrue(len(x[0])<len(x[1]) and len(x[1])<len(x[2]))

        x = morlet_multi([2,2,2],5,100,[7,8,9])
        # len(freqs) wavelets are expected:
        self.assertEqual(len(x),3)
        # length of wavelet should be longer for larger sampling
        # windows (all else being equal)
        self.assertTrue(len(x[0])<len(x[1]) and len(x[1])<len(x[2]))

        x = morlet_multi([2,2,2,2,2,2],[3,6],100,[7,8,9])
        # len(freqs) wavelets are expected:
        self.assertEqual(len(x),6)
        # make sure wavelets with same input are equal:
        assert_array_equal(x[0],x[1])
        assert_array_equal(x[4],x[5])
        # test length relationships based on (combination of) widths
        # and sampling window:
        self.assertTrue(len(x[0])<len(x[2]))
        self.assertTrue(len(x[2])<len(x[3]))
        self.assertTrue(len(x[3])<len(x[4]))
        self.assertTrue(len(x[2])-len(x[0])<len(x[3])-len(x[2]))
        self.assertTrue(len(x[4])-len(x[3])<len(x[3])-len(x[2]))

        x = morlet_multi(10,5,200)
        # len(freqs) wavelets are expected:
        self.assertEqual(len(x),1)
        # compare wavelet length with known length:
        self.assertEqual(len(x[0]),112)

        x = morlet_multi([9,9,9,9,9,9,9,9,9,9,9,9],[5,6],
                         [100,200,300],[7,8,9,10])
        # len(freqs) wavelets are expected:
        self.assertEqual(len(x),12)
        # make sure wavelets with same input are equal:
        assert_array_equal(x[0],x[1])
        assert_array_equal(x[1],x[2])
        assert_array_equal(x[4],x[5])
        assert_array_equal(x[6],x[7])
        assert_array_equal(x[9],x[10])
        assert_array_equal(x[10],x[11])
        # test length relationships based on widths, sampling rate and
        # sampling window:
        self.assertTrue(len(x[0])<len(x[3]))
        self.assertTrue(len(x[3])<len(x[4]))
        self.assertTrue(len(x[4])<len(x[6]))
        self.assertTrue(len(x[6])<len(x[8]))
        self.assertTrue(len(x[8])<len(x[9]))

    def test_phase_pow_multi(self):
        dat = np.vstack((np.arange(0,1000),np.arange(0,1000)))
        # make sure errors are raised when not called with enough or
        # the right kind of arguments:
        self.assertRaises(TypeError,phase_pow_multi)
        self.assertRaises(ValueError,phase_pow_multi,[],dat,100)
        self.assertRaises(ValueError,phase_pow_multi,[1],dat,100,
                          to_return='results')
        self.assertRaises(ValueError,phase_pow_multi,[1],dat,100,
                          conv_dtype=np.float)
        dat_short = np.reshape(np.arange(0,20),(2,10))
        self.assertRaises(ValueError,phase_pow_multi,[1],dat_short,100)

        dat_ts = TimeSeries(dat, tdim = 'time', samplerate = 100,
                            dims = [Dim(np.arange(2),name='dim1'),
                                    Dim(np.arange(1000),name='time')])
        
        # make sure errors are raised when not called with enough or
        # the right kind of arguments:
        self.assertRaises(ValueError,phase_pow_multi,[],dat_ts)
        self.assertRaises(ValueError,phase_pow_multi,[1],dat_ts,
                          to_return='results')
        self.assertRaises(ValueError,phase_pow_multi,[1],dat_ts,
                          conv_dtype=np.float)
        dat_short_ts = TimeSeries(dat_short, tdim = 'time', samplerate = 100,
                                  dims = [Dim(np.arange(2),name='dim1'),
                                          Dim(np.arange(10),name='time')])
        self.assertRaises(ValueError,phase_pow_multi,[1],dat_short_ts)

        x = phase_pow_multi(1,dat,100)
        # ensure correct output shape:
        self.assertEqual(np.shape(x),(2,1,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x[0][0][0],x[0][0][1])
        #print x[0][0][0][:20]
        assert_array_equal(x[1][0][0],x[1][0][1])
        # ensure valid output values:
        phaseTest = np.abs(x[0]) <= np.pi
        powerTest = x[1] >= 0
        self.assertTrue(phaseTest.all())
        self.assertTrue(powerTest.all())
        
        x_ts = phase_pow_multi(1,dat_ts)
        # ensure correct output shape:
        self.assertEqual(np.shape(x_ts),(2,1,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x_ts[0][0][0],x_ts[0][0][1])
        assert_array_equal(x_ts[1][0][0],x_ts[1][0][1])
        # ensure valid output values:
        phaseTest = np.abs(x_ts[0]) <= np.pi
        powerTest = x_ts[1] >= 0
        self.assertTrue(phaseTest.all())
        self.assertTrue(powerTest.all())
        
        y = phase_pow_multi([1],dat,100,to_return='phase')
        # ensure correct output shape:
        self.assertEqual(np.shape(y),(1,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x[0][0][0],y[0][1])
        # ensure valid output values:
        phaseTest = np.abs(y[0]) <= np.pi
        self.assertTrue(phaseTest.all())

        y_ts = phase_pow_multi([1],dat_ts,to_return='phase')
        # ensure correct output shape:
        self.assertEqual(np.shape(y_ts),(1,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x_ts[0][0][0],y_ts[0][1])
        # ensure valid output values:
        phaseTest = np.abs(y_ts[0]) <= np.pi
        self.assertTrue(phaseTest.all())

        z = phase_pow_multi(1,dat,[100],to_return='power')
        # ensure correct output shape:
        self.assertEqual(np.shape(z),(1,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x[1][0][0],z[0][1])
        # ensure valid output values:
        powerTest = z >= 0
        self.assertTrue(powerTest.all())

        z_ts = phase_pow_multi(1,dat_ts,to_return='power')
        # ensure correct output shape:
        self.assertEqual(np.shape(z_ts),(1,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x_ts[1][0][0],z_ts[0][1])
        # ensure valid output values:
        powerTest = z_ts >= 0
        self.assertTrue(powerTest.all())

        x = phase_pow_multi([1,2,3],dat,100,widths=6)
        # ensure correct output shape:
        self.assertEqual(np.shape(x),(2,3,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x[0][0][0],x[0][0][1])
        assert_array_equal(x[1][0][0],x[1][0][1])
        assert_array_equal(x[0][1][0],x[0][1][1])
        assert_array_equal(x[1][1][0],x[1][1][1])
        assert_array_equal(x[0][2][0],x[0][2][1])
        assert_array_equal(x[1][2][0],x[1][2][1])
        # ensure valid output values:
        phaseTest = np.abs(x[0]) <= np.pi
        powerTest = x[1] >= 0
        self.assertTrue(phaseTest.all())
        self.assertTrue(powerTest.all())
        
        x_ts = phase_pow_multi([1,2,3],dat_ts,widths=6)
        # ensure correct output shape:
        self.assertEqual(np.shape(x_ts),(2,3,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x_ts[0][0][0],x_ts[0][0][1])
        assert_array_equal(x_ts[1][0][0],x_ts[1][0][1])
        assert_array_equal(x_ts[0][1][0],x_ts[0][1][1])
        assert_array_equal(x_ts[1][1][0],x_ts[1][1][1])
        assert_array_equal(x_ts[0][2][0],x_ts[0][2][1])
        assert_array_equal(x_ts[1][2][0],x_ts[1][2][1])
        # ensure valid output values:
        phaseTest = np.abs(x_ts[0]) <= np.pi
        powerTest = x_ts[1] >= 0
        self.assertTrue(phaseTest.all())
        self.assertTrue(powerTest.all())
        
        y = phase_pow_multi([1,2,3],dat,[100],widths=6,to_return='phase')
        # ensure correct output shape:
        self.assertEqual(np.shape(y),(3,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x[0][0][0],y[0][1])
        assert_array_equal(x[0][1][0],y[1][1])
        assert_array_equal(x[0][2][0],y[2][1])
        # ensure valid output values:
        phaseTest = np.abs(y) <= np.pi
        self.assertTrue(phaseTest.all())

        y_ts = phase_pow_multi([1,2,3],dat,[100],widths=6,to_return='phase')
        # ensure correct output shape:
        self.assertEqual(np.shape(y_ts),(3,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x_ts[0][0][0],y_ts[0][1])
        assert_array_equal(x_ts[0][1][0],y_ts[1][1])
        assert_array_equal(x_ts[0][2][0],y_ts[2][1])
        # ensure valid output values:
        phaseTest = np.abs(y_ts) <= np.pi
        self.assertTrue(phaseTest.all())

        z = phase_pow_multi([1,2,3],dat,100,widths=[6],to_return='power')
        # ensure correct output shape:
        self.assertEqual(np.shape(z),(3,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x[1][0][0],z[0][1])
        assert_array_equal(x[1][1][0],z[1][1])
        assert_array_equal(x[1][2][0],z[2][1])
        # ensure valid output values:
        powerTest = z >= 0
        self.assertTrue(powerTest.all())

        z_ts = phase_pow_multi([1,2,3],dat,100,widths=[6],to_return='power')
        # ensure correct output shape:
        self.assertEqual(np.shape(z_ts),(3,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x_ts[1][0][0],z_ts[0][1])
        assert_array_equal(x_ts[1][1][0],z_ts[1][1])
        assert_array_equal(x_ts[1][2][0],z_ts[2][1])
        # ensure valid output values:
        powerTest = z_ts >= 0
        self.assertTrue(powerTest.all())

        x = phase_pow_multi([4,9,8],dat,[100,200,300],widths=[6,5,4])
        # ensure correct output shape:
        self.assertEqual(np.shape(x),(2,3,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x[0][0][0],x[0][0][1])
        assert_array_equal(x[1][0][0],x[1][0][1])
        assert_array_equal(x[0][1][0],x[0][1][1])
        assert_array_equal(x[1][1][0],x[1][1][1])
        assert_array_equal(x[0][2][0],x[0][2][1])
        assert_array_equal(x[1][2][0],x[1][2][1])
        # ensure valid output values:
        phaseTest = np.abs(x[0]) <= np.pi
        powerTest = x[1] >= 0
        self.assertTrue(phaseTest.all())
        self.assertTrue(powerTest.all())
        
        x_ts = phase_pow_multi([4,9,8],dat,[100,200,300],widths=[6,5,4])
        # ensure correct output shape:
        self.assertEqual(np.shape(x_ts),(2,3,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x_ts[0][0][0],x_ts[0][0][1])
        assert_array_equal(x_ts[1][0][0],x_ts[1][0][1])
        assert_array_equal(x_ts[0][1][0],x_ts[0][1][1])
        assert_array_equal(x_ts[1][1][0],x_ts[1][1][1])
        assert_array_equal(x_ts[0][2][0],x_ts[0][2][1])
        assert_array_equal(x_ts[1][2][0],x_ts[1][2][1])
        # ensure valid output values:
        phaseTest = np.abs(x_ts[0]) <= np.pi
        powerTest = x_ts[1] >= 0
        self.assertTrue(phaseTest.all())
        self.assertTrue(powerTest.all())
        
        y = phase_pow_multi([4,9,8],dat,[100,200,300],
                            widths=[6,5,4],to_return='phase')
        # ensure correct output shape:
        self.assertEqual(np.shape(y),(3,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x[0][0][0],y[0][1])
        assert_array_equal(x[0][1][0],y[1][1])
        assert_array_equal(x[0][2][0],y[2][1])
        # ensure valid output values:
        phaseTest = np.abs(y) <= np.pi
        self.assertTrue(phaseTest.all())

        y_ts = phase_pow_multi([4,9,8],dat,[100,200,300],
                               widths=[6,5,4],to_return='phase')
        # ensure correct output shape:
        self.assertEqual(np.shape(y_ts),(3,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x_ts[0][0][0],y_ts[0][1])
        assert_array_equal(x_ts[0][1][0],y_ts[1][1])
        assert_array_equal(x_ts[0][2][0],y_ts[2][1])
        # ensure valid output values:
        phaseTest = np.abs(y_ts) <= np.pi
        self.assertTrue(phaseTest.all())

        z = phase_pow_multi([4,9,8],dat,[100,200,300],
                             widths=[6,5,4],to_return='power')
        # ensure correct output shape:
        self.assertEqual(np.shape(z),(3,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x[1][0][0],z[0][1])
        assert_array_equal(x[1][1][0],z[1][1])
        assert_array_equal(x[1][2][0],z[2][1])
        # ensure valid output values:
        powerTest = z >= 0
        self.assertTrue(powerTest.all())

        z_ts = phase_pow_multi([4,9,8],dat,[100,200,300],
                             widths=[6,5,4],to_return='power')
        # ensure correct output shape:
        self.assertEqual(np.shape(z_ts),(3,2,1000))
        # dat has two identical rows, ensure output has corresponding
        # identities:
        assert_array_equal(x_ts[1][0][0],z_ts[0][1])
        assert_array_equal(x_ts[1][1][0],z_ts[1][1])
        assert_array_equal(x_ts[1][2][0],z_ts[2][1])
        # ensure valid output values:
        powerTest = z_ts >= 0
        self.assertTrue(powerTest.all())
