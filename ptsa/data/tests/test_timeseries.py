#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
import re
from numpy.testing import NumpyTest, NumpyTestCase


from ptsa.data import Dim,DimArray,TimeSeries
from ptsa import filt

# from numpy.testing import NumpyTest, NumpyTestCase

# class test_Template(NumpyTestCase):

#     def setUp(self): pass
#         #print "TestCase initialization..."

#     def test_foo(self): pass
#         #print "testing foo"

#     def test_bar(self): pass
#         #print "testing bar"
      

# if __name__ == '__main__':
#     NumpyTest.main()

# I don't know why I can't just include this
#from testdata import TestData

class TestData:
    def __init__(self):
        # create 10 Hz sine waves at 200 and 50 Hz 4000ms long
        numSecs = 4.
        numPoints = int(numSecs*200.)
        Hz = 10
        d200_10 = np.sin(np.arange(numPoints,dtype=np.float)*2*np.pi*Hz*numSecs/numPoints)
        Hz = 5
        d200_5 = np.sin(np.arange(numPoints,dtype=np.float)*2*np.pi*Hz*numSecs/numPoints)
        self.dat200 = np.array([d200_10,d200_5])
        # calc the time range
        offset = -200
        duration = numPoints
        samplesize = 1./200.
        sampStart = offset*samplesize
        sampEnd = sampStart + (duration-1)*samplesize
        timeRange = np.linspace(sampStart,sampEnd,duration)
        self.dims200 = [Dim(np.arange(self.dat200.shape[0]),'channel'),
                        Dim(timeRange,'time',unit='ms')]
        
        numSecs = 4.
        numPoints = int(numSecs*50.)
        Hz = 10
        d50_10 = np.sin(np.arange(numPoints,dtype=np.float)*2*np.pi*Hz*numSecs/numPoints)
        Hz = 5
        d50_5 = np.sin(np.arange(numPoints,dtype=np.float)*2*np.pi*Hz*numSecs/numPoints)
        self.dat50 = np.array([d50_10,d50_5])
        # calc the time range in MS
        offset = -50
        duration = numPoints
        samplesize = 1000./50.
        sampStart = offset*samplesize
        sampEnd = sampStart + (duration-1)*samplesize
        timeRange = np.linspace(sampStart,sampEnd,duration)
        self.dims50 = [Dim(np.arange(self.dat50.shape[0]),'channel'),
                        Dim(timeRange,'time',unit='ms')]
 


# test TimeSeries
class test_TimeSeries(NumpyTestCase):
    def setUp(self):
        td = TestData()
        self.dat200 = td.dat200
        self.dims200 = td.dims200
        self.dat50 = td.dat50
        self.dims50 = td.dims50
    def test_init(self):
        # init a TimeSeries with all combos of options and verify that
        # the attributes are correct

        # fewest params
        ts = TimeSeries(self.dat200,self.dims200,'time',200)
        np.testing.assert_equal(ts[:], self.dat200[:])
        self.assertEquals(ts.shape, self.dat200.shape)
        self.assertEquals(ts.taxis, len(self.dat200.shape)-1)
        self.assertEquals(ts.samplerate,200)
        self.assertRaises(ValueError,TimeSeries,self.dat200,self.dims200,'bla',200)
        self.assertRaises(ValueError,TimeSeries,self.dat200,self.dims200,'time',-200)

        
    def test_remove_buffer(self):
        buf = 200
        numsamp = 4*200
        ts = TimeSeries(self.dat200,self.dims200,'time',200)
        ts_nobuff = ts.remove_buffer(1)
        self.assertEquals(ts_nobuff.shape[ts_nobuff.taxis],numsamp-2*buf)
        self.assertEquals(len(ts_nobuff['time']),numsamp-2*buf)
        ts_nobuff = ts.remove_buffer((1,1))
        self.assertEquals(ts_nobuff.shape[ts_nobuff.taxis],numsamp-2*buf)
        self.assertEquals(len(ts_nobuff['time']),numsamp-2*buf)
        # make sure that negative durations throw exception
        self.assertRaises(ValueError,ts.remove_buffer,-1)

    def test_filter(self):
        samplerate = 200
        filtType='stop'
        freqRange = [10,20]
        order = 4
        ts = TimeSeries(self.dat200,self.dims200,'time',samplerate)
        ts_filt = ts.filtered(freqRange, filtType, order)
        test = filt.buttfilt(self.dat200,freqRange,samplerate,filtType,
                             order,axis=ts.taxis)
        np.testing.assert_array_almost_equal(ts_filt[:],test[:],decimal=6)

    def test_resample(self):
        ts200 = TimeSeries(self.dat200,self.dims200,'time',200)
        ts50 = TimeSeries(self.dat50,self.dims50,'time',50).remove_buffer(1.0)
        ts50_200 = ts200.resampled(50).remove_buffer(1.0)
        np.testing.assert_equal(ts50_200.shape[:],ts50.shape[:])
        #print type(ts200['time'])
        #print type(ts50['time'])
        np.testing.assert_array_almost_equal(ts50_200['time']*1000,ts50['time'],decimal=6)
        np.testing.assert_array_almost_equal(ts50_200[:],ts50[:],decimal=6)

# test RawBinaryEEG

# load data from file

# make sure the time range is correct

# make sure we get the expected number of samples


if __name__ == '__main__':
    NumpyTest.main()
