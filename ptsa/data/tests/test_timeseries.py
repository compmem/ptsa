import numpy as N
import re
from numpy.testing import NumpyTest, NumpyTestCase


from ptsa.data import Dim,Dims,DimData,TimeSeries
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
        d200_10 = N.sin(N.arange(numPoints,dtype=N.float)*2*N.pi*Hz*numSecs/numPoints)
        Hz = 5
        d200_5 = N.sin(N.arange(numPoints,dtype=N.float)*2*N.pi*Hz*numSecs/numPoints)
        self.dat200 = N.array([d200_10,d200_5])
        # calc the time range in MS
        offset = -200
        duration = numPoints
        samplesize = 1000./200.
        sampStart = offset*samplesize
        sampEnd = sampStart + (duration-1)*samplesize
        timeRange = N.linspace(sampStart,sampEnd,duration)
        self.dims200 = [Dim('channel',N.arange(self.dat200.shape[0])),
                        Dim('time',timeRange,'ms')]
        
        numSecs = 4.
        numPoints = int(numSecs*50.)
        Hz = 10
        d50_10 = N.sin(N.arange(numPoints,dtype=N.float)*2*N.pi*Hz*numSecs/numPoints)
        Hz = 5
        d50_5 = N.sin(N.arange(numPoints,dtype=N.float)*2*N.pi*Hz*numSecs/numPoints)
        self.dat50 = N.array([d50_10,d50_5])
        # calc the time range in MS
        offset = -50
        duration = numPoints
        samplesize = 1000./50.
        sampStart = offset*samplesize
        sampEnd = sampStart + (duration-1)*samplesize
        timeRange = N.linspace(sampStart,sampEnd,duration)
        self.dims50 = [Dim('channel',N.arange(self.dat50.shape[0])),
                       Dim('time',timeRange,'ms')]
 



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
        # (data,dims,samplerate,unit=None,tdim=-1,buf=0)

        # fewest params
        ts = TimeSeries(self.dat200,self.dims200,200)
        N.testing.assert_equal(ts.data[:], self.dat200[:])
        self.assertEquals(ts.shape, self.dat200.shape)
        self.assertEquals(ts.ndim, len(self.dat200.shape))
        self.assertEquals(ts.tdim, len(self.dat200.shape)-1)
        self.assertEquals(ts.buf_samp, 0)
        self.assertEquals(ts.unit,None)
        self.assertEquals(ts.samplerate,200)

        
    def test_removeBuf(self):
        buf = 200
        numsamp = 4*200
        ts = TimeSeries(self.dat200,self.dims200,200,buf_samp=buf)
        ts.removeBuf()
        self.assertEquals(ts.shape[ts.tdim],numsamp-2*buf)
        self.assertEquals(len(ts['time']),numsamp-2*buf)

#    def test_getitem(self):
#        ts = TimeSeries(self.dat200,self.dims200,200)
#        for i in range(N.shape(ts)[0]):
#            N.testing.assert_equal(ts.__getitem__(i),self.dat200[i])

#    def test_setitem(self):        
#        pass
#         ts = TimeSeries(self.dat200,200)
#         x = N.arange(10)
#         ts[11:11+10] = x
#         N.testing.assert_equal(ts[11:11+10],x[:])

    def test_filter(self):
        samplerate = 200
        filtType='stop'
        freqRange = [10,20]
        order = 4
        ts = TimeSeries(self.dat200,self.dims200,samplerate)
        ts.filter(freqRange)
        test = filt.buttfilt(self.dat200,freqRange,samplerate,filtType,
                             order,axis=ts.tdim)
        N.testing.assert_array_almost_equal(ts[:],test[:],decimal=6)

    def test_resample(self):
        ts200 = TimeSeries(self.dat200,self.dims200,200,buf_samp=200)
        ts50 = TimeSeries(self.dat50,self.dims50,50,buf_samp=50)
        ts200.resample(50)
        ts200.removeBuf()
        ts50.removeBuf()
        N.testing.assert_equal(ts200.shape[:],ts50.shape[:])
        #print type(ts200['time'])
        #print type(ts50['time'])
        N.testing.assert_array_almost_equal(ts200['time'],ts50['time'],decimal=6)
        N.testing.assert_array_almost_equal(ts200.data[:],ts50.data[:],decimal=6)

# test RawBinaryEEG

# load data from file

# make sure the time range is correct

# make sure we get the expected number of samples


if __name__ == '__main__':
    NumpyTest.main()
