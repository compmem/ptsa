import numpy as N
import re
from numpy.testing import NumpyTest, NumpyTestCase


from pyeeg.data import Dim,Dims,DimData,EegTimeSeries
from pyeeg import filter

from testdata import TestData

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


# test EegTimeSeries
class test_EegTimeSeries(NumpyTestCase):
    def setUp(self):
        td = TestData()
        self.dat200 = td.dat200
        self.dims200 = td.dims200
        self.dat50 = td.dat50
        self.dims50 = td.dims50
    def test_init(self):
        # init a TimeSeries with all combos of options and verify that
        # the attributes are correct
        # (data,dims,samplerate,unit=None,tdim=-1,buffer=0)

        # fewest params
        ts = EegTimeSeries(self.dat200,self.dims200,200)
        N.testing.assert_equal(ts.data[:], self.dat200[:])
        self.assertEquals(ts.shape, self.dat200.shape)
        self.assertEquals(ts.ndim, len(self.dat200.shape))
        self.assertEquals(ts.tdim, len(self.dat200.shape)-1)
        self.assertEquals(ts.buffer, 0)
        self.assertEquals(ts.unit,None)
        self.assertEquals(ts.samplerate,200)

        
    def test_removeBuffer(self):
        buffer = 200
        numsamp = 4*200
        ts = EegTimeSeries(self.dat200,self.dims200,200,buffer=buffer)
        ts.removeBuffer()
        self.assertEquals(ts.shape[ts.tdim],numsamp-2*buffer)
        self.assertEquals(len(ts['time']),numsamp-2*buffer)

#    def test_getitem(self):
#        ts = EegTimeSeries(self.dat200,self.dims200,200)
#        for i in range(N.shape(ts)[0]):
#            N.testing.assert_equal(ts.__getitem__(i),self.dat200[i])

#    def test_setitem(self):        
#        pass
#         ts = EegTimeSeries(self.dat200,200)
#         x = N.arange(10)
#         ts[11:11+10] = x
#         N.testing.assert_equal(ts[11:11+10],x[:])

    def test_filter(self):
        samplerate = 200
        filtType='stop'
        freqRange = [10,20]
        order = 4
        ts = EegTimeSeries(self.dat200,self.dims200,samplerate)
        ts.filter(freqRange)
        test = filter.buttfilt(self.dat200,freqRange,samplerate,filtType,
                               order,axis=ts.tdim)
        N.testing.assert_array_almost_equal(ts[:],test[:],decimal=6)

    def test_resample(self):
        ts200 = EegTimeSeries(self.dat200,self.dims200,200,buffer=200)
        ts50 = EegTimeSeries(self.dat50,self.dims50,50,buffer=50)
        ts200.resample(50)
        ts200.removeBuffer()
        ts50.removeBuffer()
        N.testing.assert_equal(ts200.shape[:],ts50.shape[:])
        N.testing.assert_array_almost_equal(ts200['time'],ts50['time'],decimal=6)
        N.testing.assert_array_almost_equal(ts200.data[:],ts50.data[:],decimal=6)

# test RawBinaryEEG

# load data from file

# make sure the time range is correct

# make sure we get the expected number of samples


if __name__ == '__main__':
    NumpyTest.main()
