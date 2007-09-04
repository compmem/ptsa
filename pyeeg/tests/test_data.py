import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

from pyeeg import EegTimeSeries

class test_Template(NumpyTestCase):

    def setUp(self): pass
        #print "TestCase initialization..."

    def test_foo(self): pass
        #print "testing foo"

    def test_bar(self): pass
        #print "testing bar"
      

if __name__ == '__main__':
    NumpyTest.main()

# test TimeSeries
class test_EegTimeSeries(NumpyTestCase):
    def setUp(self): 
        # create 10 Hz sine waves at 200 and 50 Hz 4000ms long
        numSecs = 4.
        numPoints = int(numSecs*200.)
        Hz = 10
        self.dat200 = N.sin(N.arange(numPoints,dtype=N.float)*2*N.pi*Hz*numSecs/numPoints)

        numSecs = 4.
        numPoints = int(numSecs*50.)
        Hz = 10
        self.dat50 = N.sin(N.arange(numPoints,dtype=N.float)*2*N.pi*Hz*numSecs/numPoints)

    def test_init(self):
        # init a TimeSeries with all combos of options and verify that
        # the attributes are correct
        
        # fewest params
        srate = 200
        durMS = 4000
        ts = EegTimeSeries(self.dat200,srate)
        N.testing.assert_equal(ts.data[:], self.dat200[:])
        self.assertEquals(ts.shape, self.dat200.shape)
        self.assertEquals(ts.ndim, len(self.dat200.shape))
        self.assertEquals(ts.tdim, len(self.dat200.shape)-1)
        self.assertEquals(ts.buffer, 0)
        self.assertEquals(ts.bufferMS, 0)
        self.assertEquals(ts.offset, 0)
        self.assertEquals(ts.offsetMS, 0)
        self.assertEquals(ts.durationMS, durMS)
        N.testing.assert_equal(ts.trangeMS[:], N.linspace(0,durMS,self.dat200.shape[-1]))

        # with offset and buffer
        srate = 200
        buffer = 200
        bufferMS = 1000
        offset = -50
        offsetMS = offset*1000./srate
        durMS = 4000-2*bufferMS
        ts = EegTimeSeries(self.dat200,srate,offset=offset,buffer=buffer)
        N.testing.assert_equal(ts.data[:], self.dat200[:])
        self.assertEquals(ts.shape, self.dat200.shape)
        self.assertEquals(ts.ndim, len(self.dat200.shape))
        self.assertEquals(ts.tdim, len(self.dat200.shape)-1)
        self.assertEquals(ts.buffer, buffer)
        self.assertEquals(ts.bufferMS, bufferMS)
        self.assertEquals(ts.offset, offset)
        self.assertEquals(ts.offsetMS, offsetMS)
        self.assertEquals(ts.durationMS, durMS)
        N.testing.assert_equal(ts.trangeMS[:], N.linspace(offsetMS-bufferMS,offsetMS+durMS+bufferMS,self.dat200.shape[-1]))

        # with offsetMS and bufferMS
        srate = 200
        buffer = 200
        bufferMS = 1000
        offset = -50
        offsetMS = offset*1000./srate
        durMS = 4000-2*bufferMS
        ts = EegTimeSeries(self.dat200,srate,offsetMS=offsetMS,bufferMS=bufferMS)
        N.testing.assert_equal(ts.data[:], self.dat200[:])
        self.assertEquals(ts.shape, self.dat200.shape)
        self.assertEquals(ts.ndim, len(self.dat200.shape))
        self.assertEquals(ts.tdim, len(self.dat200.shape)-1)
        self.assertEquals(ts.buffer, buffer)
        self.assertEquals(ts.bufferMS, bufferMS)
        self.assertEquals(ts.offset, offset)
        self.assertEquals(ts.offsetMS, offsetMS)
        self.assertEquals(ts.durationMS, durMS)
        N.testing.assert_equal(ts.trangeMS[:], N.linspace(offsetMS-bufferMS,offsetMS+durMS+bufferMS,self.dat200.shape[-1]))

        
        
    def test_removeBuffer(self):
        pass
    def test_getitem(self):
        pass
    def test_setitem(self):
        pass
    def test_filter(self):
        pass
    def test_resample(self):
        pass
    def test_decimate(self):
        pass

# test RawBinaryEEG

# load data from file

# with resampling

# without resampling

# make sure the time range is correct

# make sure we get the expected number of samples



