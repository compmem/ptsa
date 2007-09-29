import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

from pyeeg import Dim,Dims,DimData,EegTimeSeries

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

# test Dim
class test_Dim(NumpyTestCase):
    def test_init(self):
        pass
    def test_copy(self):
        pass
    def test_extend(self):
        pass
    def test_getitem(self):
        pass
    def test_setitem(self):
        pass
    def test_comparisons(self):
        pass

# test Dims
class test_Dims(NumpyTestCase):
    def test_init(self):
        pass
    def test_copy(self):
        pass
    def test_index(self):
        pass
    def test_getitem(self):
        pass
    def test_setitem(self):
        pass
    def test_iter(self):
        pass

# test DimData
class test_DimData(NumpyTestCase):
    def test_init(self):
        pass
    def test_copy(self):
        pass
    def test_dim(self):
        pass
    def test_getitem(self):
        pass
    def test_setitem(self):
        pass
    def test_select(self):
        pass
    def test_extend(self):
        pass


# test EegTimeSeries
class test_EegTimeSeries(NumpyTestCase):
    def setUp(self): 
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

    def test_getitem(self):
        pass

    def test_setitem(self):
        pass
#         ts = EegTimeSeries(self.dat200,200)
#         x = N.arange(10)
#         ts[11:11+10] = x
#         N.testing.assert_equal(ts[11:11+10],x[:])

    def test_filter(self):
        pass

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
