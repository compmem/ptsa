import numpy as N
import re
from numpy.testing import NumpyTest, NumpyTestCase

from pyeeg.data import Dim,Dims,DimData
from pyeeg import filter


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

class TestData():
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
        
        
        numSecs = 20.
        numPoints = int(numSecs*200.)
        numSubj = 3
        numChans = 2
        self.randData3D = N.random.random_sample((numSubj,numChans,numPoints))
        # calc the time range in MS
        offset = -200
        duration = numPoints
        samplesize = 1000./200.
        sampStart = offset*samplesize
        sampEnd = sampStart + (duration-1)*samplesize
        timeRange = N.linspace(sampStart,sampEnd,duration)
        self.randDims3D = [Dim('subject',N.arange(self.randData3D.shape[0])),
                           Dim('channel',N.arange(self.randData3D.shape[1])),
                           Dim('time',timeRange,'ms')]
        
        numSecs = 20.
        numPoints = int(numSecs*200.)
        numSubj = 10
        numChans = 5
        numConds = 4
        self.randData4D = N.random.random_sample((numSubj,numChans,numConds,numPoints))
        # calc the time range in MS
        offset = -200
        duration = numPoints
        samplesize = 1000./200.
        sampStart = offset*samplesize
        sampEnd = sampStart + (duration-1)*samplesize
        timeRange = N.linspace(sampStart,sampEnd,duration)
        self.randDims4D = [Dim('subject',N.arange(self.randData4D.shape[0])),
                           Dim('channel',N.arange(self.randData4D.shape[1])),
                           Dim('condition',N.arange(self.randData4D.shape[2])),
                           Dim('time',timeRange,'ms')]
        


# test Dim
class test_Dim(NumpyTestCase):
    def setUp(self):
        td = TestData()
        self.dat200 = td.dat200
        self.dims200 = td.dims200
        self.dat50 = td.dat50
        self.dims50 = td.dims50

    def test_init(self):
        name = "test 1"
        data = self.dat200
        dim = Dim(name,data)
        self.assertEquals(dim.name,name)
        N.testing.assert_array_almost_equal(dim.data,data)
        name = "Test 2"
        data = self.dat50
        units = "ms"
        dim = Dim(name,data,units)
        self.assertEquals(dim.name,name)
        N.testing.assert_array_almost_equal(dim.data,data)
        self.assertEquals(dim.units,units)
       
    def test_copy(self):
        name = "test 1"
        data = self.dat200
        dim1 = Dim(name,data)
        dim2 = dim1.copy()
        self.assertEquals(dim1.name,dim2.name)
        N.testing.assert_array_almost_equal(dim1.data,dim2.data)
        name = "Test 2"
        data = self.dat50
        units = "ms"
        dim1 = Dim(name,data,units)
        dim2 = dim1.copy()
        self.assertEquals(dim1.name,dim2.name)
        N.testing.assert_array_almost_equal(dim1.data,dim2.data)
        self.assertEquals(dim1.units,dim2.units)
        
    def test_extend(self):
        # CTW: Perhaps it would be better to just have a place holder for this function and require subclasses to specify this class in a way that takes sampling rate into account.
        pass
#         name1 = "test 1"
#         data1 = self.dat200
#         units1 = "ms"
#         dim1 = Dim(name1,data1,units1)
#         name2 = "test 2"
#         data2 = self.dat50
#         units2 = "ms"
#         dim2 = Dim(name2,data2,units2)
#         dim12  = Dim(name1,N.concatenate((dim1.data,dim2.data),axis=0),units1)
#         dim12e = dim1.extend(dim2)
#         self.assertEquals(dim12,dim12e)

    def test_getitem(self):
        name = "test 1"
        data = self.dat200
        dim1 = Dim(name,data)
        N.testing.assert_array_equal(data[20:30], dim1[20:30])

    def test_setitem(self):
        name = "test 1"
        data = self.dat200
        dim = Dim(name,data)
        dim[10:20] = data[20:30]
        N.testing.assert_array_equal(dim.data[10:20], data[20:30])
        
    def test_comparisons(self):
        name1 = "test 1"
        data1 = self.dat200.copy()
        dim1 = Dim(name1,data1)
        name2 = "test 2"
        data2 = self.dat200.copy()
        dim2 = Dim(name2,data2)
        new_vals = N.random.random_sample(N.shape(dim1.data))
        for index, val in enumerate(new_vals):
            dim1[index] = val.copy()
            dim2[index] = val.copy()
            self.assertEquals(sum(dim1[index]==dim2[index]),
                              N.prod(N.shape(dim1[index])))
            self.assertEquals(sum(dim1[index]>=dim2[index]),
                              N.prod(N.shape(dim1[index])))
            self.assertEquals(sum(dim1[index]<=dim2[index]),
                              N.prod(N.shape(dim1[index])))
            self.assertEquals(sum((dim1[index]!=dim2[index])+1),
                              N.prod(N.shape(dim1[index])))
            self.assertEquals(sum((dim1[index]>dim2[index])+1), N.prod(N.shape(dim1[index])))
            self.assertEquals(sum((dim1[index]<dim2[index])+1), N.prod(N.shape(dim1[index])))

# test Dims
class test_Dims(NumpyTestCase):
    def setUp(self):
        td = TestData()
        self.dat200 = td.dat200
        self.dims200 = td.dims200
        self.dat50 = td.dat50
        self.dims50 = td.dims50

    def test_init(self):
        test1 = Dims(self.dims50)
        N.testing.assert_array_equal(test1.names, [dim.name for dim in self.dims50])
        N.testing.assert_array_equal(test1._namesRE, re.compile('\\b'+'\\b|\\b'.join(test1.names)+'\\b'))
        N.testing.assert_array_equal(test1._nameOnlyRE, re.compile('(?<!.)\\b' + '\\b(?!.)|(?<!.)\\b'.join(test1.names) + '\\b(?!.)'))
        N.testing.assert_array_equal(test1.dims,self.dims50)
        
        test2 = Dims(self.dims200)
        N.testing.assert_array_equal(test2.names, [dim.name for dim in self.dims50])
        N.testing.assert_array_equal(test2._namesRE, re.compile('\\b'+'\\b|\\b'.join(test2.names)+'\\b'))
        N.testing.assert_array_equal(test2._nameOnlyRE, re.compile('(?<!.)\\b' + '\\b(?!.)|(?<!.)\\b'.join(test2.names) + '\\b(?!.)'))
        N.testing.assert_array_equal(test2.dims,self.dims200)

    def test_index(self):
        test1 = Dims(self.dims50)
        self.assertEquals(test1.index('channel'), 0)
        self.assertEquals(test1.index('time'), 1)

    def test_copy(self):
        test1 = Dims(self.dims50)
        test2 = test1.copy()
        #N.testing.assert_array_equal([dim.name for dim in test1.dims],[dim.name for dim in test2.dims])
        for index,dim in enumerate(test1.dims):
            N.testing.assert_array_equal(dim.name,test2.dims[index].name)
            N.testing.assert_array_equal(dim.data,test2.dims[index].data)
            N.testing.assert_array_equal(dim.units,test2.dims[index].units)
        #N.testing.assert_array_equal([dim.units for dim in test1.dims],[dim.units for dim in test2.dims])
        N.testing.assert_array_equal(test1.names,test2.names)
        N.testing.assert_array_equal(test1._namesRE,test2._namesRE)
        N.testing.assert_array_equal(test1._nameOnlyRE,test2._nameOnlyRE)
        N.testing.assert_array_equal([test1.index(name) for name in test1.names],[test2.index(name) for name in test2.names])
        

    def test_getitem(self):
        test1 = Dims(self.dims50)
        test2 = test1[:]
        for index,dim in enumerate(test1.dims):        
            N.testing.assert_array_equal(dim.name,test2[index].name)
            N.testing.assert_array_equal(dim.data,test2[index].data)
            N.testing.assert_array_equal(dim.units,test2[index].units)
        test3 = [test1.__getitem__(i) for i in range(len(test1.dims))]
        for index,dim in enumerate(test1.dims):        
            N.testing.assert_array_equal(dim.name,test3[index].name)
            N.testing.assert_array_equal(dim.data,test3[index].data)
            N.testing.assert_array_equal(dim.units,test3[index].units)

    def test_setitem(self):
        test1 = Dims(self.dims50)
        test1[0] = Dim('newDim',N.arange(5),'newUnits')
        self.assertEqual(test1[0].name,'newDim')
        self.assertEqual(test1[0].units,'newUnits')
        self.assertEqual(len(test1[0].data),5)

    def test_iter(self):
        #CTW: Not sure how best to test this. Also, this function just calls the __iter__() function for lists, so a test here is probably not necessary.
        pass

# test DimData
class test_DimData(NumpyTestCase):
    def setUp(self):
        td = TestData()
        self.dat200 = td.dat200
        self.dims200 = td.dims200
        self.dat50 = td.dat50
        self.dims50 = td.dims50
        self.randData3D = td.randData3D
        self.randDims3D = td.randDims3D
        self.randData4D = td.randData4D
        self.randDims4D = td.randDims4D
    
    def test_init(self):
        test1 = DimData(self.dat200,self.dims200,unit='dimDatUnit')
        N.testing.assert_array_equal(test1.data,self.dat200)
        for index,dim in enumerate(test1.dims):        
            N.testing.assert_array_equal(dim.name,self.dims200[index].name)
            N.testing.assert_array_equal(dim.data,self.dims200[index].data)
            N.testing.assert_array_equal(dim.units,self.dims200[index].units)
        self.assertEqual(test1.unit,'dimDatUnit')
        self.assertEqual(test1.dtype,self.dat200.dtype)
        self.assertEqual(test1.shape,self.dat200.shape)
        self.assertEqual(test1.ndim,len(test1.shape))
        
        test2 = DimData(self.dat200,Dims(self.dims200),unit='dimDatUnit')
        N.testing.assert_array_equal(test2.data,self.dat200)
        for index,dim in enumerate(test2.dims):        
            N.testing.assert_array_equal(dim.name,self.dims200[index].name)
            N.testing.assert_array_equal(dim.data,self.dims200[index].data)
            N.testing.assert_array_equal(dim.units,self.dims200[index].units)
        self.assertEqual(test2.unit,'dimDatUnit')
        self.assertEqual(test2.dtype,self.dat200.dtype)
        self.assertEqual(test2.shape,self.dat200.shape)
        self.assertEqual(test2.ndim,len(test2.shape))

        self.assertRaises(ValueError,DimData,self.dat200.reshape((2,2,400)),self.dims200)
        self.assertRaises(ValueError,DimData,self.dat200.reshape((2,2,400)),Dims(self.dims200))
        self.assertRaises(ValueError,DimData,self.dat200[:,0:799],self.dims200)

    def test_copy(self):
        test1 = DimData(self.dat200,self.dims200,unit='dimDatUnit')
        test2 = test1.copy()
        N.testing.assert_array_equal(test1.data,test2.data)
        for index,dim in enumerate(test1.dims.dims):        
            N.testing.assert_array_equal(dim.name,test2.dims.dims[index].name)
            N.testing.assert_array_equal(dim.data,test2.dims.dims[index].data)
            N.testing.assert_array_equal(dim.units,test2.dims.dims[index].units)
        N.testing.assert_array_equal(test1.data,test2.data)
        self.assertEqual(test1.unit,test2.unit)
        self.assertEqual(test1.dtype,test2.dtype)
        self.assertEqual(test1.shape,test2.shape)
        self.assertEqual(test1.ndim,test2.ndim)
        
    def test_dim(self):
        test1 = DimData(self.dat200,self.dims200)
        self.assertEqual(test1.dim('channel'),0)
        self.assertEqual(test1.dim('time'),1)
        
    def test_getitem(self):
        test1 = DimData(self.dat200,self.dims200)
        N.testing.assert_array_equal(test1[0],test1.data[0])
        N.testing.assert_array_equal(test1[1],test1.data[1])
        N.testing.assert_array_equal(test1['channel'],test1.dims[0].data)
        N.testing.assert_array_equal(test1['time'],test1.dims[1].data)
        #CTW: How can isinstance(item,tuple) be true?
        #self.assert(((min(test1['time>0']['time']))>0),True)
        #print ((min(test1['time>0']['time']))>0)
        self.assertEquals(((min(test1.select('time>0')['time']))>0),True)
        #CTW: Is it necessary to test this with __getitem__(), too? Probably not.
        
    def test_setitem(self):
        test1 = DimData(self.dat200,self.dims200)
        newVal = N.random.random_sample(N.shape(test1.data[0]))
        test1[0] = newVal
        N.testing.assert_array_equal(test1.data[0],newVal)
        
    def test_find(self):
        #CTW: minimal test ... there probably should be more 
        test1 = DimData(self.dat200,self.dims200)
        filterstring = 'time>0'
        array1 = test1.select(filterstring).data
        array2 = test1.data[test1.find(filterstring)]
        N.testing.assert_array_equal(array1,array2)
    
    def test_data_ind(self):
        #CTW: minimal test ... there probably should be more 
        test1 = DimData(self.dat200,self.dims200)
        filterstring = 'time>0'
        array1 = test1.data[test1.find(filterstring)]
        array2 = test1.data[test1.data_ind(filterstring)].reshape(array1.shape)
        N.testing.assert_array_equal(array1,array2)
    
    def test_select(self):
        #CTW: minimal test ... there probably should be more 
        test1 = DimData(self.dat200,self.dims200)
        self.assertEquals(((min(test1.select('time>0')['time']))>0),True)
    
    def test_extend(self):
        #CTW: Let's chat about this function. I think it's a bit dangerous, because it lets you do crazy stuff without complaining.
        pass
    
    def test_aggregate(self):
        # A lot of these tests will only work for certain functions
        # (such as N.mean) where the result is not influenced by the
        # order in which the aggregation is done. For some other
        # functions (such as N.std) different orders of aggregation
        # will produce different results (cf. docstring).
        test1 = DimData(self.dat200,self.dims200)
        test1_pos = test1.aggregate([],N.mean)
        N.testing.assert_array_equal(test1.data,test1_pos.data)
        N.testing.assert_array_equal(test1.dims.names,test1_pos.dims.names)
        test1_neg = test1.aggregate(['channel','time'],N.mean,dimval=False)
        N.testing.assert_array_equal(test1.data,test1_neg.data)
        N.testing.assert_array_equal(test1.dims.names,test1_neg.dims.names)

        test2a = test1.aggregate(['channel'],N.mean)
        dimArray = []
        for i,dim in enumerate(test1.dims):
            if dim.name!='channel':
                dimArray.append(dim)
        test2b = DimData(N.mean(test1.data,test1.dim('channel')),dimArray)
        test2c = test1.aggregate(['time'],N.mean,dimval=False)
        N.testing.assert_array_equal(test2a.data,test2b.data)
        N.testing.assert_array_equal(test2a.dims.names,test2b.dims.names)
        N.testing.assert_array_equal(test2a.data,test2c.data)
        N.testing.assert_array_equal(test2a.dims.names,test2c.dims.names)
        
        test3a = test1.aggregate(['time'],N.mean)
        dimArray = []
        for i,dim in enumerate(test1.dims):
            if dim.name!='time':
                dimArray.append(dim)
        test3b = DimData(N.mean(test1.data,test1.dim('time')),dimArray)
        test3c = test1.aggregate(['channel'],N.mean,dimval=False)
        N.testing.assert_array_equal(test3a.data,test3b.data)
        N.testing.assert_array_equal(test3a.dims.names,test3b.dims.names)
        N.testing.assert_array_equal(test3a.data,test3c.data)
        N.testing.assert_array_equal(test3a.dims.names,test3c.dims.names)
        
        test4a = test2a.aggregate(['time'],N.mean)
        test4b = test2b.aggregate(['time'],N.mean)
        test4c = test2b.aggregate([test2b.dim('time')],N.mean)
        test5a = test3a.aggregate(['channel'],N.mean)
        test5b = test3b.aggregate(['channel'],N.mean)
        test5c = test3a.aggregate([test3a.dim('channel')],N.mean)
        test6a = test1.aggregate(['channel','time'],N.mean)
        test6b = test1.aggregate(['time','channel'],N.mean)
        test6c = test1.aggregate([test1.dim('time'),test1.dim('channel')],N.mean)
        test6d = test1.aggregate([test1.dim('channel'),test1.dim('time')],N.mean)
        test7 = test1.aggregate([],N.mean,dimval=False)
        
        self.assertRaises(ValueError,test1.aggregate,[test1.dim('channel'),'time'],N.mean)
        
        N.testing.assert_array_equal(test4a.data,test4b.data)
        N.testing.assert_array_equal(test4a.dims.names,test4b.dims.names)
        N.testing.assert_array_equal(test4a.data,test4c.data)
        N.testing.assert_array_equal(test4a.dims.names,test4c.dims.names)
        N.testing.assert_array_equal(test5a.data,test5b.data)
        N.testing.assert_array_equal(test5a.dims.names,test5b.dims.names)
        N.testing.assert_array_equal(test5a.data,test5c.data)
        N.testing.assert_array_equal(test5a.dims.names,test5c.dims.names)
        N.testing.assert_array_almost_equal(test6a.data,test6b.data)
        N.testing.assert_array_equal(test6a.dims.names,test6b.dims.names)
        N.testing.assert_array_almost_equal(test6a.data,test6c.data)
        N.testing.assert_array_equal(test6a.dims.names,test6c.dims.names)
        N.testing.assert_array_almost_equal(test6a.data,test6d.data)
        N.testing.assert_array_equal(test6a.dims.names,test6d.dims.names)
        N.testing.assert_array_almost_equal(test4a.data,test5a.data)
        N.testing.assert_array_equal(test4a.dims.names,test5a.dims.names)
        N.testing.assert_array_almost_equal(test4b.data,test6a.data)
        N.testing.assert_array_equal(test4b.dims.names,test6a.dims.names)
        N.testing.assert_array_almost_equal(test6a.data,test7.data)
        N.testing.assert_array_almost_equal(test6a.dims.names,test7.dims.names)

        test4D = DimData(self.randData4D,self.randDims4D)
        test4D_pos = test4D.aggregate([],N.mean)
        N.testing.assert_array_equal(test4D.data,test4D_pos.data)
        N.testing.assert_array_equal(test4D.dims.names,test4D_pos.dims.names)
        test4D_neg = test4D.aggregate(['subject','channel','condition','time'],
                                    N.mean,dimval=False)
        N.testing.assert_array_equal(test4D.data,test4D_neg.data)
        N.testing.assert_array_equal(test4D.dims.names,test4D_neg.dims.names)

        test4D_2a = test4D.aggregate(['condition'],N.mean)
        dimArray = []
        for i,dim in enumerate(test4D.dims):
            if dim.name!='condition':
                dimArray.append(dim)
        test4D_2b = DimData(N.mean(test4D.data,test4D.dim('condition')),dimArray)
        test4D_2c = test4D.aggregate(['subject','channel','time'],N.mean,dimval=False)
        N.testing.assert_array_equal(test4D_2a.data,test4D_2b.data)
        N.testing.assert_array_equal(test4D_2a.dims.names,test4D_2b.dims.names)
        N.testing.assert_array_equal(test4D_2a.data,test4D_2c.data)
        N.testing.assert_array_equal(test4D_2a.dims.names,test4D_2c.dims.names)

        test4D_3a = test4D.aggregate(['channel','condition'],N.mean)
        dimArray = []
        for i,dim in enumerate(test4D.dims):
            if (dim.name!='channel') and (dim.name!='condition'):
                dimArray.append(dim)
        test4D_3b = DimData(N.mean(N.mean(test4D.data,test4D.dim('condition')),test4D.dim('channel')),dimArray)
        test4D_3c = test4D.aggregate(['subject','time'],N.mean,dimval=False)
        N.testing.assert_array_equal(test4D_3a.data,test4D_3b.data)
        N.testing.assert_array_equal(test4D_3a.dims.names,test4D_3b.dims.names)
        N.testing.assert_array_equal(test4D_3a.data,test4D_3c.data)
        N.testing.assert_array_equal(test4D_3a.dims.names,test4D_3c.dims.names)

        
    def test_margin(self):
        #test1 = DimData(self.dat200,self.dims200)
        test1 = DimData(self.randData3D,self.randDims3D)
        test1_1d_a = test1.margin('channel',N.mean)
        test1_nd_a = test1.aggregate('channel',N.mean,dimval=False)
        N.testing.assert_array_almost_equal(test1_1d_a.data,test1_nd_a.data)
        self.assertEqual(test1_1d_a.dims[0].name,'channel')
        N.testing.assert_array_equal(test1_1d_a.dims['channel'].data,test1.dims['channel'].data)
        N.testing.assert_array_equal(test1_1d_a.dims['channel'].data,test1_nd_a.dims['channel'].data)
        
        test1_1d_b = test1.margin('subject',N.mean)
        test1_nd_b = test1.aggregate('subject',N.mean,dimval=False)
        N.testing.assert_array_almost_equal(test1_1d_b.data,test1_nd_b.data)
        self.assertEqual(test1_1d_b.dims[0].name,'subject')
        N.testing.assert_array_equal(test1_1d_b.dims['subject'].data,test1.dims['subject'].data)
        N.testing.assert_array_equal(test1_1d_b.dims['subject'].data,test1_nd_b.dims['subject'].data)

        test1_1d_c = test1.margin('channel',N.std)
        test1_nd_c = test1.aggregate('time',N.std)
        test1_nd_c = test1_nd_c.aggregate('channel',N.mean,dimval=False)
        N.testing.assert_array_equal(N.round(test1_1d_c.data,2),N.round(test1_nd_c.data,2))
        self.assertEqual(test1_1d_c.dims[0].name,'channel')
        N.testing.assert_array_equal(test1_1d_c.dims['channel'].data,test1.dims['channel'].data)
        N.testing.assert_array_equal(test1_1d_c.dims['channel'].data,test1_nd_c.dims['channel'].data)
        
        test1_1d_d = test1.margin('subject',N.std)
        test1_nd_d = test1.aggregate('time',N.std)
        test1_nd_d = test1_nd_d.aggregate('subject',N.mean,dimval=False)
        N.testing.assert_array_equal(N.round(test1_1d_d.data,2),N.round(test1_nd_d.data,2))
        self.assertEqual(test1_1d_d.dims[0].name,'subject')
        N.testing.assert_array_equal(test1_1d_d.dims['subject'].data,test1.dims['subject'].data)
        N.testing.assert_array_equal(test1_1d_d.dims['subject'].data,test1_nd_d.dims['subject'].data)

        
        test2 = DimData(self.randData4D,self.randDims4D)
        test2_1d_a = test2.margin('channel',N.mean)
        test2_1d_a2= test2.margin(test2.dim('channel'),N.mean)
        test2_nd_a = test2.aggregate('channel',N.mean,dimval=False)
        N.testing.assert_array_almost_equal(test2_1d_a.data,test2_nd_a.data)
        N.testing.assert_array_almost_equal(test2_1d_a.data,test2_1d_a2.data)
        self.assertEqual(test2_1d_a.dims[0].name,'channel')
        self.assertEqual(test2_1d_a2.dims[0].name,'channel')
        N.testing.assert_array_equal(test2_1d_a.dims['channel'].data,test2.dims['channel'].data)
        N.testing.assert_array_equal(test2_1d_a.dims['channel'].data,test2_nd_a.dims['channel'].data)
        N.testing.assert_array_equal(test2_1d_a2.dims['channel'].data,test2.dims['channel'].data)
        N.testing.assert_array_equal(test2_1d_a2.dims['channel'].data,test2_nd_a.dims['channel'].data)

        test2_1d_b = test2.margin('subject',N.mean)
        test2_nd_b = test2.aggregate('subject',N.mean,dimval=False)
        N.testing.assert_array_almost_equal(test2_1d_b.data,test2_nd_b.data)
        self.assertEqual(test2_1d_b.dims[0].name,'subject')
        N.testing.assert_array_equal(test2_1d_b.dims['subject'].data,test2.dims['subject'].data)
        N.testing.assert_array_equal(test2_1d_b.dims['subject'].data,test2_nd_b.dims['subject'].data)
        
        test2_1d_c = test2.margin('condition',N.mean)
        test2_nd_c = test2.aggregate('condition',N.mean,dimval=False)
        N.testing.assert_array_almost_equal(test2_1d_c.data,test2_nd_c.data)
        self.assertEqual(test2_1d_c.dims[0].name,'condition')
        N.testing.assert_array_equal(test2_1d_c.dims['condition'].data,test2.dims['condition'].data)
        N.testing.assert_array_equal(test2_1d_c.dims['condition'].data,test2_nd_c.dims['condition'].data)

        test2_1d_d = test2.margin('channel',N.std)
        test2_nd_d = test2.aggregate('time',N.std)
        test2_nd_d = test2_nd_d.aggregate('channel',N.mean,dimval=False)
        N.testing.assert_array_equal(N.round(test2_1d_d.data,2),N.round(test2_nd_d.data,2))
        self.assertEqual(test2_1d_d.dims[0].name,'channel')
        N.testing.assert_array_equal(test2_1d_d.dims['channel'].data,test2.dims['channel'].data)
        N.testing.assert_array_equal(test2_1d_d.dims['channel'].data,test2_nd_d.dims['channel'].data)

        test2_1d_e = test2.margin('subject',N.std)
        test2_nd_e = test2.aggregate('time',N.std)
        test2_nd_e = test2_nd_e.aggregate('subject',N.mean,dimval=False)
        N.testing.assert_array_equal(N.round(test2_1d_e.data,2),N.round(test2_nd_e.data,2))
        self.assertEqual(test2_1d_e.dims[0].name,'subject')
        N.testing.assert_array_equal(test2_1d_e.dims['subject'].data,test2.dims['subject'].data)
        N.testing.assert_array_equal(test2_1d_e.dims['subject'].data,test2_nd_e.dims['subject'].data)

        test2_1d_f = test2.margin('condition',N.std)
        test2_nd_f = test2.aggregate('time',N.std)
        test2_nd_f = test2_nd_f.aggregate('condition',N.mean,dimval=False)
        N.testing.assert_array_equal(N.round(test2_1d_f.data,2),N.round(test2_nd_f.data,2))
        self.assertEqual(test2_1d_f.dims[0].name,'condition')
        N.testing.assert_array_equal(test2_1d_f.dims['condition'].data,test2.dims['condition'].data)
        N.testing.assert_array_equal(test2_1d_f.dims['condition'].data,test2_nd_f.dims['condition'].data)

        

# test RawBinaryEEG

# load data from file

# make sure the time range is correct

# make sure we get the expected number of samples


if __name__ == '__main__':
    NumpyTest.main()
