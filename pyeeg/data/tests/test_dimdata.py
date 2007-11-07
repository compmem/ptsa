import numpy as N
import re
from numpy.testing import NumpyTest, NumpyTestCase

from pyeeg.data import Dim,Dims,DimData
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
        dim2 = dim1[:]
        self.assertEquals(dim1.name,dim2.name)
        N.testing.assert_array_almost_equal(dim1.data,dim2.data)
        self.assertEquals(dim1.units,dim2.units)
        dim3 = dim1.__getitem__(range(len(dim1.data)))
        self.assertEquals(dim1.name,dim3.name)
        self.assertEquals(dim2.name,dim3.name)
        N.testing.assert_array_almost_equal(dim1.data,dim3.data)
        N.testing.assert_array_almost_equal(dim2.data,dim3.data)
        self.assertEquals(dim1.units,dim3.units)
        self.assertEquals(dim2.units,dim3.units)

    def test_setitem(self):
        name = "test 1"
        data = self.dat200
        dim = Dim(name,data)
        new_vals = N.random.random_sample(N.shape(dim.data))
        for index, val in enumerate(new_vals):
            dim[index] = val
            N.testing.assert_array_almost_equal(dim[index].data,new_vals[index])
        new_vals = N.random.random_sample(N.shape(dim.data))
        for index, val in enumerate(new_vals):
            dim.__setitem__(index,val)
            N.testing.assert_array_almost_equal(dim[index].data,new_vals[index])
        
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
            self.assertEquals(sum(dim1[index].data==dim2[index].data), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data>=dim2[index].data), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data<=dim2[index].data), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum((dim1[index].data!=dim2[index].data)+1), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum((dim1[index].data>dim2[index].data)+1), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum((dim1[index].data<dim2[index].data)+1), N.prod(N.shape(dim1[index].data)))
            
            self.assertEquals(sum(dim1[index].data.__eq__(dim2[index].data)), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data.__ge__(dim2[index].data)), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data.__le__(dim2[index].data)), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum((dim1[index].data.__ne__(dim2[index].data))+1), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum((dim1[index].data.__gt__(dim2[index].data))+1), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum((dim1[index].data.__lt__(dim2[index].data))+1), N.prod(N.shape(dim1[index].data)))
            
            dim2[index] = val+1
            self.assertEquals(sum((dim1[index].data==dim2[index].data)+1), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum((dim1[index].data>=dim2[index].data)+1), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data<=dim2[index].data), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data!=dim2[index].data), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum((dim1[index].data>dim2[index].data)+1), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data<dim2[index].data), N.prod(N.shape(dim1[index].data)))
            
            self.assertEquals(sum((dim1[index].data.__eq__(dim2[index].data))+1), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum((dim1[index].data.__ge__(dim2[index].data))+1), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data.__le__(dim2[index].data)), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data.__ne__(dim2[index].data)), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum((dim1[index].data.__gt__(dim2[index].data))+1), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data.__lt__(dim2[index].data)), N.prod(N.shape(dim1[index].data)))
            
            dim2[index] = val-1
            self.assertEquals(sum((dim1[index].data==dim2[index].data)+1), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data>=dim2[index].data), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum((dim1[index].data<=dim2[index].data)+1), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data!=dim2[index].data), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data>dim2[index].data), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum((dim1[index].data<dim2[index].data)+1), N.prod(N.shape(dim1[index].data)))      

            self.assertEquals(sum((dim1[index].data.__eq__(dim2[index].data))+1), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data.__ge__(dim2[index].data)), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum((dim1[index].data.__le__(dim2[index].data))+1), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data.__ne__(dim2[index].data)), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum(dim1[index].data.__gt__(dim2[index].data)), N.prod(N.shape(dim1[index].data)))
            self.assertEquals(sum((dim1[index].data.__lt__(dim2[index].data))+1), N.prod(N.shape(dim1[index].data)))      

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
        N.testing.assert_array_equal(test1.namesRE, re.compile('\\b'+'\\b|\\b'.join(test1.names)+'\\b'))
        N.testing.assert_array_equal(test1.nameOnlyRE, re.compile('(?<!.)\\b' + '\\b(?!.)|(?<!.)\\b'.join(test1.names) + '\\b(?!.)'))
        N.testing.assert_array_equal(test1.dims,self.dims50)
        
        test2 = Dims(self.dims200)
        N.testing.assert_array_equal(test2.names, [dim.name for dim in self.dims50])
        N.testing.assert_array_equal(test2.namesRE, re.compile('\\b'+'\\b|\\b'.join(test2.names)+'\\b'))
        N.testing.assert_array_equal(test2.nameOnlyRE, re.compile('(?<!.)\\b' + '\\b(?!.)|(?<!.)\\b'.join(test2.names) + '\\b(?!.)'))
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
        N.testing.assert_array_equal(test1.namesRE,test2.namesRE)
        N.testing.assert_array_equal(test1.nameOnlyRE,test2.nameOnlyRE)
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
        #CTW: Not sure how best to test this. Also, this function just calls teh __iter__() function for lists, so a test here is probably not necessary.
        pass

# test DimData
class test_DimData(NumpyTestCase):
    def setUp(self):
        td = TestData()
        self.dat200 = td.dat200
        self.dims200 = td.dims200
        self.dat50 = td.dat50
        self.dims50 = td.dims50
    
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
        self.assertEquals(((min(test1['time>0']['time']))>0),True)
        #CTW: Is it necessary to test this with __getitem__(), too? Probably not.
        
    def test_setitem(self):
        test1 = DimData(self.dat200,self.dims200)
        newVal = N.random.random_sample(N.shape(test1.data[0]))
        test1[0] = newVal
        N.testing.assert_array_equal(test1.data[0],newVal)
        
    def test_select(self):
        #CTW: minimal test ... there probably should be more 
        test1 = DimData(self.dat200,self.dims200)
        self.assertEquals(((min(test1['time>0']['time']))>0),True)
    
    def test_extend(self):
        #CTW: Let's chat about this function. I think it's a bit dangerous, because it lets you do crazy stuff without complaining.
        pass


# test RawBinaryEEG

# load data from file

# make sure the time range is correct

# make sure we get the expected number of samples


if __name__ == '__main__':
    NumpyTest.main()
