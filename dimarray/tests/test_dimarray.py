#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from numpy.testing import NumpyTest, NumpyTestCase, assert_array_equal,\
     assert_array_almost_equal
from numpy.random import random_sample as rnd

from dimarray import DimArray, Dim
from dimarray import AttrArray



# Dim class
class test_Dim(NumpyTestCase):
    def setUp(self):
        pass
    
    def test_new(self):
        # should throw TypeError if no name is specified:
        self.assertRaises(TypeError,Dim,range(3))
        # should throw ValueError if not 1-D
        self.assertRaises(ValueError,Dim,rnd((2,3)),name='test')
        # should work fine with any number of dimensions as long as it
        # is squeezable or expandable to 1-D:
        tst = Dim(rnd((3,1,1,1,1)),name='test')
        self.assertEquals(tst.name,'test')
        tst = Dim(np.array(5),name='test2')
        self.assertEquals(tst.name,'test2')
        # custom attributes should work, too:
        tst = Dim(range(2),name='test3',custom='attribute')
        self.assertEquals(tst.name,'test3')
        self.assertEquals(tst.custom,'attribute')


# DimArray class
class test_DimArray(NumpyTestCase):
    def setUp(self):
        pass
    
    def test_new(self):
        # should throw Error if dims are not specified:
        self.assertRaises(TypeError,DimArray,np.random.rand(5,10))
        # should throw ValueError if dims is not a list:
        self.assertRaises(AttributeError,DimArray,np.random.rand(5,10),
                          dims = np.arange(4))

        # should throw Error if dims contains non-Dim instances:
        self.assertRaises(AttributeError,DimArray,np.random.rand(5,10),
                          dims=[Dim(range(5),name='freqs',unit='Hz'),
                                AttrArray(range(10),name='time',unit='sec')])
        self.assertRaises(AttributeError,DimArray,np.random.rand(5,10),
                          dims=[AttrArray(range(5),name='freqs',unit='Hz'),
                                Dim(range(10),name='time',unit='sec')])
 
        # should throw Error if dims do not match data shape:
        self.assertRaises(AttributeError,DimArray,np.random.rand(5,10),
                          dims=[Dim(range(10),name='freqs',unit='Hz'),
                                Dim(range(5),name='time',unit='sec')])
        self.assertRaises(AttributeError,DimArray,np.random.rand(5,10),
                          dims=[Dim(range(5),name='freqs',unit='Hz')])

        # should throw Error if 2 dims have the same name:
        self.assertRaises(AttributeError,DimArray,np.random.rand(10,5),
                          dims=[Dim(range(10),name='dim1',unit='Hz'),
                                Dim(range(5),name='dim1',unit='sec')])
        self.assertRaises(AttributeError,DimArray,np.random.rand(10,3,5),
                          dims=[Dim(range(10),name='dim1',unit='Hz'),
                                Dim(range(3),name='dim2',unit='Hz'),
                                Dim(range(5),name='dim1',unit='sec')])
        self.assertRaises(AttributeError,DimArray,np.random.rand(10,3,5),
                          dims=[Dim(range(10),name='dim1',unit='Hz'),
                                Dim(range(3),name='dim1',unit='Hz'),
                                Dim(range(5),name='dim1',unit='sec')])

        # this is a proper initialization:
        dat = DimArray(np.random.rand(5,10),
                       dims=[Dim(range(5),name='freqs',unit='Hz'),
                             Dim(range(10),name='time',unit='sec')])
        # ensure dim_names attribute is set properly:
        self.assertEquals(dat.dim_names,['freqs','time'])
        # ensure proper shape
        self.assertEquals(dat.shape,(5,10))
        # ensure dims have proper lengths:
        self.assertEquals(len(dat.dims[0]),5)
        self.assertEquals(len(dat.dims[1]),10)
        # ensure that dims attributes are copied properly:
        self.assertEquals(dat.dims[0].unit,'Hz')
        self.assertEquals(dat.dims[1].unit,'sec')
        # check that dims values are preserved:
        self.assertEquals(dat.dims[0][-1],4)
        self.assertEquals(dat.dims[1][-1],9)
        
        dat = DimArray(np.random.rand(2,4,5),
                       dims=[Dim(range(2),name='dim1',unit='Hz'),
                             Dim(range(4),name='dim2',bla='bla'),
                             Dim(range(5),name='dim3',attr1='attr1',attr2='attr2')])
        # ensure dim_names attribute is set properly:
        self.assertEquals(dat.dim_names,['dim1','dim2','dim3'])
        # ensure proper shape
        self.assertEquals(dat.shape,(2,4,5))
        # ensure dims have proper lengths:
        self.assertEquals(len(dat.dims[0]),2)
        self.assertEquals(len(dat.dims[1]),4)
        self.assertEquals(len(dat.dims[2]),5)
        # ensure that dims attributes are copied properly:
        self.assertEquals(dat.dims[0].unit,'Hz')
        self.assertEquals(dat.dims[1].bla,'bla')
        self.assertEquals(dat.dims[2].attr1,'attr1')
        self.assertEquals(dat.dims[2].attr2,'attr2')
        # check that dims values are preserved:
        self.assertEquals(dat.dims[0][-1],1)
        self.assertEquals(dat.dims[1][-1],3)
        self.assertEquals(dat.dims[2][-1],4)

    def test_getitem(self):
        dat_array = np.random.rand(2,4,5)
        dat = DimArray(dat_array,
                       dims=[Dim(range(2),name='dim1',unit='Hz'),
                             Dim(range(4),name='dim2',bla='bla'),
                             Dim(range(5),name='dim3',attr1='attr1',attr2='attr2')])

        # check that the correct elements are returned:
        self.assertEquals(dat[0,0,0],dat_array[0,0,0])
        self.assertEquals(dat[0,1,2],dat_array[0,1,2])
        self.assertEquals(dat[1,0,3],dat_array[1,0,3])
        
        # check that the returned DimArray and its dims have proper shapes:
        self.assertEquals(dat[0].shape,dat_array[0].shape)
        self.assertEquals(len(dat[0].dims[0]),dat_array[0].shape[0])
        self.assertEquals(len(dat[0].dims[1]),dat_array[0].shape[1])
        self.assertEquals(dat[0].dim_names,['dim2','dim3'])
        
        self.assertEquals(dat[1].shape,dat_array[1].shape)
        self.assertEquals(len(dat[1].dims[0]),dat_array[1].shape[0])
        self.assertEquals(len(dat[1].dims[1]),dat_array[1].shape[1])
        self.assertEquals(dat[1].dim_names,['dim2','dim3'])

        self.assertEquals(dat[0,0].shape,dat_array[0,0].shape)
        self.assertEquals(len(dat[0,0].dims[0]),dat_array[0,0].shape[0])
        self.assertEquals(dat[0,0].dim_names,['dim3'])

        self.assertEquals(dat[:,:,0].shape,dat_array[:,:,0].shape)
        self.assertEquals(len(dat[:,:,0].dims[0]),dat_array[:,:,0].shape[0])
        self.assertEquals(len(dat[:,:,0].dims[1]),dat_array[:,:,0].shape[1])
        self.assertEquals(dat[:,:,0].dim_names,['dim1','dim2'])

        self.assertEquals(dat[0:1,2,0:3].shape,dat_array[0:1,2,0:3].shape)
        self.assertEquals(len(dat[0:1,2,0:3].dims[0]),dat_array[0:1,2,0:3].shape[0])
        self.assertEquals(len(dat[0:1,2,0:3].dims[1]),dat_array[0:1,2,0:3].shape[1])
        self.assertEquals(dat[0:1,2,0:3].dim_names,['dim1','dim3'])

        # when the name of a Dim instance is given, that dim should be returned:
        self.assertTrue(isinstance(dat['dim1'],Dim))
        self.assertTrue(isinstance(dat['dim2'],Dim))
        self.assertTrue(isinstance(dat['dim3'],Dim))

        self.assertEquals(dat['dim1'].name,'dim1')
        self.assertEquals(dat['dim1'].unit,'Hz')
        self.assertEquals(dat['dim1'][-1],1)
        self.assertEquals(len(dat['dim1']),2)
        self.assertEquals(dat['dim2'].name,'dim2')
        self.assertEquals(dat['dim2'].bla,'bla')
        self.assertEquals(dat['dim2'][-1],3)
        self.assertEquals(len(dat['dim2']),4)
        self.assertEquals(dat['dim3'].name,'dim3')
        self.assertEquals(dat['dim3'].attr1,'attr1')
        self.assertEquals(dat['dim3'].attr2,'attr2')
        self.assertEquals(dat['dim3'][-1],4)
        self.assertEquals(len(dat['dim3']),5)

        # when another string is given, it should be evaluated:
        self.assertEquals(dat['dim1==0'].shape,(1,4,5))
        self.assertEquals(len(dat['dim1==0'].dims[0]),1)
        self.assertEquals(len(dat['dim1==0'].dims[1]),4)
        self.assertEquals(len(dat['dim1==0'].dims[2]),5)
        self.assertEquals(dat['dim1==0'].dim_names,['dim1','dim2','dim3'])

        self.assertEquals(dat['dim2==1'].shape,(2,1,5))
        self.assertEquals(len(dat['dim2==1'].dims[0]),2)
        self.assertEquals(len(dat['dim2==1'].dims[1]),1)
        self.assertEquals(len(dat['dim2==1'].dims[2]),5)
        self.assertEquals(dat['dim2==1'].dim_names,['dim1','dim2','dim3'])

        self.assertEquals(dat['dim2<2'].shape,(2,2,5))
        self.assertEquals(len(dat['dim2<2'].dims[0]),2)
        self.assertEquals(len(dat['dim2<2'].dims[1]),2)
        self.assertEquals(len(dat['dim2<2'].dims[2]),5)
        self.assertEquals(dat['dim2<2'].dim_names,['dim1','dim2','dim3'])
        
        self.assertEquals(dat['dim3!=2'].shape,(2,4,4))
        self.assertEquals(len(dat['dim3!=2'].dims[0]),2)
        self.assertEquals(len(dat['dim3!=2'].dims[1]),4)
        self.assertEquals(len(dat['dim3!=2'].dims[2]),4)
        self.assertEquals(dat['dim3!=2'].dim_names,['dim1','dim2','dim3'])

        # check that the right values are returned:
        self.assertEquals(dat['dim3!=2'][0,0,0],dat_array[0,0,0])
        self.assertEquals(dat['dim3!=2'][1,2,1],dat_array[1,2,1])
        self.assertEquals(dat['dim3!=2'][1,2,3],dat_array[1,2,4])

        # check indexing with a tuple of arrays and with 1-level dimensions:
        dim1=Dim(['dim'],'dim1')
        dim2=Dim([1,2],'dim2')
        dim3=Dim([3,4,5],'dim3')
        dat=DimArray([[[6,7,8],[9,10,11]]],[dim1,dim2,dim3])
        self.assertEquals(dat[np.ix_([0],[0,1],[0,1])].shape,(1,2,2))

    def test_select(self):
        # check indexing with a tuple of arrays and with 1-level dimensions:
        dim1=Dim(['dim'],'dim1')
        dim2=Dim([1,2],'dim2')
        dim3=Dim([3,4,5],'dim3')
        dat=DimArray([[[6,7,8],[9,10,11]]],[dim1,dim2,dim3])
        self.assertEquals(dat.select(dim2=dat['dim2']>1,dim3=dat['dim3']>3).shape,(1,1,2))

    def test_find(self):
        # check indexing with a tuple of arrays and with 1-level dimensions:
        dim1=Dim(['dim'],'dim1')
        dim2=Dim([1,2],'dim2')
        dim3=Dim([3,4,5],'dim3')
        dat=DimArray([[[6,7,8],[9,10,11]]],[dim1,dim2,dim3])
        indx = dat.find(dim2=dat['dim2']>1,dim3=dat['dim3']>3)
        assert_array_equal(dat.select(dim2=dat['dim2']>1,dim3=dat['dim3']>3),
                            dat[indx])

    def test_get_axis(self):
        dat = DimArray(np.random.rand(5,10,3),
                       dims=[Dim(range(5),name='one'),
                             Dim(range(10),name='two'),
                             Dim(range(3),name='three')],test='tst')
        self.assertEqual(dat.get_axis(0),0)
        self.assertEqual(dat.get_axis(1),1)
        self.assertEqual(dat.get_axis(2),2)
        self.assertEqual(dat.get_axis('one'),0)
        self.assertEqual(dat.get_axis('two'),1)
        self.assertEqual(dat.get_axis('three'),2)

    def test_reshape(self):
        # make ndarray an Dimaray with identical data
        arr = np.random.rand(5,12,3,1)
        dat = DimArray(arr,dims=[Dim(range(5),name='one'),
                                 Dim(range(12),name='two'),
                                 Dim(range(3),name='three'),
                                 Dim(range(1),name='four')],test='tst')
        newshapes = [(5,2,2,3,3),(2,3,5,3,2),(15,12),(6,2,15,1,1,1,1,1,1,1),
                     180,(1,1,1,180,1,1,1)]
        for newshape in newshapes:
            assert_array_equal(arr.reshape(newshape),dat.reshape(newshape))
            assert_array_equal(np.reshape(arr,newshape),np.reshape(dat,newshape))
        
    def test_resize(self):
        # make ndarray an Dimaray with identical data
        arr = np.random.rand(5,12,3,1)
        dat = DimArray(arr,dims=[Dim(range(5),name='one'),
                                 Dim(range(12),name='two'),
                                 Dim(range(3),name='three'),
                                 Dim(range(1),name='four')],test='tst')
        newshapes = [(5,2,2,3,3),(2,3,5,3,2),(15,12),(6,2,15,1,1,1,1,1,1,1),
                     180,(1,1,1,180,1,1,1)]
        for newshape in newshapes:
            assert_array_equal(arr.resize(newshape),dat.resize(newshape))
            assert_array_equal(np.resize(arr,newshape),np.resize(dat,newshape))
               
    def test_funcs(self):
        """Test the numpy functions"""
        # make ndarray an Dimaray with identical data
        arr = np.random.rand(5,10,3,1)
        dat = DimArray(arr,dims=[Dim(range(5),name='one'),
                                 Dim(range(10),name='two'),
                                 Dim(range(3),name='three'),
                                 Dim(range(1),name='four')],test='tst')

        # these are functions that take an axis argument:
        funcs = [np.mean,np.all,np.any,np.argmax,np.argmin,np.argsort,
                 np.cumprod,np.cumsum,np.max,np.mean,np.min,np.prod,
                 np.ptp,np.std,np.sum,np.var]
        
        # The axes for the ndarray:
        axes_arr = [None,0,1,2,3,0,1,2,3]
        # The axes for the DimArray (we want to test indexing them by
        # number and name):
        axes_dat = [None,0,1,2,3,'one','two','three','four']

        # loop through the functions and axes:
        for func in funcs:
            for a in range(len(axes_arr)):
                # apply the function to the ndarray and the DimArray
                arr_func = func(arr,axis=axes_arr[a])
                dat_func = func(dat,axis=axes_dat[a])
                # make sure they are the same:
                assert_array_equal(arr_func,dat_func)
                if not(axes_dat[a] is None):
                    # ensure we still have a DimArray
                    self.assertTrue(isinstance(dat_func,DimArray))
                    # ensure that the attributes are preserved
                    self.assertEquals(dat_func.test,'tst')
                    
        # same tests as above but this time calling the DimArray
        # methods directly (this test is necessary because it is in
        # principle possoble for the numpy function to work and the
        # DimArray method not to work (or vice versa):
        for a in range(len(axes_arr)):
            assert_array_equal(arr.all(axes_arr[a]),dat.all(axes_dat[a]))
            assert_array_equal(arr.any(axes_arr[a]),dat.any(axes_dat[a]))
            assert_array_equal(arr.argmax(axes_arr[a]),dat.argmax(axes_dat[a]))
            assert_array_equal(arr.argmin(axes_arr[a]),dat.argmin(axes_dat[a]))
            assert_array_equal(arr.argsort(axes_arr[a]),dat.argsort(axes_dat[a]))
            assert_array_equal(arr.cumprod(axes_arr[a]),dat.cumprod(axes_dat[a]))
            assert_array_equal(arr.cumsum(axes_arr[a]),dat.cumsum(axes_dat[a]))
            assert_array_equal(arr.max(axes_arr[a]),dat.max(axes_dat[a]))
            assert_array_equal(arr.mean(axes_arr[a]),dat.mean(axes_dat[a]))
            assert_array_equal(arr.min(axes_arr[a]),dat.min(axes_dat[a]))
            assert_array_equal(arr.prod(axes_arr[a]),dat.prod(axes_dat[a]))
            assert_array_equal(arr.ptp(axes_arr[a]),dat.ptp(axes_dat[a]))
            assert_array_equal(arr.std(axes_arr[a]),dat.std(axes_dat[a]))
            assert_array_equal(arr.sum(axes_arr[a]),dat.sum(axes_dat[a]))
            assert_array_equal(arr.var(axes_arr[a]),dat.var(axes_dat[a]))
            if not(axes_dat[a] is None):
                self.assertTrue(isinstance(dat.all(axes_arr[a]),DimArray))
                self.assertTrue(isinstance(dat.any(axes_arr[a]),DimArray))
                self.assertTrue(isinstance(dat.argmax(axes_arr[a]),DimArray))
                self.assertTrue(isinstance(dat.argmin(axes_arr[a]),DimArray))
                self.assertTrue(isinstance(dat.argsort(axes_arr[a]),DimArray))
                self.assertTrue(isinstance(dat.cumprod(axes_arr[a]),DimArray))
                self.assertTrue(isinstance(dat.cumsum(axes_arr[a]),DimArray))
                self.assertTrue(isinstance(dat.max(axes_arr[a]),DimArray))
                self.assertTrue(isinstance(dat.mean(axes_arr[a]),DimArray))
                self.assertTrue(isinstance(dat.min(axes_arr[a]),DimArray))
                self.assertTrue(isinstance(dat.prod(axes_arr[a]),DimArray))
                self.assertTrue(isinstance(dat.ptp(axes_arr[a]),DimArray))
                self.assertTrue(isinstance(dat.std(axes_arr[a]),DimArray))
                self.assertTrue(isinstance(dat.sum(axes_arr[a]),DimArray))
                self.assertTrue(isinstance(dat.var(axes_arr[a]),DimArray))                
                
                self.assertEquals(dat.all(axes_arr[a]).test,'tst')
                self.assertEquals(dat.any(axes_arr[a]).test,'tst')
                self.assertEquals(dat.argmax(axes_arr[a]).test,'tst')
                self.assertEquals(dat.argmin(axes_arr[a]).test,'tst')
                self.assertEquals(dat.argsort(axes_arr[a]).test,'tst')
                self.assertEquals(dat.cumprod(axes_arr[a]).test,'tst')
                self.assertEquals(dat.cumsum(axes_arr[a]).test,'tst')
                self.assertEquals(dat.max(axes_arr[a]).test,'tst')
                self.assertEquals(dat.mean(axes_arr[a]).test,'tst')
                self.assertEquals(dat.min(axes_arr[a]).test,'tst')
                self.assertEquals(dat.prod(axes_arr[a]).test,'tst')
                self.assertEquals(dat.ptp(axes_arr[a]).test,'tst')
                self.assertEquals(dat.std(axes_arr[a]).test,'tst')
                self.assertEquals(dat.sum(axes_arr[a]).test,'tst')
                self.assertEquals(dat.var(axes_arr[a]).test,'tst')                

        # test functions that require function specific input:
        for a in range(len(axes_arr)):
            if axes_arr[a] is None:
                length = len(arr)
            else:
                length = np.shape(arr)[axes_arr[a]]
            cond = np.random.random(length)>0.5
            # calling the compress method directly:
            arr_func = arr.compress(cond,axis=axes_arr[a])
            dat_func = dat.compress(cond,axis=axes_dat[a])
            assert_array_equal(arr_func,dat_func)
            if axes_dat[a] is not None:
                self.assertTrue(isinstance(dat_func,DimArray))
                self.assertEquals(dat_func.test,'tst')
            # calling the numpy compress function:
            arr_func = np.compress(cond,arr,axis=axes_arr[a])
            dat_func = np.compress(cond,dat,axis=axes_dat[a])
            assert_array_equal(arr_func,dat_func)
            if axes_dat[a] is not None:
                self.assertTrue(isinstance(dat_func,DimArray))
                self.assertEquals(dat_func.test,'tst')            

            # the below tests should not run with axis==None:
            if axes_arr[a] is None:
                continue
            
            reps = np.random.random_integers(low=1, high=10, size=length)
            # calling the repeat method directly:
            arr_func = arr.repeat(reps,axis=axes_arr[a])
            dat_func = dat.repeat(reps,axis=axes_dat[a])
            assert_array_equal(arr_func,dat_func)
            if axes_dat[a] is not None:
                self.assertTrue(isinstance(dat_func,DimArray))
                self.assertEquals(dat_func.test,'tst')
            # calling the numpy repeat function:
            arr_func = np.repeat(arr,reps,axis=axes_arr[a])
            dat_func = np.repeat(dat,reps,axis=axes_dat[a])
            assert_array_equal(arr_func,dat_func)
            if axes_dat[a] is not None:
                self.assertTrue(isinstance(dat_func,DimArray))
                self.assertEquals(dat_func.test,'tst')

            # skippint the last dimension for this test for
            # convenience (the last dimension only has 1 level):
            if a >= 3:
                continue
            # the highest valid index is 2 because the lowest number
            # of levels of all dimensions (except for the last one
            # which we skipped) is 3:
            indcs = np.random.random_integers(low=0, high=2, size=len(arr.shape))
            # calling the take method directly (squeeze, to get rid of
            # the last dimension):
            arr_func = arr.squeeze().take(indcs,axis=axes_arr[a])
            dat_func = dat.squeeze().take(indcs,axis=axes_dat[a])
            assert_array_equal(arr_func,dat_func)
            if axes_dat[a] is not None:
                self.assertTrue(isinstance(dat_func,DimArray))
                self.assertEquals(dat_func.test,'tst')
            # calling the numpy take function directly (squeeze, to get rid of
            # the last dimension):
            arr_func = np.take(arr.squeeze(),indcs,axis=axes_arr[a])
            dat_func = np.take(dat.squeeze(),indcs,axis=axes_dat[a])
            assert_array_equal(arr_func,dat_func)
            if axes_dat[a] is not None:
                self.assertTrue(isinstance(dat_func,DimArray))
                self.assertEquals(dat_func.test,'tst')

        # This should work with numpy 1.2 (possibly 1.1.1) but doesn't with 1.1.0
        #arr_func = arr.clip(0.4,0.6)
        #dat_func = dat.clip(0.4,0.6)
        #assert_array_equal(arr_func,dat_func)
        #self.assertTrue(isinstance(dat_func,DimArray))
        #self.assertEquals(dat_func.test,'tst')
        #arr_func = np.clip(arr,0.4,0.6)
        #dat_func = np.clip(dat,0.4,0.6)
        #assert_array_equal(arr_func,dat_func)
        #self.assertTrue(isinstance(dat_func,DimArray))
        #self.assertEquals(dat_func.test,'tst')

        # other functions that don't necessarily take return a
        # DimArray:
        funcs = [np.diagonal,np.nonzero,np.ravel,np.squeeze,np.sort,np.trace,np.transpose]
        for func in funcs:
            arr_func = func(arr)
            dat_func = func(dat)
            assert_array_equal(arr_func,dat_func)

        # same tests as above, but calling the methods directly:
        assert_array_equal(arr.diagonal(),dat.diagonal())
        assert_array_equal(arr.nonzero(),dat.nonzero())
        assert_array_equal(arr.ravel(),dat.ravel())
        assert_array_equal(arr.squeeze(),dat.squeeze())
        assert_array_equal(arr.sort(),dat.sort())
        assert_array_equal(arr.trace(),dat.trace())
        assert_array_equal(arr.transpose(),dat.transpose())
        # there is no numpy.flatten() function, so we only call the
        # method directly:
        assert_array_equal(arr.flatten(),dat.flatten())
        
        assert_array_equal(arr.swapaxes(0,1),dat.swapaxes(0,1))
        self.assertTrue(isinstance(dat.swapaxes(0,1),DimArray))
        self.assertEquals(dat.swapaxes(0,1).test,'tst')
        assert_array_equal(arr.swapaxes(0,1),dat.swapaxes('one','two'))
        self.assertTrue(isinstance(dat.swapaxes('one','two'),DimArray))
        self.assertEquals(dat.swapaxes('one','two').test,'tst')
        assert_array_equal(arr.swapaxes(1,3),dat.swapaxes(1,3))
        self.assertTrue(isinstance(dat.swapaxes(1,3),DimArray))
        self.assertEquals(dat.swapaxes(1,3).test,'tst')
        assert_array_equal(arr.swapaxes(1,3),dat.swapaxes('two','four'))
        self.assertTrue(isinstance(dat.swapaxes('two','four'),DimArray))
        self.assertEquals(dat.swapaxes('two','four').test,'tst')

        

        
