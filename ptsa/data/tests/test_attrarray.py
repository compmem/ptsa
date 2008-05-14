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

from ptsa.data import AttrArray

class test_AttrArray(NumpyTestCase):
    def setUp(self):
        pass

    def test_new(self):
        # test new instantiation (attributes are defined by kwargs)
        
        # instatiation with a numpy ndarray:
        shape = (10,)
        arr = np.random.random_sample(shape)
        dat_array = AttrArray(arr,name='randvals')
        self.assertTrue(dat_array.name == 'randvals')
        self.assertEquals(shape,dat_array.shape)
        self.assertTrue((dat_array==arr).all())
        # another instatioation with an ndarray, but this time with dtype set:
        shape = (1,2,3,4)
        arr = np.random.random_sample(shape)
        dat_array = AttrArray(arr,name='randvals',dtype=np.float32)
        self.assertTrue(dat_array.name == 'randvals')
        self.assertEquals(shape,dat_array.shape)
        # XXX not sure why only "almost" equal:
        assert_array_almost_equal(dat_array,arr)
        self.assertTrue(dat_array.dtype==np.float32)
        

        # another ndarray, with copy = True vs. copy = False
        shape = (10,9,8,7,6,1,8,8)
        arr = np.random.random_sample(shape)
        dat_array = AttrArray(arr,name='randvals', test1=33,
                              test2='test', copy = True)
        self.assertTrue(dat_array.name == 'randvals')
        self.assertTrue(dat_array.test1 == 33)
        self.assertTrue(dat_array.test2 == 'test')
        self.assertEquals(shape,dat_array.shape)
        assert_array_equal(dat_array,arr)
        dat_array[0] += 5
        # XXX not sure why only "almost" equal:
        assert_array_almost_equal((dat_array[0]-5), arr[0])
        dat_array = AttrArray(arr,name='randvals', test1=33,
                              test2='test', copy = False)
        self.assertTrue(dat_array.name == 'randvals')
        self.assertTrue(dat_array.test1 == 33)
        self.assertTrue(dat_array.test2 == 'test')
        self.assertEquals(shape,dat_array.shape)
        self.assertTrue((dat_array==arr).all())
        dat_array[0] += 5
        assert_array_equal(dat_array[0],arr[0])
        
        # instantiation with a list:
        lst = range(10)
        dat_list = AttrArray(lst,name='range')
        self.assertTrue(dat_list.name == 'range')
        self.assertTrue((lst==dat_list).all())
        lst = [['a','b','c']]
        dat_list = AttrArray(lst,name='list')
        self.assertTrue(dat_list.name == 'list')
        self.assertTrue((lst==dat_list).all())
        lst = [[1,2,3],[4.5,6,7]]
        dat_list = AttrArray(lst,name='list')
        self.assertTrue(dat_list.name == 'list')
        self.assertTrue((lst==dat_list).all())

        # instantiation with a AttrArray:
        dat_attrarray = AttrArray(dat_array,name='attrarray')
        self.assertTrue(dat_attrarray.name == 'attrarray')
        dat_attrarray = AttrArray(dat_list,newname='attrarray',test=44)
        self.assertTrue(dat_attrarray.newname == 'attrarray')
        self.assertTrue(dat_attrarray.test == 44)        

    def test_setattr(self):
        dat = AttrArray(np.random.rand(10),name='randvals')
        # add a custom attribute:
        dat.custom = 'attribute'
        self.assertEquals(dat.custom,'attribute')

    def test_getattr(self):
        dat = AttrArray(np.random.rand(10),name='randvals')
        self.assertEquals(dat.name,'randvals')

    def test_method(self):
        # make sure ndarray methods work and return a new AttrArray
        # instance with the attributes intact
        dat = AttrArray(np.random.rand(10),name='randvals')
        sdat = np.sqrt(dat)
        self.assertEquals(sdat.name,'randvals')
