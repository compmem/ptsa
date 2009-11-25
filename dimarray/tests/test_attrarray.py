#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from numpy.testing import TestCase,\
     assert_array_equal, assert_array_almost_equal

from dimarray import AttrArray

import cPickle as pickle

class test_AttrArray(TestCase):
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
        # another instatioation with an ndarray, but this time with
        # dtype set:
        shape = (1,2,3,4)
        arr = np.random.random_sample(shape)
        dat_array = AttrArray(arr,name='randvals',dtype=np.float32)
        self.assertTrue(dat_array.name == 'randvals')
        self.assertEquals(shape,dat_array.shape)
        # "almost" equal because of the casting to np.float32:
        assert_array_almost_equal(dat_array,arr)
        self.assertTrue(dat_array.dtype==np.float32)

        # # another ndarray, with copy = True vs. copy = False
        shape = (10,9,8,7,6,1,8)
        arr = np.random.random_sample(shape)
        dat_array = AttrArray(arr,name='randvals', test1=33,
                              test2='test', copy = True)
        self.assertTrue(dat_array.name == 'randvals')
        self.assertTrue(dat_array.test1 == 33)
        self.assertTrue(dat_array.test2 == 'test')
        self.assertEquals(shape,dat_array.shape)
        assert_array_equal(dat_array,arr)
        dat_array[0] += 5
        # # "almost" equal because of slight inaccuracies in the the
        # # representation of floats:
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
        # _required_attrs is read only:
        self.assertRaises(AttributeError,dat.__setattr__,'_required_attrs','test')

    def test_getattr(self):
        dat = AttrArray(np.random.rand(10),name='randvals')
        self.assertEquals(dat.name,'randvals')

    def test_method(self):
        # make sure ndarray methods work and return a new AttrArray
        # instance with the attributes intact
        dat = AttrArray(np.random.rand(10),name='randvals')
        sdat = np.sqrt(dat)
        self.assertEquals(sdat.name,'randvals')

    def test_pickle(self):
        # make sure we can pickle this thing
        dat = AttrArray(np.random.rand(10),name='randvals')
        # dump to string
        pstr = pickle.dumps(dat)

        # load to new variable
        dat2 = pickle.loads(pstr)

        # make sure data same
        assert_array_equal(dat,dat2)

        # make sure has attr and it's correct
        self.assertTrue(hasattr(dat2,'_attrs'))
        self.assertTrue(hasattr(dat2,'name'))
        self.assertEquals(dat2.name, 'randvals')

        # make sure has required attr
        self.assertTrue(hasattr(dat2,'_required_attrs'))

    def test_nanstd(self):
        arr = np.random.rand(10,10,10)
        dat = AttrArray(arr,name='randvals')
        # if there are no NaN's, std and nanstd should give equal
        # results:
        self.assertEquals(dat.std(),dat.nanstd())
        assert_array_almost_equal(dat.std(0),dat.nanstd(0))
        self.assertEquals(dat.nanstd(0).name, 'randvals')
        assert_array_almost_equal(dat.std(1),dat.nanstd(1))
        self.assertEquals(dat.nanstd(1).name, 'randvals')
        assert_array_almost_equal(dat.std(2),dat.nanstd(2))
        self.assertEquals(dat.nanstd(2).name, 'randvals')
        # test ddof:
        for d in range(3):
            self.assertEquals(dat.std(ddof=d),dat.nanstd(ddof=d))
            assert_array_almost_equal(dat.std(0,ddof=d),dat.nanstd(0,ddof=d))
            self.assertEquals(dat.nanstd(0,ddof=d).name, 'randvals')
            assert_array_almost_equal(dat.std(1,ddof=d),dat.nanstd(1,ddof=d))
            self.assertEquals(dat.nanstd(1,ddof=d).name, 'randvals')
            assert_array_almost_equal(dat.std(2,ddof=d),dat.nanstd(2,ddof=d))
            self.assertEquals(dat.nanstd(2,ddof=d).name, 'randvals')

        # Now, make sure results are as expected with NaN present:
        arr[0,0,0] = np.nan
        dat = AttrArray(arr,name='randvals')
        self.assertEquals(dat[~np.isnan(dat)].std(),dat.nanstd())
        for i in range(len(arr.shape)):
            tmp1 = dat.std(i)
            tmp1[0,0] = 0
            tmp2 = dat.nanstd(i)
            tmp2[0,0] = 0
            assert_array_almost_equal(tmp1,tmp2)
            self.assertEquals(tmp2.name, 'randvals')
        arr[3,6,2] = np.nan
        dat = AttrArray(arr,name='randvals')
        self.assertEquals(dat[~np.isnan(dat)].std(),dat.nanstd())
        tmp1 = dat.std(0)
        tmp1[0,0] = 0
        tmp1[6,2] = 0
        tmp2 = dat.nanstd(0)
        tmp2[0,0] = 0
        tmp2[6,2] = 0
        assert_array_almost_equal(tmp1,tmp2)
        self.assertEquals(tmp2.name, 'randvals')
        tmp1 = dat.std(1)
        tmp1[0,0] = 0
        tmp1[3,2] = 0
        tmp2 = dat.nanstd(1)
        tmp2[0,0] = 0
        tmp2[3,2] = 0
        assert_array_almost_equal(tmp1,tmp2)
        self.assertEquals(tmp2.name, 'randvals')
        tmp1 = dat.std(2)
        tmp1[0,0] = 0
        tmp1[3,6] = 0
        tmp2 = dat.nanstd(2)
        tmp2[0,0] = 0
        tmp2[3,6] = 0
        assert_array_almost_equal(tmp1,tmp2)
        self.assertEquals(tmp2.name, 'randvals')
        # Test ddof:
        for d in range(3):
            self.assertEquals(dat[~np.isnan(dat)].std(ddof=d),dat.nanstd(ddof=d))
            tmp1 = dat.std(0,ddof=d)
            tmp1[0,0] = 0
            tmp1[6,2] = 0
            tmp2 = dat.nanstd(0,ddof=d)
            tmp2[0,0] = 0
            tmp2[6,2] = 0
            assert_array_almost_equal(tmp1,tmp2)
            self.assertEquals(tmp2.name, 'randvals')
            tmp1 = dat.std(1,ddof=d)
            tmp1[0,0] = 0
            tmp1[3,2] = 0
            tmp2 = dat.nanstd(1,ddof=d)
            tmp2[0,0] = 0
            tmp2[3,2] = 0
            assert_array_almost_equal(tmp1,tmp2)
            self.assertEquals(tmp2.name, 'randvals')
            tmp1 = dat.std(2,ddof=d)
            tmp1[0,0] = 0
            tmp1[3,6] = 0
            tmp2 = dat.nanstd(2,ddof=d)
            tmp2[0,0] = 0
            tmp2[3,6] = 0
            assert_array_almost_equal(tmp1,tmp2)
            self.assertEquals(tmp2.name, 'randvals')

    def test_nanvar(self):
        arr = np.random.rand(10,10,10)
        dat = AttrArray(arr,name='randvals')
        # if there are no NaN's, var and nanvar should give equal
        # results:
        self.assertEquals(dat.var(),dat.nanvar())
        assert_array_almost_equal(dat.var(0),dat.nanvar(0))
        self.assertEquals(dat.nanvar(0).name, 'randvals')
        assert_array_almost_equal(dat.var(1),dat.nanvar(1))
        self.assertEquals(dat.nanvar(1).name, 'randvals')
        assert_array_almost_equal(dat.var(2),dat.nanvar(2))
        self.assertEquals(dat.nanvar(2).name, 'randvals')
        # test ddof:
        for d in range(3):
            self.assertEquals(dat.var(ddof=d),dat.nanvar(ddof=d))
            assert_array_almost_equal(dat.var(0,ddof=d),dat.nanvar(0,ddof=d))
            self.assertEquals(dat.nanvar(0,ddof=d).name, 'randvals')
            assert_array_almost_equal(dat.var(1,ddof=d),dat.nanvar(1,ddof=d))
            self.assertEquals(dat.nanvar(1,ddof=d).name, 'randvals')
            assert_array_almost_equal(dat.var(2,ddof=d),dat.nanvar(2,ddof=d))
            self.assertEquals(dat.nanvar(2,ddof=d).name, 'randvals')

        # Now, make sure results are as expected with NaN present:
        arr[0,0,0] = np.nan
        dat = AttrArray(arr,name='randvals')
        self.assertEquals(dat[~np.isnan(dat)].var(),dat.nanvar())
        for i in range(len(arr.shape)):
            tmp1 = dat.var(i)
            tmp1[0,0] = 0
            tmp2 = dat.nanvar(i)
            tmp2[0,0] = 0
            assert_array_almost_equal(tmp1,tmp2)
            self.assertEquals(tmp2.name, 'randvals')
        arr[3,6,2] = np.nan
        dat = AttrArray(arr,name='randvals')
        self.assertEquals(dat[~np.isnan(dat)].var(),dat.nanvar())
        tmp1 = dat.var(0)
        tmp1[0,0] = 0
        tmp1[6,2] = 0
        tmp2 = dat.nanvar(0)
        tmp2[0,0] = 0
        tmp2[6,2] = 0
        assert_array_almost_equal(tmp1,tmp2)
        self.assertEquals(tmp2.name, 'randvals')
        tmp1 = dat.var(1)
        tmp1[0,0] = 0
        tmp1[3,2] = 0
        tmp2 = dat.nanvar(1)
        tmp2[0,0] = 0
        tmp2[3,2] = 0
        assert_array_almost_equal(tmp1,tmp2)
        self.assertEquals(tmp2.name, 'randvals')
        tmp1 = dat.var(2)
        tmp1[0,0] = 0
        tmp1[3,6] = 0
        tmp2 = dat.nanvar(2)
        tmp2[0,0] = 0
        tmp2[3,6] = 0
        assert_array_almost_equal(tmp1,tmp2)
        self.assertEquals(tmp2.name, 'randvals')
        # Test ddof:
        for d in range(3):
            self.assertEquals(dat[~np.isnan(dat)].var(ddof=d),dat.nanvar(ddof=d))
            tmp1 = dat.var(0,ddof=d)
            tmp1[0,0] = 0
            tmp1[6,2] = 0
            tmp2 = dat.nanvar(0,ddof=d)
            tmp2[0,0] = 0
            tmp2[6,2] = 0
            assert_array_almost_equal(tmp1,tmp2)
            self.assertEquals(tmp2.name, 'randvals')
            tmp1 = dat.var(1,ddof=d)
            tmp1[0,0] = 0
            tmp1[3,2] = 0
            tmp2 = dat.nanvar(1,ddof=d)
            tmp2[0,0] = 0
            tmp2[3,2] = 0
            assert_array_almost_equal(tmp1,tmp2)
            self.assertEquals(tmp2.name, 'randvals')
            tmp1 = dat.var(2,ddof=d)
            tmp1[0,0] = 0
            tmp1[3,6] = 0
            tmp2 = dat.nanvar(2,ddof=d)
            tmp2[0,0] = 0
            tmp2[3,6] = 0
            assert_array_almost_equal(tmp1,tmp2)
            self.assertEquals(tmp2.name, 'randvals')

    def test_nanmean(self):
        arr = np.random.rand(10,10,10)
        dat = AttrArray(arr,name='randvals')
        # if there are no NaN's, mean and nanmean should give equal
        # results:
        self.assertEquals(dat.mean(),dat.nanmean())
        assert_array_almost_equal(dat.mean(0),dat.nanmean(0))
        self.assertEquals(dat.nanmean(0).name, 'randvals')
        assert_array_almost_equal(dat.mean(1),dat.nanmean(1))
        self.assertEquals(dat.nanmean(1).name, 'randvals')
        assert_array_almost_equal(dat.mean(2),dat.nanmean(2))
        self.assertEquals(dat.nanmean(2).name, 'randvals')
        # Now, make sure results are as expected with NaN present:
        arr[0,0,0] = np.nan
        dat = AttrArray(arr,name='randvals')
        self.assertEquals(dat[~np.isnan(dat)].mean(),dat.nanmean())
        for i in range(len(arr.shape)):
            tmp1 = dat.mean(i)
            tmp1[0,0] = 0
            tmp2 = dat.nanmean(i)
            tmp2[0,0] = 0
            assert_array_almost_equal(tmp1,tmp2)
            self.assertEquals(tmp2.name, 'randvals')
        arr[3,6,2] = np.nan
        dat = AttrArray(arr,name='randvals')
        self.assertEquals(dat[~np.isnan(dat)].mean(),dat.nanmean())
        tmp1 = dat.mean(0)
        tmp1[0,0] = 0
        tmp1[6,2] = 0
        tmp2 = dat.nanmean(0)
        tmp2[0,0] = 0
        tmp2[6,2] = 0
        assert_array_almost_equal(tmp1,tmp2)
        self.assertEquals(tmp2.name, 'randvals')
        tmp1 = dat.mean(1)
        tmp1[0,0] = 0
        tmp1[3,2] = 0
        tmp2 = dat.nanmean(1)
        tmp2[0,0] = 0
        tmp2[3,2] = 0
        assert_array_almost_equal(tmp1,tmp2)
        self.assertEquals(tmp2.name, 'randvals')
        tmp1 = dat.mean(2)
        tmp1[0,0] = 0
        tmp1[3,6] = 0
        tmp2 = dat.nanmean(2)
        tmp2[0,0] = 0
        tmp2[3,6] = 0
        assert_array_almost_equal(tmp1,tmp2)
        self.assertEquals(tmp2.name, 'randvals')
