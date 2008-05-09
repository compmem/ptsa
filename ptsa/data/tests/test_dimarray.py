#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from numpy.testing import NumpyTest, NumpyTestCase
from numpy.random import random_sample as rnd

from ptsa.data.dimarray import DimArray, Dim
from ptsa.data.attrarray import AttrArray



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
        # ensure names attribute is set properly:
        self.assertEquals(dat.names,['freqs','time'])
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
        # ensure names attribute is set properly:
        self.assertEquals(dat.names,['dim1','dim2','dim3'])
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
        self.assertEquals(dat[0].names,['dim2','dim3'])
        
        self.assertEquals(dat[1].shape,dat_array[1].shape)
        self.assertEquals(len(dat[1].dims[0]),dat_array[1].shape[0])
        self.assertEquals(len(dat[1].dims[1]),dat_array[1].shape[1])
        self.assertEquals(dat[1].names,['dim2','dim3'])

        self.assertEquals(dat[0,0].shape,dat_array[0,0].shape)
        self.assertEquals(len(dat[0,0].dims[0]),dat_array[0,0].shape[0])
        self.assertEquals(dat[0,0].names,['dim3'])

        self.assertEquals(dat[:,:,0].shape,dat_array[:,:,0].shape)
        self.assertEquals(len(dat[:,:,0].dims[0]),dat_array[:,:,0].shape[0])
        self.assertEquals(len(dat[:,:,0].dims[1]),dat_array[:,:,0].shape[1])
        self.assertEquals(dat[:,:,0].names,['dim1','dim2'])

        self.assertEquals(dat[0:1,2,0:3].shape,dat_array[0:1,2,0:3].shape)
        self.assertEquals(len(dat[0:1,2,0:3].dims[0]),dat_array[0:1,2,0:3].shape[0])
        self.assertEquals(len(dat[0:1,2,0:3].dims[1]),dat_array[0:1,2,0:3].shape[1])
        self.assertEquals(dat[0:1,2,0:3].names,['dim1','dim3'])

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
        self.assertEquals(dat['dim1==0'].names,['dim1','dim2','dim3'])

        self.assertEquals(dat['dim2==1'].shape,(2,1,5))
        self.assertEquals(len(dat['dim2==1'].dims[0]),2)
        self.assertEquals(len(dat['dim2==1'].dims[1]),1)
        self.assertEquals(len(dat['dim2==1'].dims[2]),5)
        self.assertEquals(dat['dim2==1'].names,['dim1','dim2','dim3'])

        self.assertEquals(dat['dim2<2'].shape,(2,2,5))
        self.assertEquals(len(dat['dim2<2'].dims[0]),2)
        self.assertEquals(len(dat['dim2<2'].dims[1]),2)
        self.assertEquals(len(dat['dim2<2'].dims[2]),5)
        self.assertEquals(dat['dim2<2'].names,['dim1','dim2','dim3'])
        
        self.assertEquals(dat['dim3!=2'].shape,(2,4,4))
        self.assertEquals(len(dat['dim3!=2'].dims[0]),2)
        self.assertEquals(len(dat['dim3!=2'].dims[1]),4)
        self.assertEquals(len(dat['dim3!=2'].dims[2]),4)
        self.assertEquals(dat['dim3!=2'].names,['dim1','dim2','dim3'])

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
        

    def test_func(self):
        dat = DimArray(np.random.rand(5,10),
                       dims=[Dim(range(5),name='freqs',unit='Hz'),
                             Dim(range(10),name='time',unit='sec')])

        # what we want

        # get the ind of a dim
        #t_ind = dat.dim('time')
        #f_ind = dat.dim('freqs')

        # get a dim's data
        #times = dat.time
        #times = dat['time']
        #times = dat.dims['time']
        #freqs = dat['freqs']
        
        # get data, itself
        #dat['(time>=-200) & (time<=1000)','freqs==4']
        #dat[:,2:8]
        
        # call methods of class
        #dat.mean(axis='time')
        #dat.mean(axis=dat.dim('time'))
        
