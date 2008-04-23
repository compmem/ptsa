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
        tst = Dim(np.array(5),name='test')
        tst = Dim(range(2),name='test')


# DimArray class
class test_DimArray(NumpyTestCase):
    def setUp(self):
        pass
    
    def test_new(self):
        dat = DimArray(np.random.rand(5,10),
                       dims=(Dim(range(5),name='freqs',unit='Hz'),
                             Dim(range(10),name='time',unit='sec')))

    def test_func(self):
        dat = DimArray(np.random.rand(5,10),
                       dims=(Dim(range(5),name='freqs',unit='Hz'),
                             Dim(range(10),name='time',unit='sec')))

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
        
