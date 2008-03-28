#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

from ptsa.data import DimArray


# DimArray class
class test_DimArray(NumpyTestCase):
    def setUp(self):
        pass
#         td = TestData()
#         self.dat200 = td.dat200
#         self.dims200 = td.dims200
#         self.dat50 = td.dat50
#         self.dims50 = td.dims50

    def test_init(self):
        pass

    def test_func(self):
        dat = DimArray(N.random.rand(),
                       dims=())

        # what we want

        # get the ind of a dim
        t_ind = dat.dim('time')
        f_ind = dat.dim('freqs')

        # get a dim's data
        times = dat.time
        times = dat['time']
        times = dat.dims['time']
        freqs = dat['freqs']
        
        # get data, itself
        dat['(time>=-200) & (time<=1000)','freqs==4']
        dat[:,2:8]
        
        # call methods of class
        dat.mean(axis='time')
        dat.mean(axis=dat.dim('time'))
        
