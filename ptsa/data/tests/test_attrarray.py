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

from ptsa.data import AttrArray

class test_AttrArray(NumpyTestCase):
    def setUp(self):
        pass

    def test_new(self):
        # test new instantiation
        # attributes are defined by kwargs
        dat = AttrArray(N.random.rand(10),name='randvals')

    def test_getattr(self):
        dat = AttrArray(N.random.rand(10),name='randvals')
        self.assertEquals(dat.name == 'randvals')

    def test_method(self):
        # make sure ndarray methods work and return a new AttrArray
        # instance with the attributes intact
        dat = AttrArray(N.random.rand(10),name='randvals')
        sdat = N.sqrt(dat)
        self.assertEquals(sdat.name == 'randvals')
