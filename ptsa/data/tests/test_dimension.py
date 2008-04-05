#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as N
from numpy.testing import *

from ptsa.data.dimension import *

class test_NamedArray(NumpyTestCase):
    def test_init(self):
        x = NamedArray(N.arange(5))
        self.assertTrue(x.name is None)
        x = NamedArray(N.arange(5),name='test')
        self.assertEqual(x.name,'test')
        y = N.arange(10).view(NamedArray)
        self.assertTrue(y.name is None)
        y = x.view(NamedArray)
        self.assertEqual(x.name,y.name)

class test_Dimension(NumpyTestCase):
    def test_init(self):
        dat = N.arange(10)
        dat = dat.view(NamedArray)
        dat.name = 'test'
        dim = dat.view(Dimension)
        self.assertEqual(dim.name,'test')
        x = Dimension(N.arange(5),name='test')
        self.assertEqual(x.name,'test')
        y = dim.view(Dimension)
        self.assertEqual(x.name,y.name)


