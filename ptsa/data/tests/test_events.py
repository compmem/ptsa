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

from ptsa import Events,EegEvents

class test_Events(NumpyTestCase):
    def setUp(self):
        # set up a fake dataset
        self.events = Events(N.rec.fromarrays([N.random.rand(3),
                                               ['jubba','wubba','lubba']],
                                              names='val,type'))

    def test_init(self):
        pass

    def test_getitem(self):
        pass

    def test_setitem(self):
        pass

    def test_select(self):
        pass

    def test_extend(self):
        pass

    def test_remove_fields(self):
        pass

    def test_add_fields(self):
        pass

    
