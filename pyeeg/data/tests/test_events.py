import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase


class test_Events(NumpyTestCase):
    def setUp(self):
        # set up a fake dataset
        self.events = N.rec.fromarrays([N.random.rand(3),
                                 ['jubba','wubba','lubba']],
                                names='val,type')
    def test_extend(self):
