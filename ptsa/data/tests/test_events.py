#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


import numpy as np
from numpy.testing import TestCase

from ptsa.data import ArrayWrapper
from ptsa.data.events import Events

class test_Events(TestCase):
    def setUp(self):
        self.dat = np.random.rand(10,1000)
        self.aw = ArrayWrapper(self.dat,200)
        self.eoffsets = [80,140,270]
    def test_get_data(self):
        # get data directly from the wrapper
        ed = self.aw.get_event_data(3,self.eoffsets,.5,-.1,.25)

        # get data from a tsevents
        # make some events
        events = np.rec.fromarrays(([self.aw]*len(self.eoffsets),self.eoffsets),
                                   names='esrc,eoffset').view(Events)
        ed2 = events.get_data(3,.5,-.1,.25)

        np.testing.assert_array_almost_equal(ed[:],ed2[:],decimal=6)
