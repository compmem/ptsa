#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

# local imports
from basewrapper import BaseWrapper

# global imports
import numpy as np

class ArrayWrapper(BaseWrapper):
    """
    Interface to data stored in a numpy ndarray where the first
    dimension is the channel and the second dimension is samples.
    """
    def __init__(self,data,samplerate):
        """Initialize the interface to the data.  You must specify the
        data and the samplerate."""
        # set up the basic params of the data
        self.data = data
        self.samplerate = samplerate

    def _load_data(self,channel,eventOffsets,dur_samp,offset_samp):
        """        
        """
        # allocate for data
	eventdata = np.empty((len(eventOffsets),dur_samp),
                             dtype=self.data.dtype)

	# loop over events
	for e,evOffset in enumerate(eventOffsets):
            # set the range
            ssamp = offset_samp+evOffset
            esamp = ssamp + dur_samp

            eventdata[e,:] = self.data[channel,ssamp:esamp]

        return eventdata
    


