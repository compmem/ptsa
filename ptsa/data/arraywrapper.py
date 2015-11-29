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
    def __init__(self,data,samplerate,annotations=None):
        """Initialize the interface to the data.  You must specify the
        data and the samplerate."""
        # set up the basic params of the data
        self._data = data
        self._samplerate = samplerate
        self._annotations = annotations

    def _get_nchannels(self):
        return self._data.shape[0]

    def _get_nsamples(self, channel=None):
        return self._data.shape[1]

    def _get_samplerate(self, channel=None):
        # Same samplerate for all channels:
        return self._samplerate

    def _get_annotations(self):
        return self._annotations

    def _load_data(self,channels,event_offsets,dur_samp,offset_samp):
        """        
        """
        # allocate for data
	eventdata = np.empty((len(channels),len(event_offsets),dur_samp),
                             dtype=self._data.dtype)*np.nan

	# loop over events
	for e,evOffset in enumerate(event_offsets):
            # set the range
            ssamp = offset_samp+evOffset
            esamp = ssamp + dur_samp
            
            # check the ranges
            if ssamp < 0 or esamp > self._data.shape[1]:
                raise IOError('Event with offset '+str(evOffset)+
                              ' is outside the bounds of the data.')
            eventdata[:,e,:] = self._data[channels,ssamp:esamp]

        return eventdata
