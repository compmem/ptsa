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
from edf import read_samples, read_samplerate
# global imports
import numpy as np

class EdfWrapper(BaseWrapper):
    """
    Interface to data stored in a EDF/BDF file.
    """
    def __init__(self, filename):
        """Initialize the interface to the data.  You must specify the
        data and the samplerate."""
        # set up the basic params of the data
        self.filename = filename

    def get_samplerate(self, channel):
        # Same samplerate for all channels:
        return read_samplerate(self.filename, channel)

    def _load_data(self,channel,event_offsets,dur_samp,offset_samp):
        """        
        """
        # allocate for data
	eventdata = np.empty((len(event_offsets),dur_samp),
                             dtype=np.float64)*np.nan

	# loop over events
        # PBS: eventually move this to the cython file
	for e,evOffset in enumerate(event_offsets):
            # set the range
            ssamp = offset_samp+evOffset

            # read the data
            dat = read_samples(self.filename,
                               channel,
                               ssamp, dur_samp)
            
            # check the ranges
            if len(dat) < dur_samp:
                raise IOError('Event with offset '+str(evOffset)+
                              ' is outside the bounds of the data.')
            eventdata[e,:] = dat


        return eventdata
    


