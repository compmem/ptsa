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
from edf import read_samples, read_number_of_samples
from edf import read_samplerate, read_annotations
from edf import read_number_of_signals
# global imports
import numpy as np
import os.path

class EdfWrapper(BaseWrapper):
    """
    Interface to data stored in a EDF file and related formats (such
    as BDF).
    """
    def __init__(self, filepath):
        """
        Initialize the interface to the data.

        Parameters
        ----------
        filepath : string
            String specifiying the filename (with full path if
            applicable).
        """
        # set up the basic params of the data
        if os.path.exists(filepath):
            self.filepath = filepath
        else:
            raise IOError(str(filepath)+'\n does not exist!'+
                          'Valid path to data file is needed!')

    # def _get_nchannels(self):
    #     return read_number_of_signals(self.filepath)

    def _get_channels(self):
        # this needs to be replaced by a list of the propper channel labels!
        # return [str(channel)
        #         for channel in range(read_number_of_signals(self.filepath))]
        return [channel
                for channel in range(read_number_of_signals(self.filepath))]

    def _get_nsamples(self, channel=None):
        if channel is None:
            # pick first channel
            channel = 0
        return read_number_of_samples(self.filepath, channel)

    def _get_samplerate(self, channel=None):
        if channel is None:
            # pick first channel
            channel = 0
        return read_samplerate(self.filepath, channel)

    def _get_annotations(self):
        return read_annotations(self.filepath)

    def _load_data(self,channels,event_offsets,dur_samp,offset_samp):
        """        
        """
        # allocate for data
	eventdata = np.empty((len(channels),len(event_offsets),dur_samp),
                             dtype=np.float64)*np.nan

	# loop over events
        # PBS: eventually move this to the cython file
        for c,channel in enumerate(channels):
            for e,ev_offset in enumerate(event_offsets):
                # set the range
                ssamp = offset_samp+ev_offset

                # read the data
                dat = read_samples(self.filepath,
                                   channel,
                                   ssamp, dur_samp)

                # check the ranges
                if len(dat) < dur_samp:
                    raise IOError('Event with offset '+str(ev_offset)+
                                  ' is outside the bounds of the data.')
                eventdata[c,e,:] = dat

        return eventdata    


