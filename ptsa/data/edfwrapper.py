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

    def get_number_of_signals(self):
        return read_number_of_signals(self.filepath)

    def get_number_of_samples(self, channel):
        return read_number_of_samples(self.filepath, channel)

    def get_samplerate(self, channel):
        # Same samplerate for all channels:
        return read_samplerate(self.filepath, channel)

    def get_annotations(self):
        return read_annotations(self.filepath)

    def _load_data(self,channel,event_offsets,dur_samp,offset_samp):
        """        
        """
        # allocate for data
	eventdata = np.empty((len(event_offsets),dur_samp),
                             dtype=np.float64)*np.nan

	# loop over events
        # PBS: eventually move this to the cython file
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
            eventdata[e,:] = dat


        return eventdata
    
    def _load_all_data(self,channel,dur_chunk=7372800):
        """
        """
        dat = None # the data array
        offset = 0 # we start from the very beginning
        end_reached = False
        while not(end_reached):
            tmp_dat = read_samples(self.filepath,channel,offset,dur_chunk)
            if(len(tmp_dat)<dur_chunk):
                end_reached = True
            else:
                offset += dur_chunk
                
            if dat is None:
                dat = tmp_dat
            else:
                dat = np.r_[dat,tmp_dat]
                
        return dat
    


