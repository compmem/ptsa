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
import h5py

class HDF5Wrapper(BaseWrapper):
    """
    Interface to data stored in an HDF5 file.
    """
    def __init__(self, filepath, dataset_name='data'):
        """
        Initialize the interface to the data.
        """
        # set up the basic params of the data
        self.filepath = filepath
        self.dataset_name = dataset_name

    def get_samplerate(self, channel):
        # Same samplerate for all channels:
        # get the samplerate property of the dataset
        f = h5py.File(self.filepath,'r')
        data = f[self.dataset_name]
        samplerate = data.attrs['samplerate']
        f.close()
        return samplerate

    def _load_data(self,channel,event_offsets,dur_samp,offset_samp):
        """        
        """
        # connect to the file and get the dataset
        f = h5py.File(self.filepath,'r')
        data = f[self.dataset_name]
        
        # allocate for data
	eventdata = np.empty((len(event_offsets),dur_samp),
                             dtype=data.dtype)*np.nan

	# loop over events
	for e,evOffset in enumerate(event_offsets):
            # set the range
            ssamp = offset_samp+evOffset
            esamp = ssamp + dur_samp
            
            # check the ranges
            if ssamp < 0 or esamp > data.shape[1]:
                raise IOError('Event with offset '+str(evOffset)+
                              ' is outside the bounds of the data.')
            eventdata[e,:] = data[channel,ssamp:esamp]

        # close the file
        f.close()
        
        return eventdata
    
    def _load_all_data(self,channel):
        """
        """
        # connect to the file and get the dataset
        f = h5py.File(self.filepath,'r')
        data = f[self.dataset_name]

        # select the data to return
        toreturn = data[channel,:]

        # close the file
        f.close()
        
        return toreturn

