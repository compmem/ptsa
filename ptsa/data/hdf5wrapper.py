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
    def __init__(self, filepath, dataset_name='data',
                 annotations_name='annotations'):
        """
        Initialize the interface to the data.
        """
        # set up the basic params of the data
        self.filepath = filepath
        self.dataset_name = dataset_name
        self.annotations_name = annotations_name
        
    def _get_samplerate(self, channel=None):
        # Same samplerate for all channels.
        # get the samplerate property of the dataset
        f = h5py.File(self.filepath,'r')
        data = f[self.dataset_name]
        samplerate = data.attrs['samplerate']
        f.close()
        return samplerate

    def _get_nsamples(self,channel=None):
        # get the dimensions of the data
        f = h5py.File(self.filepath,'r')
        data = f[self.dataset_name]
        nsamples = data.shape[1]
        f.close()
        return nsamples

    def _get_nchannels(self):
        # get the dimensions of the data
        f = h5py.File(self.filepath,'r')
        data = f[self.dataset_name]
        nchannels = data.shape[0]
        f.close()
        return nchannels

    def _get_annotations(self):
        # get the dimensions of the data
        f = h5py.File(self.filepath,'r')
        if self.annotations_name in f:
            annot = f[self.annotations_name][:]
        else:
            annot = None
        f.close()
        return annot

    def _load_data(self,channels,event_offsets,dur_samp,offset_samp):
        """        
        """
        # connect to the file and get the dataset
        f = h5py.File(self.filepath,'r')
        data = f[self.dataset_name]
        
        # allocate for data
	eventdata = np.empty((len(channels),len(event_offsets),dur_samp),
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
            eventdata[:,e,:] = data[channels,ssamp:esamp]

        # close the file
        f.close()
        
        return eventdata
    

