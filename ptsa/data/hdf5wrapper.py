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
                 annotations_name='annotations',
                 create_dataset=False,
                 samplerate=None, nchannels=None, nsamples=None,
                 data=None, dtype=np.float32, annotations=None, **hdf5opts):
        """
        Initialize the interface to the data.
        """
        # set up the basic params of the data
        self.filepath = filepath
        self.dataset_name = dataset_name
        self.annotations_name = annotations_name

        # see if create dataset
        if create_dataset:
            # must provide samplerate, nchannels, and data (or dtype)
            # XXX eventually put in sanity checks for them
            # connect to the file and get the dataset
            f = h5py.File(self.filepath,'a')

            # create the dataset
            if self.dataset_name in f:
                d = f[self.dataset_name]
            else:
                # must provide either data or dtype/nchannels
                if data is None:
                    # use dtype and nchannels
                    if nsamples is None:
                        nsamples = 1
                    d = f.create_dataset(self.dataset_name,
                                         (nchannels,nsamples),
                                         dtype=dtype,**hdf5opts)
                else:
                    # use the data
                    d = f.create_dataset(self.dataset_name,
                                         data=data, **hdf5opts)
            if not 'samplerate' in d.attrs:
                # must have provided samplerate
                if samplerate is None:
                    raise ValueError("You must specify a samplerate " +
                                     "if the dataset does not already exist.")
                # set the samplerate
                d.attrs['samplerate'] = samplerate    

            # create annotations if necessary
            if not annotations is None:
                if self.annotations_name in f:
                    raise ValueError("Told to create dataset annotations, " +
                                     "but %s already exists." %
                                     self.annotations_name)
                a = f.create_dataset(self.annotations_name,
                                     data=annotations, **hdf5opts)
            
            # close the hdf5 file
            f.close()
            
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

    def append_data(self, data):
        """
        Must be all channels.
        """
        # connect to the file and get the dataset
        f = h5py.File(self.filepath,'a')

        # get the dataset (must already exist)
        d = f[self.dataset_name]

        # check data size
        if data.shape[0] != d.shape[0]:
            raise ValueError("New data must have the same number of channels: %d." %
                             d.shape[0])

        # reshape to hold new data
        cursamp = d.shape[1]
        newsamp = data.shape[1]
        d.shape = (d.shape[0], cursamp+newsamp)

        # append the data
        d[:,cursamp:cursamp+newsamp] = data

        # close the file
        f.close()

    def set_channel_data(self, channel, data):
        """
        Set the data for an entire channel.  Will reshape the nsamples
        of the entire dataset to match.
        """
        # connect to the file and get the dataset
        f = h5py.File(self.filepath,'a')

        # get the dataset (must already exist)
        d = f[self.dataset_name]
        
        # reshape if necessary
        cursamp = d.shape[1]
        newsamp = len(data)
        d.shape = (d.shape[0], newsamp)

        # set the data
        d[channel,:] = data

        # close the file
        f.close()
