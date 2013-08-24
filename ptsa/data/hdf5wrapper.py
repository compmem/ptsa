#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

# global imports
import numpy as np
import h5py

# local imports
from basewrapper import BaseWrapper
from timeseries import TimeSeries

class HDF5Wrapper(BaseWrapper):
    """
    Interface to data stored in an HDF5 file.
    """
    def __init__(self, filepath, dataset_name='data',
                 annotations_name='annotations',
                 channel_info_name='channel_info',
                 data=None, file_dtype=None, apply_gain=True, gain_buffer=.005,
                 samplerate=None, nchannels=None, nsamples=None,
                 annotations=None, channel_info=None, **hdf5opts):
        """
        Initialize the interface to the data.

        Much documentation is needed here.

        For example, here is one way to create an HDF5 dataset from a
        TimeSeries instance:
        
        HDF5Wrapper('data.hdf5', data=data, compression='gzip')

        Now let's say the TimeSeries is float64, but you want to save
        space (and lose significant digits), you can specify a
        file_dtype, which will apply a gain factor to ensure you
        retain as much data accuracy as possible.  Here's how you can
        save the data in int16:

        HDF5Wrapper('data.hdf5', data=data, file_dtype=np.int16, compression='gzip')
        
        """
        # set up the basic params of the data
        self.filepath = filepath
        self.dataset_name = dataset_name
        self.annotations_name = annotations_name
        self.channel_info_name = channel_info_name
        self.apply_gain = apply_gain
        self.gain_buffer = gain_buffer
        self.gain = None
        self.hdf5opts = hdf5opts
        
        self.file_dtype = file_dtype
        self.data_dtype = None
        
        # see if create dataset
        if not data is None:
            # must provide samplerate and data
            # connect to the file and get the dataset
            f = h5py.File(self.filepath,'a')

            # use the data to create a dataset
            self.data_dtype = data.dtype
            d = f.create_dataset(self.dataset_name,
                                 data=self._data_to_file(data),
                                 **hdf5opts)
            d.attrs['data_dtype'] = data.dtype.char
            d.attrs['gain'] = self.gain

            if not 'samplerate' in d.attrs:
                # must have provided samplerate
                if isinstance(data, TimeSeries):
                    # get the samplerate from the TimeSeries
                    samplerate = data.samplerate
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

            # create channel_info if necessary
            if not channel_info is None:
                if self.channel_info_name in f:
                    raise ValueError("Told to create dataset channel_info, " +
                                     "but %s already exists." %
                                     self.channel_info_name)
                c = f.create_dataset(self.channel_info_name,
                                     data=channel_info, **hdf5opts)

            # close the hdf5 file
            f.close()
        else:
            # connect to the file and get info
            f = h5py.File(self.filepath,'r')
            d = f[self.dataset_name]
            self.data_dtype = np.dtype(d.attrs['data_dtype'])
            self.file_dtype = d.dtype
            self.gain = d.attrs['gain']
            
    def _data_to_file(self, data):
        # process the datatypes
        if self.file_dtype is None:
            # load from data
            self.file_dtype = data.dtype
        else:
            # make sure it's a dtype
            if not isinstance(self.file_dtype, np.dtype):
                try:
                    self.file_dtype = np.dtype(self.file_dtype)
                except:
                    ValueError("file_dtype should be a numpy dtype.")

        # process the gain
        if self.gain is None:
            # default to 1.0
            self.gain = 1.0
            # calc it if we are going from float to int
            if (self.file_dtype.kind == 'i') and (self.data_dtype.kind == 'f'):
                fr = np.iinfo(self.file_dtype).max*2
                dr = np.abs(data).max()*2 * (1.+self.gain_buffer)
                self.gain = dr/fr
                
        # calc and apply gain if necessary
        if self.apply_gain and self.gain != 1.0:
            return np.asarray(data/self.gain,dtype=self.file_dtype)
        else:
            return np.asarray(data,dtype=self.file_dtype)

    def _data_from_file(self, data):
        # see if apply gain we've already calculated
        if self.apply_gain and self.gain != 1.0:
            return np.asarray(data*self.gain, dtype=self.data_dtype)
        else:
            return np.asarray(data, dtype=self.data_dtype)

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

    def _set_annotations(self, annotations):
        # get the dimensions of the data
        f = h5py.File(self.filepath,'a')
        if self.annotations_name in f:
            del f[self.annotations_name]

        a = f.create_dataset(self.annotations_name,
                             data=annotations, **self.hdf5opts)
        f.close()

    def _get_channel_info(self):
        # get the dimensions of the data
        f = h5py.File(self.filepath,'r')
        if self.channel_info_name in f:
            chan_info = f[self.channel_info_name][:]
        else:
            chan_info = None
        f.close()
        return chan_info

    def _set_channel_info(self, channel_info):
        # get the dimensions of the data
        f = h5py.File(self.filepath,'a')
        if self.channel_info_name in f:
            del f[self.channel_info_name]

        a = f.create_dataset(self.channel_info_name,
                             data=channel_info, **self.hdf5opts)
        f.close()

    def _load_data(self,channels,event_offsets,dur_samp,offset_samp):
        """        
        """
        # connect to the file and get the dataset
        f = h5py.File(self.filepath,'r')
        data = f[self.dataset_name]
        
        # allocate for data
	eventdata = np.empty((len(channels),len(event_offsets),dur_samp),
                             dtype=self.data_dtype)*np.nan

	# loop over events
	for e,evOffset in enumerate(event_offsets):
            # set the range
            ssamp = offset_samp+evOffset
            esamp = ssamp + dur_samp
            
            # check the ranges
            if ssamp < 0 or esamp > data.shape[1]:
                raise IOError('Event with offset '+str(evOffset)+
                              ' is outside the bounds of the data.')
            eventdata[:,e,:] = self._data_from_file(data[channels,ssamp:esamp])

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
        d[:,cursamp:cursamp+newsamp] = self._data_to_file(data)

        # close the file
        f.close()

    def set_channel_data(self, channel, data):
        """
        Set the data for an entire channel.  Will reshape the nsamples
        of the entire dataset to match, throwing out data if smaller.
        """
        # connect to the file and get the dataset
        f = h5py.File(self.filepath,'a')

        # get the dataset (must already exist)
        d = f[self.dataset_name]
        
        # reshape if necessary
        cursamp = d.shape[1]
        newsamp = len(data)
        if cursamp != newsamp:
            d.shape = (d.shape[0], newsamp)

        # set the data
        d[channel,:] = self._data_to_file(data)

        # close the file
        f.close()
