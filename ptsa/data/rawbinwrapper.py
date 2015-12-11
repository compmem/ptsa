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
import string
import struct
import os
from glob import glob

class RawBinWrapper(BaseWrapper):
    """
    Interface to data stored in binary format with a separate file for
    each channel.  
    """
    def __init__(self,dataroot,samplerate=None,format='int16',gain=1):
        """Initialize the interface to the data.  You must specify the
        dataroot, which is a string that contains the path to and
        root, up to the channel numbers, where the data are stored."""
        # set up the basic params of the data
        self._dataroot = dataroot
        self._samplerate = samplerate
        self._format = format
        self._gain = gain

        # see if can find them from a params file in dataroot
        self._params = self._get_params(dataroot)

        # set what we can get from the params 
        if self._params.has_key('samplerate'):
            self._samplerate = self._params['samplerate']
        if self._params.has_key('format'):
            self._format = self._params['format']
        if self._params.has_key('dataformat'):
            self._format = self._params['dataformat']
        if self._params.has_key('gain'):
            self._gain = self._params['gain']

        # set the nBytes and format str
        if self._format == 'single':
            self._nbytes = 4
            self._fmt_str = 'f'
        elif self._format == 'short' or self._format == 'int16':
            self._nbytes = 2
            self._fmt_str = 'h'
        elif self._format == 'double':
            self._nbytes = 8
            self._fmt_str = 'd'

        self._chanfiles = glob(self._dataroot+'.*[0-9]')
        # sorting because the order of the output from glob is
        # arbitrary (not strictly necessary, but nice to have
        # consistency):
        self._chanfiles.sort()
        self._nchannels = len(self._chanfiles)
        self._nsamples = None

        # collate channel info:
        numbers = []
        names = []
        for i in range(self._nchannels):
            numbers.append(np.int(self._chanfiles[i].split('.')[-1]))
            names.append(self._chanfiles[i].split('.')[-1])
        self._channel_info = np.rec.fromarrays(
            [numbers, names], names='number,name')
                    
    def _get_dataroot(self, channel=None):
        # Same dataroot for all channels:
        return self._dataroot            

    def _get_samplerate(self, channel=None):
        # Same samplerate for all channels:
        return self._samplerate

    def _get_nsamples(self,channel=None):
        # get the dimensions of the data
        # must open a valid channel and seek to the end
        if channel is not None:
            raise NotImplementedError('Channel cannot be specified!') 
        if self._nsamples is None:
            chanfile = open(self._chanfiles[0], 'rb')
            chanfile.seek(0, 2)
            if chanfile.tell() % self._nbytes != 0:
                raise ValueError(
                    'File length does not correspond to data format!')
            else:
                self._nsamples = chanfile.tell()/self._nbytes
        return self._nsamples

    def _get_nchannels(self):
        # get the dimensions of the data
        # must loop through directory identifying valid channels
        return self._nchannels

    def _get_channel_info(self):
        return self._channel_info
    
    def _get_annotations(self):
        # no annotations for raw data
        annot = None
        return annot

    def _get_params(self,dataroot):
        """Get parameters of the data from the dataroot."""
        params = {}

        # first look for dataroot.params file
        param_file = dataroot + '.params'
        if not os.path.isfile(param_file):
            # see if it's params.txt
            param_file = os.path.join(os.path.dirname(dataroot), 'params.txt')
            if not os.path.isfile(param_file):
                raise IOError(
                    'No params file found in '+str(dataroot)+
                    '. Params files must be in the same directory ' +
                    'as the EEG data and must be named \".params\" ' +
                    'or \"params.txt\".')        
        # we have a file, so open and process it
        for line in open(param_file,'r').readlines():
            # get the columns by splitting
            cols = line.strip().split()
            # set the params
            params[cols[0]] = eval(string.join(cols[1:]))
        if (not params.has_key('samplerate')) or (not params.has_key('gain')):
            raise ValueError(
                'Params file must contain samplerate and gain!\n' +
                'The following fields were supplied:\n' + str(params.keys()))
        # return the params dict
        return params
        

    def _load_data(self,channels,event_offsets,dur_samp,offset_samp):
        """
        """

        # allocate for data
        eventdata = np.empty((len(channels),len(event_offsets),dur_samp),
                             dtype=np.float)*np.nan

        # loop over channels
        for c, channel in enumerate(channels):
            # determine the file
            eegfname = self._dataroot+'.'+self._channel_info['name'][channel]
            # eegfname = '{}.{:0>3}'.format(self._dataroot,channel)
            if os.path.isfile(eegfname):
                efile = open(eegfname,'rb')
            else:
                raise IOError(
                    'EEG file not found: '+eegfname)
                    # 'EEG file not found for channel {:0>3} '.format(channel) +
                    # 'and file root {}\n'.format(self._dataroot))

            # loop over events
            for e, ev_offset in enumerate(event_offsets):
                # seek to the position in the file
                thetime = offset_samp + ev_offset
                efile.seek(self._nbytes * thetime,0)

                # read the data
                data = efile.read(int(self._nbytes * dur_samp))

                # convert from string to array based on the format
                # hard codes little endian
                data = np.array(struct.unpack(
                    '<' + str(len(data) / self._nbytes) +
                    self._fmt_str, data))

                # make sure we got some data
                if len(data) < dur_samp:
                    raise IOError(
                        'Event with offset ' + str(ev_offset) +
                        ' is outside the bounds of file ' + str(eegfname))

                # append it to the events
                eventdata[c, e, :] = data

        # multiply by the gain
	eventdata *= self._gain
	
        return eventdata

    dataroot = property(lambda self: self._get_dataroot())

# # This doesn't seem to work and also doesn't seem to belong in this file:
# def createEventsFromMatFile(matfile):
#     """Create an events data array with data wrapper information from
#     an events structure saved in a Matlab mat file."""
#     # load the mat file
#     mat = loadmat(matfile)

#     if 'events' not in mat.keys():
#         raise "\nError processing the Matlab file: %s\n" + \
#               "This file must contain an events structure" + \
#               "with the name \"events\" (case sensitive)!\n" +\
#               "(All other content of the file is ignored.)" % matfile 

#     # get the events
#     events = mat['events'][0]
    
#     # get num events
#     numEvents = len(events)

#     # determine the fieldnames and formats
#     fields = events[0]._fieldnames

#     def loadfield(events,field,dtype=None):
#         data = []
#         for x in events:
#             dat = getattr(x,field)
#             if len(dat) == 0:
#                 data.append(None)
#             else:
#                 data.append(dtype(dat[0]))
#         return data
    
#     # create list with array for each field
#     data = []
#     for f,field in enumerate(fields):
# 	# handle special cases
# 	if field == 'eegfile':
# 	    # get unique files
# 	    #eegfiles = np.unique(map(lambda x: str(x.eegfile),events))
# 	    #eegfiles = np.unique([str(x.eegfile[0]) for x in events])
#             eegfiles = np.unique(loadfield(events,field))
#             eegfiles = eegfiles[eegfiles!=None]
	    
# 	    # make dictionary of data wrapers for the unique eeg files
# 	    efile_dict = {}
# 	    for eegfile in eegfiles:
# 		efile_dict[eegfile] = RawBinWrapper(eegfile)

# 	    # Handle when the eegfile field is blank
# 	    efile_dict[''] = None
	
# 	    # set the eegfile to the correct data wrapper
            
# 	    # newdat = np.array(
#             #     map(lambda x: efile_dict[str(x.__getattribute__(field))],
#             #         events))
#             newdat = np.array([efile_dict[str(x.__getattribute__(field)[0])]
#                                for x in events])
			
# 	    # change field name to esrc
# 	    fields[f] = 'esrc'
# 	elif field == 'eegoffset':
#             # change the field name
#             fields[f] = 'eoffset'
#             newdat = np.array([x.__getattribute__(field)[0]
#                                for x in events])
# 	else:
# 	    # get the data in normal fashion
# 	    # newdat = np.array(
#             #     map(lambda x: x.__getattribute__(field),events))
#             newdat = np.array([x.__getattribute__(field)[0]
#                                for x in events])

# 	# append the data
# 	data.append(newdat)

#     # allocate for new array
#     newrec = np.rec.fromarrays(data,names=fields).view(Events)

#     return newrec



