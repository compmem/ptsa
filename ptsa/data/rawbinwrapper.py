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
from events import Events

# global imports
import numpy as np
import string
import struct
import os
from scipy.io import loadmat

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
        self.dataroot = dataroot
        self.samplerate = samplerate
        self.format = format
        self.gain = gain

        # see if can find them from a params file in dataroot
        self.params = self._get_params(dataroot)

        # set what we can from the params 
        if self.params.has_key('samplerate'):
            self.samplerate = self.params['samplerate']
        if self.params.has_key('format'):
            self.format = self.params['format']
        if self.params.has_key('dataformat'):
            self.format = self.params['dataformat']
        if self.params.has_key('gain'):
            self.gain = self.params['gain']

        # set the nBytes and format str
        if self.format == 'single':
            self.nBytes = 4
            self.fmtStr = 'f'
        elif self.format == 'short' or self.format == 'int16':
            self.nBytes = 2
            self.fmtStr = 'h'
        elif self.format == 'double':
            self.nBytes = 8
            self.fmtStr = 'd'

    def _get_params(self,dataroot):
        """Get parameters of the data from the dataroot."""
        # set default params
        params = {'samplerate':256.03,'gain':1.}

        # first look for dataroot.params file
        paramFile = dataroot + '.params'
        if not os.path.isfile(paramFile):
            # see if it's params.txt
            paramFile = os.path.join(os.path.dirname(dataroot),'params.txt')
            if not os.path.isfile(paramFile):
                #raise "file not found"  # fix this
                return params
        
        # we have a file, so open and process it
        for line in open(paramFile,'r').readlines():
            # get the columns by splitting
            cols = line.strip().split()
            # set the params
            params[cols[0]] = eval(string.join(cols[1:]))
        
        # return the params dict
        return params
        

    def _load_data(self,channel,event_offsets,dur_samp,offset_samp):
        """
        """

        # determine the file
	eegfname = '%s.%03i' % (self.dataroot,channel)
	if os.path.isfile(eegfname):
	    efile = open(eegfname,'rb')
	else:
	    # try unpadded lead
	    eegfname = '%s.%i' % (self.dataroot,channel)
	    if os.path.isfile(eegfname):
		efile = open(eegfname,'rb')
	    else:
		raise IOError('EEG file not found for channel %i and file root %s\n' 
			      % (channel,self.dataroot))
                
        # allocate for data
	eventdata = np.empty((len(event_offsets),dur_samp),
                             dtype=self.data.dtype)

	# loop over events
	for e,evOffset in enumerate(event_offsets):
	    # seek to the position in the file
	    thetime = offset_samp+evOffset
	    efile.seek(self.nBytes*thetime,0)

	    # read the data
	    data = efile.read(int(self.nBytes*dur_samp))

	    # convert from string to array based on the format
	    # hard codes little endian
	    data = np.array(struct.unpack('<'+str(len(data)/self.nBytes)+self.fmtStr,data))

	    # make sure we got some data
	    if len(data) < dur_samp:
		raise IOError('Event with offset %d is outside the bounds of file %s.\n'
			      % (evOffset,eegfname))

	    # append it to the events
            eventdata[e,:] = data

        # multiply by the gain
	eventdata *= self.gain
	
        return eventdata


def createEventsFromMatFile(matfile):
    """Create an events data array with data wrapper information from
    an events structure saved in a Matlab mat file."""
    # load the mat file
    mat = loadmat(matfile)

    if 'events' not in mat.keys():
        raise "\nError processing the Matlab file: %s\n" + \
              "This file must contain an events structure" + \
              "with the name \"events\" (case sensitive)!\n" +\
              "(All other content of the file is ignored.)" % matfile 

    # get the events
    events = mat['events'][0]
    
    # get num events
    numEvents = len(events)

    # determine the fieldnames and formats
    fields = events[0]._fieldnames

    def loadfield(events,field,dtype=None):
        data = []
        for x in events:
            dat = getattr(x,field)
            if len(dat) == 0:
                data.append(None)
            else:
                data.append(dtype(dat[0]))
        return data
    
    # create list with array for each field
    data = []
    for f,field in enumerate(fields):
	# handle special cases
	if field == 'eegfile':
	    # get unique files
	    #eegfiles = np.unique(map(lambda x: str(x.eegfile),events))
	    #eegfiles = np.unique([str(x.eegfile[0]) for x in events])
            eegfiles = np.unique(loadfield(events,field))
            eegfiles = eegfiles[eegfiles!=None]
	    
	    # make dictionary of data wrapers for the unique eeg files
	    efile_dict = {}
	    for eegfile in eegfiles:
		efile_dict[eegfile] = RawBinWrapper(eegfile)

	    # Handle when the eegfile field is blank
	    efile_dict[''] = None
	
	    # set the eegfile to the correct data wrapper
	    #newdat = np.array(map(lambda x: efile_dict[str(x.__getattribute__(field))],
            #                  events))
            newdat = np.array([efile_dict[str(x.__getattribute__(field)[0])]
                               for x in events])
			
	    # change field name to esrc
	    fields[f] = 'esrc'
	elif field == 'eegoffset':
            # change the field name
            fields[f] = 'eoffset'
            newdat = np.array([x.__getattribute__(field)[0]
                               for x in events])
	else:
	    # get the data in normal fashion
	    #newdat = np.array(map(lambda x: x.__getattribute__(field),events))
            newdat = np.array([x.__getattribute__(field)[0]
                               for x in events])

	# append the data
	data.append(newdat)

    # allocate for new array
    newrec = np.rec.fromarrays(data,names=fields).view(Events)

    return newrec


