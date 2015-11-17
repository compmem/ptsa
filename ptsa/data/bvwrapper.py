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
import os.path
from ConfigParser import SafeConfigParser
import io

class BVWrapper(BaseWrapper):
    """
    Interface to data stored in a BrainVision Data Format.
    """
    def __init__(self, filepath):
        """
        Initialize the interface to the data.

        Parameters
        ----------
        filepath : string
            String specifiying the header filename (*.vhdr), with full
            path if applicable.
        """
        # set up the basic params of the data
        if os.path.exists(filepath):
            self.filepath = filepath
            self.filedir = os.path.split(filepath)[0]
        else:
            raise IOError(str(filepath)+'\n does not exist!\n'+
                          'Valid path to data file is needed!')

        # read in the info about the data from the header
        cp = SafeConfigParser()
        lines = open(filepath,'r').readlines()

        # must remove the first lines b/c they are not standard INI format
        # also remove everything after [Comment] b/c it doesn't parse either
        ind = None
        for i,line in enumerate(lines):
            if line.strip() == '[Comment]':
                ind = i
                break
        # try:
        #     ind = lines.index('[Comment]\r\r\n')
        # except ValueError:
        #     try:
        #         ind = lines.index('[Comment]\r\n')
        #     except ValueError:
        #         try:
        #             ind = lines.index('[Comment]\n')
        #         except ValueError:
        #             ind = None

        # join the lines back now that we've cleaned them up
        hdr_string = ''.join(lines[1:ind])

        # now read it in
        cp.readfp(io.BytesIO(hdr_string))

        # extract the info we need
        self._binaryformat = cp.get('Binary Infos','binaryformat')
        self._nchannels = int(cp.get('Common Infos','numberofchannels'))
        self._data_orient = cp.get('Common Infos','dataorientation')
        self._data_file = os.path.join(self.filedir,cp.get('Common Infos','datafile'))
        self._samplerate = float(10e5)/int(cp.get('Common Infos','samplinginterval'))
        self._markerfile = os.path.join(self.filedir,cp.get('Common Infos','markerfile'))

        # read in scale factors for each channel (and other info)
        numbers = []
        names = []
        scales = []
        units = []
        #self._channel_scale = np.ones(self._nchannels)
        for i in range(self._nchannels):
            info = cp.get('Channel Infos','Ch%d'%(i+1)).split(',')
            #self._channel_scale[i] = float(info[2])
            numbers.append(i+1)
            names.append(info[0])
            scales.append(float(info[2]))
            units.append(unicode(info[3],'utf-8'))
        # try and get the impedances
        impedances = np.ones(len(names))*-1
        for i,line in enumerate(lines[ind:]):
            if 'Impedance' in line:
                # found impedances, try and read them
                skipped = 0
                for l,li in enumerate(lines[ind+i+1:]):
                    info = li.strip().split(' ')
                    cname = info[1][:-1]
                    if cname in names:
                        impedances[names.index(cname)] = int(info[2])
                break
        self._channel_info = np.rec.fromarrays([numbers,names,scales,
                                                units,impedances],
                                               names='number,name,scale,unit,impedance')
            
        # process the binary format
        if self._binaryformat == 'INT_16':
            self._samplesize = 2
            self._dtype = np.dtype(np.int16)
        elif self._binaryformat == 'IEEE_FLOAT_32':
            self._samplesize = 4
            self._dtype = np.dtype(np.float32)
        else:
            raise ValueError('Unknown binary format: %s\n' % self._binaryformat) 

        # open the file to figure out the nsamples
        mm = np.memmap(self._data_file,dtype=self._dtype,
                       mode='r')
        self._nsamples = mm.shape[0]/self._nchannels


    def _get_nchannels(self):
        return self._nchannels

    def _get_channel_info(self):
        return self._channel_info

    def _get_nsamples(self, channel=None):
        return self._nsamples

    def _get_samplerate(self, channel=None):
        return self._samplerate

    def _get_annotations(self):
        # read in from annotations file (must strip off first lines)
        cp = SafeConfigParser()
        lines = open(self._markerfile,'r').readlines()
        cp.readfp(io.BytesIO(''.join(lines[2:])))

        # get the marker info
        markers = cp.items('Marker Infos')

        # process them
        index = []
        onsets = np.empty(len(markers))
        durations = []
        annots = []

        # see if subtract 1 because starts at 1 instead of 0
        sub_one = False
        for i in range(len(markers)):
            index.append(int(markers[i][0][2:]))
            info = markers[i][1].split(',')
            annots.append(info[1])
            # convert onset to seconds (subtracting 1 for actual offset)
            onsets[i] = (long(info[2]))/self._samplerate
            # save duration (for now, keep as string like in EDF)
            durations.append(info[3])
            if sub_one == False and info[0] == 'New Segment' and long(info[2])==1:
                # we need to sub_one
                sub_one = True
        if sub_one:
            onsets -= long(1)/self._samplerate
            
        # convert to rec array
        annotations = np.rec.fromarrays([onsets,durations,annots],
                                        names='onsets,durations,annotations')

        # sort by index and return
        return annotations[np.argsort(index)]
    
    def _load_data(self,channels,event_offsets,dur_samp,offset_samp):
        """        
        """
        # allocate for data
	eventdata = np.empty((len(channels),len(event_offsets),dur_samp),
                             dtype=np.float64)*np.nan

        # Memmap to the file
        mm = np.memmap(self._data_file,dtype=self._dtype,
                       mode='r',shape=(self._nsamples,self._nchannels))
        
	# loop over events
        for e,ev_offset in enumerate(event_offsets):
            # set the range
            ssamp = offset_samp+ev_offset

            if (ssamp + dur_samp - 1) > self._nsamples:
                raise IOError('Event with offset '+str(ev_offset)+
                              ' is outside the bounds of the data.')

            # only pick the channels of interest and scale
            dat = np.multiply(mm[ssamp:ssamp+dur_samp,channels],
                              self._channel_info['scale'][channels])
                              #self._channel_scale[channels])
            eventdata[:,e,:] = dat.T

        return eventdata    


