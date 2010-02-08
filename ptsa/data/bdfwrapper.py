#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

# http://www.biosemi.com/faq/file_format.htm

# local imports
from basewrapper import BaseWrapper

# global imports
import numpy as np
import scipy as sp
import datetime
import os

# bdf_file = 'M03_ec.bdf'
#bdf_file = 'Newtest17-256.bdf'

class BdfWrapper(BaseWrapper):
    """
    Interface to data stored in European Data Format (EDF) and
    derivatives. In addition to EDF, the BioSemi Data Format (BDF) is
    currently supported.
    """
    def __init__(self, datafile_name, yearbase = 2000):
        """
        Initialize the interface to the data

        Parameters
        ----------
        datafile : {string}
            String specifying the path to the data file.
        year_base : {int},optional
            Only two year digits are stored for the recording date in
            the EDF and related formats. This number specifies how to
            convert to four digits. For recording dates during or
            after the year 2000 this number should be 2000, for
            earlier years, this number should be 1900. 
        """
        self.header = get_bdf_header(datafile_name, yearbase)
        # self.data = get_edf_data(datafile, self.header)

    def _load_data(self,channel,event_offsets,dur_samp,offset_samp):
        if isinstance(channel,str):
            chan, = np.nonzero(channel==self.header['label'])
            if len(chan) == 1:
                channel = chan[0]
            else:
                raise ValueError(
                    'Invalid channel label!\nChannel labels must be one of '+
                    'the following: '+str(self.header['labels'])+'\nGiven: '+
                    channel)

        # determine the file
	if os.path.isfile(self.header['filename']):
	    efile = open(self.header['filename'],'rb')
	else:
            raise IOError('Invalid filename: '+str(self.header['filename']))
            
        # allocate for data
	eventdata = np.empty((len(event_offsets),dur_samp),
                             dtype=np.float)*np.nan

        # Ensure array to allow for calculations below
        event_offsets = np.array(event_offsets)
        
        # The number of data records (blocks) that correspond to the offset:
        # event_offset_blocks = np.int16(event_offsets/self.header['dat_dur'])
        event_offset_blocks = np.int16(event_offsets/
                                       np.sum(self.header['samples']))
        # The remaining number of samples in the next data records (blocks):
        event_offset_inblocks = event_offsets%self.header['dat_dur']

        # The number of additional data records (blocks) to offset each sample:
        offset_samp_blocks = np.int16((event_offset_inblocks + offset_samp) /
                                    self.header['dat_dur'])
        # The remaining number of samples in the next data records (blocks)
        offset_samp_inblocks = ((event_offset_inblocks+offset_samp) %
                                self.header['dat_dur'])

	# loop over events
	for e,evOffset in enumerate(event_offsets):
	    # seek to the position in the file

            samples_in_block = (self.header['samples'][channel] *
                                self.header['dat_dur'])

            first_offset = dur_samp - (samples_in_block *
                                       np.int(evOffset / samples_in_block))
            if((first_offset<0)|(first_offset>=samples_in_block)):
                first_offset = 0
            durations = [(samples_in_block - first_offset)]#(
                # dur_samp - (samples_in_block *
                #             np.int(evOffset / samples_in_block))))]
            

            # if durations[0] <= 0:
            #     durations[0] = samples_in_block
                
            durations[0] = np.min([durations[0],dur_samp])
            
            while(np.sum(durations) < dur_samp):
                durations.append(
                    np.min([samples_in_block, (dur_samp - np.sum(durations))]))
            # print 'bla',durations,np.min(samples_in_block, (dur_samp - np.sum(durations))),samples_in_block,(dur_samp - np.sum(durations))

            startpoints = len(durations)
            
            #durations = evOffsets
            
            # startpoints = np.int(dur_samp/(self.header['samples'][channel]*
            #                               self.header['dat_dur']))+1
            # durations = (np.ones(startpoints) *
            #              (self.header['samples'][channel] *
            #               self.header['dat_dur']))
            # if (dur_samp%(self.header['samples'][channel] *
            #               self.header['dat_dur'])) != 0:
            #     durations[-1] = (dur_samp % (self.header['samples'][channel] *
            #                                  self.header['dat_dur']))
            print startpoints,durations,e,evOffset

            print event_offset_blocks[e],offset_samp_blocks[e],self.header['dat_dur'],np.sum(self.header['samples']),offset_samp_inblocks[e]

            thetimes = (self.header['header_len'] +
                        np.sum(self.header['samples'][:channel]) +
                        (np.sum(self.header['samples']) * np.array(
                            [event_offset_blocks[e] + block
                             for block in range(startpoints)])))
            thetimes[0] += first_offset
            
            # thetimes = self.header['header_len'] + np.array([
            #     ((event_offset_blocks[e] + block + offset_samp_blocks[e]) *
            #      self.header['dat_dur'] * np.sum(self.header['samples']))
            #     for block in range(startpoints)]) + offset_samp_inblocks[e]

            # thetimes -= np.sum(self.header['samples'])
            
            # thetimes = self.header['header_len'] + np.array(
            #     [((np.sum((event_offset_blocks[e] + block) *
            #               self.header['dat_dur'] *
            #               self.header['samples'][:(channel+1)]) +
            #        event_offset_inblocks[e]) +
            #       (np.sum(
            #           offset_samp_blocks * self.header['dat_dur'] *
            #           self.header['samples'][
            #               (channel+1):(channel+offset_samp_blocks+1)]) +
            #        offset_samp_inblocks))
            #      for block in range(startpoints)])
            print thetimes
            # data = np.empty(dur_samp,float)*np.nan
            data = np.empty((dur_samp,3),float)*np.nan
            dat_indx = 0
            dat_mult = 2**np.arange(0,self.header['dat_bytes']*8,8)
            for t,time in enumerate(thetimes):
                efile.seek(np.int(self.header['dat_bytes']*time),0)
                raw_data = efile.read(
                    np.int(self.header['dat_bytes']*durations[t]))
                # make sure we got some data
                if len(raw_data) < durations[t]:
                    raise IOError('Block '+str(t)+' of Event '+str(e)+
                                  ' with offset '+str(time)+
                                  ' is outside the bounds of the file.')

                # specific to BDF! Needs to be adapted for EDF &
                # related formats!
                for d in range(np.int(durations[t])):
                    data[dat_indx,:] = np.int16(sp.fromstring(
                        raw_data[(d*self.header['dat_bytes']):\
                                 ((d+1)*self.header['dat_bytes'])],np.uint8))
                    dat_indx += 1
            data[:,2][data[:,2]>=128] -= 256
            data = np.dot(data,dat_mult)
            # append it to the events
            eventdata[e,:] = data
            
        # multiply by the gain
        eventdata *= self.header['gain'][channel]
        return eventdata    

#from ptsa.data import BdfWrapper
#bdfwrap = BdfWrapper('/home/ctw/Christoph/Analyses/biosemi/Newtest17-256.bdf')
#tmp = bdfwrap._load_data(1,np.arange(0,600,300),300,0)

def get_bdf_header(datafile_name, yearbase):
    
    # create file object:
    if isinstance(datafile_name,str):
        datafile = file(datafile_name,'rb')
    else:
        raise ValueError('data_file must be a string specifying the path '+
                         'to the data file: '+str(datafile))
    
    header = {}
    header['filename'] = datafile_name
    header['id_code'] = datafile.read(8) # File identification code
    header['subj_id'] = datafile.read(80).strip() # Local subject identification
    header['rcrd_id'] = datafile.read(80).strip() # Local recording
                                                  # identification
    header['rec_start_date'] = datafile.read(8) # Start date of
                                                # recording: dd.mm.yy
    header['rec_start_time'] = datafile.read(8) # Start time of
                                                # recording: hh.mm.ss
    # create a datetime object with start date & time:
    header['rec_start_datetime'] = datetime.datetime(
        yearbase+int(header['rec_start_date'][6:]),
        int(header['rec_start_date'][3:5]),
        int(header['rec_start_date'][0:2]),int(header['rec_start_time'][0:2]),
        int(header['rec_start_time'][3:5]),int(header['rec_start_time'][6:]))
    header['header_len'] = int(datafile.read(8)) # Number of bytes in
                                                 # header record
    header['dat_format'] = datafile.read(44).strip() # Version of data format
    if header['dat_format'] == '24BIT':
        header['dat_bytes'] = 3
    else:
        header['dat_bytes'] = 2
        raise NotImplementedError('Currently only 24-Bit BDF support!')
    header['dat_records_num'] = int(datafile.read(8)) # Number of data
                                                      # records (-1 if unknown)
    header['dat_dur'] = int(datafile.read(8)) # Duration of data
                                              # record in seconds
    header['chan_num'] = int(datafile.read(4)) # Number of channels
    # Labels of the channels:
    header['labels'] = np.array([datafile.read(16).strip()
                                 for i in range(header['chan_num'])])
    # Transducer type:
    header['transducer'] = np.array([datafile.read(80).strip()
                                     for i in range(header['chan_num'])])
    # Physical dimensions of channels:
    header['phys_dim'] = np.array([datafile.read(8).strip()
                                   for i in range(header['chan_num'])])
    # Physical minimum in units of physical dimension:
    header['phys_min'] = np.array([int(datafile.read(8))
                                   for i in range(header['chan_num'])],float)
    # Physical maximum in units of physical dimension:
    header['phys_max'] = np.array([int(datafile.read(8))
                                   for i in range(header['chan_num'])],float)
    # Digital minimum:
    header['dig_min'] = np.array([int(datafile.read(8))
                                  for i in range(header['chan_num'])],float)
    # Digital maximum:
    header['dig_max'] = np.array([int(datafile.read(8))
                                  for i in range(header['chan_num'])],float)
    # Prefiltering:
    header['prefilt'] = np.array([datafile.read(80).strip()
                                  for i in range(header['chan_num'])])
    # Number of samples in each data record (sample-rate if duration
    # of data record == 1):
    header['samples'] = np.array([int(datafile.read(8))
                                  for i in range(header['chan_num'])],float)
    # Reserved:
    header['reserved'] = np.array([datafile.read(32).strip()
                                   for i in range(header['chan_num'])])

    # Gain (LSB value in the specified physical dimension of channels):
    header['gain'] = ((header['phys_max']-header['phys_min'])/
                      (header['dig_max']-header['dig_min']))
    header['sample_rate'] = header['samples']/header['dat_dur']

    # not sure:
    header['off'] = (np.array(header['phys_min'],float)-
                     (header['gain']*np.array(header['dig_min'],float)))
    invalid = header['gain']<0
    header['gain'][invalid] = 1
    header['off'][invalid] = 0
    header['calib'] = np.r_[np.atleast_2d(header['off']),
                            np.diag(header['gain'])]

    # unknown record size, determine correct value
    if header['dat_records_num'] == -1: 
        file_pos = datafile.tell()
        data_file.seek(0,2)
        file_end_pos = datafile.tell()
        header['dat_records_num_retrieved'] = int(np.floor(
            (file_end_pos-file_pos)/(np.sum(header['samples'])*3.0)))
        
    datafile.close()
    return(header)

def get_edf_data(datafile,header):
    # if data_file is a string, convert to file object:
    if isinstance(datafile,str):
        datafile = file(datafile,'rb')
    elif not isinstance(datafile,file):
        raise ValueError('data_file must be a valid file object or '+
                         'string specifying the path to the data file: '+
                         str(datafile))
    data = np.empty((header['chan_num'],header['samples']))*np.nan
    datafile.seek(header['header_len'])
    raw_data = datafile.read()


def get_header(data_file, year_base = 2000):
    # if data_file is a string, convert to file object:
    if isinstance(data_file,str):
        data_file = file(data_file,'rb')
    elif not isinstance(data_file,file):
        raise ValueError('data_file must be a valid file object or '+
                         'string specifying the filename: '+str(bdf_file))

    id_code = data_file.read(8)
    if id_code[1:] == 'BIOSEMI':
        return(get_BDF_EDF_data(data_file, year_base, id_code, 'BDF'))
    elif id_code == '0       ':
        return(get_BDF_EDF_data(data_file, year_base, id_code, 'EDF'))
    elif id_code[:3] == 'GDF':
        return(get_GDF_data(data_file, year_base, id_code, 'EDF'))
    else:
        raise IOError('This file format is not (yet) supported!')
        

def get_BDF_EDF_data(data_file, year_base = 2000):
    
    # if data_file is a string, convert to file object:
    if isinstance(data_file,str):
        data_file = file(data_file,'rb')
    elif not isinstance(data_file,file):
        raise ValueError(
            'data_file must be a valid file object or string with filename '+
            '/ path! '+str(data_file))

    header = {}
    header['id_code'] = data_file.read(8) # File identification code
    header['subj_id'] = data_file.read(80).strip() # Local subject identification
    header['rcrd_id'] = data_file.read(80).strip() # Local recording
                                                  # identification
    header['rec_start_date'] = data_file.read(8) # Start date of
                                                # recording: dd.mm.yy
    header['rec_start_time'] = data_file.read(8) # Start time of
                                                # recording: hh.mm.ss
    # create a datetime object with start date & time assuming
    # recording took place in the 21st century:
    header['rec_start_datetime'] = datetime.datetime(
        year_base+int(header['rec_start_date'][6:]),
        int(header['rec_start_date'][3:5]),
        int(header['rec_start_date'][0:2]),int(header['rec_start_time'][0:2]),
        int(header['rec_start_time'][3:5]),int(header['rec_start_time'][6:]))
    header['header_len'] = int(data_file.read(8)) # Number of bytes in
                                                 # header record
    header['dat_format'] = data_file.read(44).strip() # Version of data format
    header['dat_records_num'] = int(data_file.read(8)) # Number of data
                                                      # records (-1 if unknown)
    header['dat_dur'] = int(data_file.read(8)) # Duration of data
                                              # record in seconds
    header['chan_num'] = int(data_file.read(4)) # Number of channels
    # Labels of the channels:
    header['labels'] = [data_file.read(16).strip()
                        for i in range(header['chan_num'])]
    # Transducer type:
    header['transducer'] = [data_file.read(80).strip()
                            for i in range(header['chan_num'])]
    # Physical dimensions of channels:
    header['phys_dim'] = [data_file.read(8).strip()
                          for i in range(header['chan_num'])]
    # Physical minimum in units of physical dimension:
    header['phys_min'] = [int(data_file.read(8))
                          for i in range(header['chan_num'])]
    # Physical maximum in units of physical dimension:
    header['phys_max'] = [int(data_file.read(8))
                          for i in range(header['chan_num'])]
    # Digital minimum:
    header['dig_min'] = [int(data_file.read(8))
                         for i in range(header['chan_num'])]
    # Digital maximum:
    header['dig_max'] = [int(data_file.read(8))
                         for i in range(header['chan_num'])]
    # Prefiltering:
    header['prefilt'] = [data_file.read(80).strip()
                         for i in range(header['chan_num'])]
    # Number of samples in each data record (sample-rate if duration
    # of data record == 1):
    header['samples'] = [int(data_file.read(8))
                         for i in range(header['chan_num'])]
    # Reserved:
    header['reserved'] = [data_file.read(32).strip()
                          for i in range(header['chan_num'])]

    # Gain (LSB value in the specified physical dimension of channels):
    header['gain'] = ((np.array(header['phys_max'],float)-
                       np.array(header['phys_min'],float))/
                      (np.array(header['dig_max'],float)-
                       np.array(header['dig_min'],float)))
    header['sample_rate'] = np.array(header['samples'],float)/header['dat_dur']

    # not sure:
    header['off'] = (np.array(header['phys_min'],float)-
                     (header['gain']*np.array(header['dig_min'],float)))
    invalid = header['gain']<0
    header['gain'][invalid] = 1
    header['off'][invalid] = 0
    header['calib'] = np.r_[np.atleast_2d(header['off']),
                            np.diag(header['gain'])]

    # unknown record size, determine correct value
    if header['dat_records_num'] == -1: 
        file_pos = data_file.tell()
        data_file.seek(0,2)
        file_end_pos = data_file.tell()
        header['dat_records_num_retrieved'] = int(np.floor(
            (file_end_pos-file_pos)/(np.sum(header['samples'])*3.0)))

    
