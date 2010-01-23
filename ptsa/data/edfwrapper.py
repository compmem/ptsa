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
import datetime

# bdf_file = 'M03_ec.bdf'
bdf_file = 'Newtest17-256.bdf'

class EdfWrapper(BaseWrapper):
    """
    Interface to data stored in European Data Format (EDF) and
    derivatives. In addition to EDF, the BioSemi Data Format (BDF) is
    currently supported.
    """
    def __init__(self, datafile, yearbase = 2000):
        """
        Initialize the interface to the data

        Parameters
        ----------
        datafile : {file,string}
            File object for the data file or string specifying the
            path to the data file.
        year_base : {int},optional
            Only two year digits are stored for the recording date in
            the EDF and related formats. This number specifies how to
            convert to four digits. For recording dates during or
            after the year 2000 this number should be 2000, for
            earlier years, this number should be 1900. 
        """
    # if isinstance(datafile,str):
    #     datafile = file(datafile,'rb')
    # elif not isinstance(datafile,file):
    #     raise ValueError('data_file must be a valid file object or '+
    #                      'string specifying the path to the data file: '+
    #                      str(datafile))

    self.header = get_edf_header(datafile, yearbase)
    self.data = get_edf_data(datafile, self.header)

    # idcode = datafile.read(8)
    # datafile.close()
    # if idcode[1:] == 'BIOSEMI':
    #     self.header,self.data = edfdata(datafile, yearbase, idcode, 'BDF')
    # elif idcode == '0       ':
    #     self.header,self.data = edfdata(datafile, yearbase, idcode, 'EDF')
    # # elif id_code[:3] == 'GDF':
    # #     return(get_GDF_data(datafile, yearbase, idcode, 'EDF'))
    # else:
    #     raise IOError('This file format is not (yet) supported!\n'+
    #                   str(id_code))
    

def get_edf_header(datafile, yearbase):
    # if data_file is a string, convert to file object:
    if isinstance(datafile,str):
        datafile = file(datafile,'rb')
    elif not isinstance(datafile,file):
        raise ValueError('data_file must be a valid file object or '+
                         'string specifying the path to the data file: '+
                         str(datafile))
    header = {}
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
    header['dat_records_num'] = int(datafile.read(8)) # Number of data
                                                      # records (-1 if unknown)
    header['dat_dur'] = int(datafile.read(8)) # Duration of data
                                              # record in seconds
    header['chan_num'] = int(datafile.read(4)) # Number of channels
    # Labels of the channels:
    header['labels'] = [datafile.read(16).strip()
                        for i in range(header['chan_num'])]
    # Transducer type:
    header['transducer'] = [datafile.read(80).strip()
                            for i in range(header['chan_num'])]
    # Physical dimensions of channels:
    header['phys_dim'] = [datafile.read(8).strip()
                          for i in range(header['chan_num'])]
    # Physical minimum in units of physical dimension:
    header['phys_min'] = [int(datafile.read(8))
                          for i in range(header['chan_num'])]
    # Physical maximum in units of physical dimension:
    header['phys_max'] = [int(datafile.read(8))
                          for i in range(header['chan_num'])]
    # Digital minimum:
    header['dig_min'] = [int(datafile.read(8))
                         for i in range(header['chan_num'])]
    # Digital maximum:
    header['dig_max'] = [int(datafile.read(8))
                         for i in range(header['chan_num'])]
    # Prefiltering:
    header['prefilt'] = [datafile.read(80).strip()
                         for i in range(header['chan_num'])]
    # Number of samples in each data record (sample-rate if duration
    # of data record == 1):
    header['samples'] = [int(datafile.read(8))
                         for i in range(header['chan_num'])]
    # Reserved:
    header['reserved'] = [datafile.read(32).strip()
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
        file_pos = datafile.tell()
        data_file.seek(0,2)
        file_end_pos = datafile.tell()
        header['dat_records_num_retrieved'] = int(np.floor(
            (file_end_pos-file_pos)/(np.sum(header['samples'])*3.0)))
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
    
    
value = sp.fromstring(raw_data[0:3], uint8).tolist()
if value[2] >= 128:	# negative
    value = value[0] + value[1] * 2**8 + (value[2] - 256) * 2**16
else:			# positive
    value = value[0] + value[1] * 2**8 + value[2] * 2**16
        


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

    
