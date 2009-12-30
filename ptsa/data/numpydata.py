#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

# local imports
from datawrapper import DataWrapper
# from events import Events,EegEvents
# from dimarray import DimArray,Dim
# from timeseries import TimeSeries

# global imports
import numpy as np
# import string
# import struct
# import os
# from scipy.io import loadmat

class NumpyData(DataWrapper):
    """
    Interface to data stored in numpy format (dummy interface for testing).  
    """
    def __init__(self,file_name):
        """
        Initialize the interface to the data. The file name must be specified.
        """
        # set up the basic params of the data
        self.file_name = file_name        

    def get_event_data(self):
        """
        Return data from the specified file.
        """
        return = TimeSeries(np.load(self.file_name),'dim1',1)
