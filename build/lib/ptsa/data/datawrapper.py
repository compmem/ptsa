#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

class DataWrapper(object):
    """
    Base class to provide interface to timeseries data.  
    """
    def get_event_data(self,channels,eventOffsets,
                       dur,offset,buf,
                       resampledRate=None,
                       filtFreq=None,filtType='stop',filtOrder=4,
                       keepBuffer=False):
        raise NotImplementedError
