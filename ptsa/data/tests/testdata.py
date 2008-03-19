#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


from ptsa.data import Dim


import numpy as N

class TestData():
    def __init__(self):
        # create 10 Hz sine waves at 200 and 50 Hz 4000ms long
        numSecs = 4.
        numPoints = int(numSecs*200.)
        Hz = 10
        d200_10 = N.sin(N.arange(numPoints,dtype=N.float)*2*N.pi*Hz*numSecs/numPoints)
        Hz = 5
        d200_5 = N.sin(N.arange(numPoints,dtype=N.float)*2*N.pi*Hz*numSecs/numPoints)
        self.dat200 = N.array([d200_10,d200_5])
        # calc the time range in MS
        offset = -200
        duration = numPoints
        samplesize = 1000./200.
        sampStart = offset*samplesize
        sampEnd = sampStart + (duration-1)*samplesize
        timeRange = N.linspace(sampStart,sampEnd,duration)
        self.dims200 = [Dim('channel',N.arange(self.dat200.shape[0])),
                        Dim('time',timeRange,'ms')]
        
        numSecs = 4.
        numPoints = int(numSecs*50.)
        Hz = 10
        d50_10 = N.sin(N.arange(numPoints,dtype=N.float)*2*N.pi*Hz*numSecs/numPoints)
        Hz = 5
        d50_5 = N.sin(N.arange(numPoints,dtype=N.float)*2*N.pi*Hz*numSecs/numPoints)
        self.dat50 = N.array([d50_10,d50_5])
        # calc the time range in MS
        offset = -50
        duration = numPoints
        samplesize = 1000./50.
        sampStart = offset*samplesize
        sampEnd = sampStart + (duration-1)*samplesize
        timeRange = N.linspace(sampStart,sampEnd,duration)
        self.dims50 = [Dim('channel',N.arange(self.dat50.shape[0])),
                       Dim('time',timeRange,'ms')]
 
