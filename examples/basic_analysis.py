
import numpy as np

from ptsa.data import ArrayWrapper, Events

# some general info
nchan = 2
samplerate = 50
nsamples = samplerate*100
event_dur = samplerate*1

# generate fake data
dat = np.random.rand(nchan, nsamples)
aw = ArrayWrapper(dat, samplerate)

# generate fake events
eoffset = np.arange(event_dur,nsamples,event_dur)
esrc = [aw]*len(eoffset)
nrec = len(eoffset)/2
recalled = [True]*nrec + [False]*(len(eoffset)-nrec)
events = Events(np.rec.fromarrays([esrc,eoffset,recalled],
                                  names='esrc,eoffset,recalled'))

# load in data with events
rdat = events[events.recalled==True].get_data(0, # channel
                                              1.0, # duration in sec
                                              0.0, # offset in sec
                                              0.5, # buffer in sec
                                              )
ndat = events[events.recalled==False].get_data(0, # channel
                                               1.0, # duration in sec
                                               0.0, # offset in sec
                                               0.5, # buffer in sec
                                               )


# filter, wavelet power


# plot ERP


# plot power spectrogram


