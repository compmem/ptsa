
import numpy as np
import pylab as pl

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
eoffset = np.arange(event_dur,nsamples-event_dur,event_dur)
esrc = [aw]*len(eoffset)
nrec = len(eoffset)/2
recalled = [True]*nrec + [False]*(len(eoffset)-nrec)
events = Events(np.rec.fromarrays([esrc,eoffset,recalled],
                                  names='esrc,eoffset,recalled'))

# load in data with events (filter at the same time)
rdat = events[events.recalled==True].get_data(0, # channel
                                              1.0, # duration in sec
                                              0.0, # offset in sec
                                              0.5, # buffer in sec
                                              filt_freq = 20,
                                              filt_type = 'low',
                                              
                                              )
ndat = events[events.recalled==False].get_data(0, # channel
                                               1.0, # duration in sec
                                               0.0, # offset in sec
                                               0.5, # buffer in sec
                                               filt_freq = 20,
                                               filt_type = 'low',
                                               )


# calc wavelet power


# plot ERP
pl.figure(1)
pl.clf()
pl.plot(rdat['time'],rdat.mean('events'),'r')
pl.plot(ndat['time'],ndat.mean('events'),'b')
pl.legend(('Recalled','Not Recalled'),loc=0)
pl.xlabel('Time (s)')
pl.ylabel('Voltage')

# plot power spectrogram


pl.show()
