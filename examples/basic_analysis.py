
import numpy as np
import pylab as pl

from ptsa.data import ArrayWrapper, Events
from ptsa.wavelet import phase_pow_multi

# some general info
nchan = 2
samplerate = 200
nsamples = samplerate*100
event_dur = samplerate*1
buf_dur = 1.0

# generate fake data
dat = np.random.rand(nchan, nsamples)
aw = ArrayWrapper(dat, samplerate)

# generate fake events
eoffset = np.arange(event_dur*2,nsamples-(2*event_dur),event_dur)
esrc = [aw]*len(eoffset)
nrec = len(eoffset)/2
recalled = [True]*nrec + [False]*(len(eoffset)-nrec)
events = Events(np.rec.fromarrays([esrc,eoffset,recalled],
                                  names='esrc,eoffset,recalled'))

# load in data with events (filter at the same time)
rdat = events[events.recalled==True].get_data(0, # channel
                                              1.0, # duration in sec
                                              0.0, # offset in sec
                                              buf_dur, # buffer in sec
                                              filt_freq = 20.,
                                              filt_type = 'low',
                                              keep_buffer=True
                                              )
ndat = events[events.recalled==False].get_data(0, # channel
                                               1.0, # duration in sec
                                               0.0, # offset in sec
                                               buf_dur, # buffer in sec
                                               filt_freq = 20.,
                                               filt_type = 'low',
                                               keep_buffer=True
                                               )

# calc wavelet power
freqs = np.arange(2,50,2)
rpow = phase_pow_multi(freqs,rdat,to_return='power')
npow = phase_pow_multi(freqs,ndat,to_return='power')

# remove the buffer now that we have filtered and calculated power
#for ts in [rdat,ndat,rpow,npow]:
#    ts = ts.remove_buffer(buf_dur)
# why does the above not work?
rdat = rdat.remove_buffer(buf_dur)
ndat = ndat.remove_buffer(buf_dur)
rpow = rpow.remove_buffer(buf_dur)
npow = npow.remove_buffer(buf_dur)
    
# plot ERP
pl.figure(1)
pl.clf()
pl.plot(rdat['time'],rdat.nanmean('events'),'r')
pl.plot(ndat['time'],ndat.nanmean('events'),'b')
pl.legend(('Recalled','Not Recalled'),loc=0)
pl.xlabel('Time (s)')
pl.ylabel('Voltage')

# plot power spectrum
pl.figure(2)
pl.clf()
pl.plot(rpow['freqs'],rpow.nanmean('events').nanmean('time'),'r')
pl.plot(npow['freqs'],npow.nanmean('events').nanmean('time'),'b')
pl.legend(('Recalled','Not Recalled'),loc=0)
pl.xlabel('Frequency (Hz)')
pl.ylabel('Power')

pl.show()
