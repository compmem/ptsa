import numpy as N
# Can't use latex on the cluster
#import matplotlib
#matplotlib.rc('text', usetex = True)
import pylab
import pdb

from pyeeg.data.rawbinarydata import createEventsFromMatFile
from pyeeg import wavelet


# hypothetical test case

# load events
print "Loading events..."
ev = createEventsFromMatFile('/home1/per/eeg/free/CH012/events/events.mat')

# we leave the buffer on after getting the data, but pull it off
# in the call to tsPhasePow
freqs = range(2,81,2)
chan = 27
dur = 2.5
offset = .500
buf = 1.000
resampledRate = 200
filtFreq = [58.0,62.0]

# load the eeg data
print "Loading EEG data..."
rEEG = ev.select(ev['recalled']==1).get_data(chan,
                                             dur,
                                             offset,
                                             buf,
                                             resampledRate,
                                             filtFreq=filtFreq,
                                             keepBuffer=True)
nEEG = ev.select(ev['recalled']==0).get_data(chan,
                                             dur,
                                             offset,
                                             buf,
                                             resampledRate,
                                             filtFreq=filtFreq,
                                             keepBuffer=True)

# power for recalled events
print "Calculating power..."
rRes = wavelet.tsPhasePow(freqs,
                          rEEG,
                          verbose=True,toReturn='pow')
# power for not recalled events
nRes = wavelet.tsPhasePow(freqs,
                          nEEG,
                          verbose=True,toReturn='pow')

# get mean power across events (axis=1)
print "Taking mean power..."
rPow = rRes.apply_func(N.log10).aggregate('event',N.mean)
nPow = nRes.apply_func(N.log10).aggregate('event',N.mean)

print "Generating plots..."
fig = 0

# erp
fig+=1
pylab.figure(fig)
pylab.plot(rEEG['time'],rEEG.aggregate('event',N.mean).data,'r')
pylab.plot(nEEG['time'],nEEG.aggregate('event',N.mean).data,'b')
pylab.legend(('Recalled','Not Recalled'))
pylab.xlabel('Time (s)')
pylab.ylabel('Voltage (mV)')

# power spectrum
fig+=1
pylab.figure(fig)
pylab.plot(rPow['freq'],N.squeeze(rPow.aggregate('time',N.mean).data),'r')
pylab.plot(nPow['freq'],N.squeeze(nPow.data.mean(nPow.dim('time'))),'b')
pylab.legend(('Recalled','Not Recalled'))
pylab.xlabel('Frequency (Hz)')
pylab.ylabel(r'Power ($log_{10}(mV^2)$)')

# plot the diff in mean power
fig+=1
pylab.figure(fig)
pylab.contourf(rPow['time'],rPow['freq'],rPow.data-nPow.data)
pylab.colorbar()
pylab.xlabel('Time (s)')
pylab.ylabel('Frequency (Hz)')
pylab.title('SME (diff in power) for channel %d' % (chan))

# show the plots
pylab.show()


