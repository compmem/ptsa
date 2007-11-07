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

# split out two conditions (recalled and not recalled)
print "Filtering events..."
rInd = ev.filterIndex('recalled==1')
nInd = ev.filterIndex('recalled==0')

# do sample erp by getting raw eeg and doing an average for a
# single channel

# get power for the events for a range of freqs

# we leave the buffer on after getting the data, but pull it off
# in the call to tsPhasePow
freqs = range(2,81,2)
chan = 27
durationMS = 2500
offsetMS = -500
bufferMS = 1000
resampledRate = 200
filtFreq = [58.0,62.0]

# load the eeg data
print "Loading EEG data..."
rEEG = ev[rInd].getDataMS(chan,
                          durationMS,
                          offsetMS,
                          bufferMS,
                          resampledRate,
                          filtFreq=filtFreq,
                          keepBuffer=True)
nEEG = ev[nInd].getDataMS(chan,
                          durationMS,
                          offsetMS,
                          bufferMS,
                          resampledRate,
                          filtFreq=filtFreq,
                          keepBuffer=True)

# power for recalled events
print "Calculating power..."
rRes = wavelet.tsPhasePow(freqs,
                          rEEG,
                          verbose=True,powOnly=True)
# power for not recalled events
nRes = wavelet.tsPhasePow(freqs,
                          nEEG,
                          verbose=True,powOnly=True)

# get mean power across events (axis=1)
print "Taking mean power..."
rPow = N.squeeze(N.mean(N.log10(rRes.data),rRes.dim('event')))
nPow = N.squeeze(N.mean(N.log10(nRes.data),nRes.dim('event')))

print "Generating plots..."
fig = 0

# erp
fig+=1
pylab.figure(fig)
pylab.plot(rEEG['time'],rEEG.data.mean(axis=rEEG.dim('event')),'r')
pylab.plot(nEEG['time'],nEEG.data.mean(axis=nEEG.dim('event')),'b')
pylab.legend(('Recalled','Not Recalled'))
pylab.xlabel('Time (ms)')
pylab.ylabel('Voltage (mV)')

# power spectrum
fig+=1
pylab.figure(fig)
pylab.plot(rRes['freq'],N.squeeze(N.mean(rPow,1)),'r')
pylab.plot(nRes['freq'],N.squeeze(N.mean(nPow,1)),'b')
pylab.legend(('Recalled','Not Recalled'))
pylab.xlabel('Frequency (Hz)')
pylab.ylabel(r'Power ($log_{10}(mV^2)$)')

# plot the diff in mean power
fig+=1
pylab.figure(fig)
pylab.contourf(rRes['time'],rRes['freq'],rPow-nPow)
pylab.colorbar()
pylab.xlabel('Time (ms)')
pylab.ylabel('Frequency (Hz)')
pylab.title('SME (diff in power) for channel %d' % (chan))

# show the plots
pylab.show()


