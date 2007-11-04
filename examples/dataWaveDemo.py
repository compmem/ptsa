import numpy as N
# Can't use latex on the cluster
#import matplotlib
#matplotlib.rc('text', usetex = True)
import pylab
import pdb

from pyeeg.data.events import createEventsFromMatFile
from pyeeg import wavelet


def testcase():
    # hypothetical test case

    # load events
    print "Loading events..."
    ev = createEventsFromMatFile('/home1/per/eeg/free/CH012/events/events.mat')
    
    # split out two conditions (recalled and not recalled)
    rInd = ev.filterIndex('recalled==1')
    nInd = ev.filterIndex('recalled==0')

    # do sample erp by getting raw eeg and doing an average for a
    # single channel

    # get power for the events for a range of freqs

    # we leave the buffer on after getting the data, but pull it off
    # in the call to tfPhasePow
    freqs = range(2,81,2)
    chan = 27
    durationMS = 2500
    offsetMS = -500
    bufferMS = 1000
    resampledRate = 200
    filtFreq = [58.0,62.0]
    # recalled events
    rRes = wavelet.tfPhasePow(freqs,
			      ev[rInd].getDataMS(chan,
						 durationMS,
						 offsetMS,
						 bufferMS,
						 resampledRate,
						 filtFreq=filtFreq,
						 keepBuffer=True),
						 verbose=True)
    # not recalled events
    nRes = wavelet.tfPhasePow(freqs,
			      ev[nInd].getDataMS(chan,
						 durationMS,
						 offsetMS,
						 bufferMS,
						 resampledRate,
						 filtFreq=filtFreq,
						 keepBuffer=True),
						 verbose=True)
    
    # get mean power across events (axis=1)
    rPow = N.squeeze(N.mean(N.log10(rRes.power),1))
    nPow = N.squeeze(N.mean(N.log10(nRes.power),1))

    print "Generating plots..."
    fig = 0

    # power spectrum
    fig+=1
    pylab.figure(fig)
    pylab.plot(rRes.freqs,N.squeeze(N.mean(rPow,1)),'r')
    pylab.plot(nRes.freqs,N.squeeze(N.mean(nPow,1)),'b')
    pylab.legend(('Recalled','Not Recalled'))
    pylab.xlabel('Frequency (Hz)')
    pylab.ylabel(r'Power ($log_{10}(mV^2)$)')

    # plot the diff in mean power
    fig+=1
    pylab.figure(fig)
    pylab.contourf(rRes.time,rRes.freqs,rPow-nPow)
    pylab.colorbar()
    pylab.xlabel('Time (ms)')
    pylab.ylabel('Frequency (Hz)')
    pylab.title('SME (diff in power) for channel %d' % (chan))

    # show the plots
    pylab.show()



# run the test case
testcase()

