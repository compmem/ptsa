"""
Empirical Mode Decomposition
"""

import numpy as N
import scipy.interpolate
import scipy.signal

import pdb

def emd(data,numModes=10,numSifts=10):
    """Calculate the Emprical Mode Decomposition of a signal."""
    # initialize modes
    modes=[]
  
    # perform sifts until we have all modes
    residue=data
    while not doneSifting(residue):
        # perform a sift
        imf,residue = doSift(residue)
        
        # append the imf
        modes.append(imf)
  
    # append the residue
    modes.append(residue)

    # return an array of modes
    return N.asarray(modes)

def doneSifting(d):
    """We are done sifting is there a monotonic function."""
    return N.sum(localmax(d))+N.sum(localmax(-d))<=2

def doSift(data):
    """
    This function is modified to use the sifting-stopping criteria
    from Huang et al (2003) (this is the suggestion of Peel et al.,
    2005).  Briefly, we sift until the number of extrema and
    zerocrossings differ by at most one, then we continue sifting
    until the number of extrema and ZCs both remain constant for at
    least five sifts."""

    # save the data (may have to copy)
    imf=data

    # sift until num extrema and ZC differ by at most 1
    while True:
        imf=doOneSift(imf)
        numExtrema,numZC = analyzeIMF(imf)
        print 'numextrema=%d, numZC=%d' %  (numExtrema, numZC) 
        if abs(numExtrema-numZC)<=1:
            break

    # then continue until numExtrema and ZCs are constant for at least
    # 5 sifts (Huang et al., 2003)
    numConstant = 0
    desiredNumConstant = 5
    lastNumExtrema = numExtrema
    lastNumZC = numZC
    while numConstant < desiredNumConstant:
        imf=doOneSift(imf)
        numExtrema,numZC = analyzeIMF(imf)
        if numExtrema == lastNumExtrema and \
                numZC == lastNumZC:
            # is the same so increment
            numConstant+=1
        else:
            # different, so reset
            numConstant = 0
        # save the last extrema and ZC
        lastNumExtrema = numExtrema
        lastNumZC = numZC
        
    # FIX THIS
#     while True:
#         imf = doOneSift(imf)
#         numExtrema[end+1],numZC[end+1] = analyzeIMF(imf)
#         print 'FINAL STAGE: numextrema=%d, numZC=%d' % (numExtrema(end), numZC(end))
#         if length(numExtrema)>=numConstant & \
#                 all(numExtrema(end-4:end)==numExtrema(end)) & \
#                 all(numZC(end-4:end)==numZC(end)):
#             break

    # calc the residue
    residue=data-imf

    # return the imf and residue
    return imf,residue


def doOneSift(data):

    upper=getUpperSpline(data)
    lower=-getUpperSpline(-data)
    #upper=jinterp(find(maxes),data(maxes),xs);
    #lower=jinterp(find(mins),data(mins),xs);

    #imf=mean([upper;lower],1)
    imf = (upper+lower)*.5

    detail=data-imf

    # plot(xs,data,'b-',xs,upper,'r--',xs,lower,'r--',xs,imf,'k-')

    return detail # imf


def getUpperSpline(data):
    """Get the upper spline using the Mirroring algoirthm from Rilling et
al. (2003)."""

    maxInds = N.nonzero(localmax(data))[0]

    if len(maxInds) == 1:
        # Special case: if there is just one max, then entire spline
        # is that number
        #s=repmat(data(maxInds),size(data));
        s = N.ones(len(data))*data[maxInds]
        return s

    # Start points
    if maxInds[0]==0:
        # first point is a local max
        preTimes=1-maxInds[1]
        preData=data[maxInds[1]]
    else:
        # first point is NOT local max
        preTimes=1-maxInds[[1,0]]
        preData=data[maxInds[[1,0]]]

    # end points
    if maxInds[-1]==len(data)-1:
        # last point is a local max
        postTimes=2*len(data)-maxInds[-2]-1;
        postData=data[maxInds[-2]];
    else:
        # last point is NOT a local max
        postTimes=2*len(data)-maxInds[[-1,-2]];
        postData=data[maxInds[[-1,-2]]]

    # perform the spline fit
    t=N.r_[preTimes,maxInds,postTimes];
    d2=N.r_[preData, data[maxInds], postData];
    #s=interp1(t,d2,1:length(data),'spline');
    rep = scipy.interpolate.splrep(t,d2)
    s = scipy.interpolate.splev(range(len(data)),rep)
    # plot(1:length(data),data,'b-',1:length(data),s,'k-',t,d2,'r--');  

    return s


def analyzeIMF(d):
    numExtrema = N.sum(localmax(d))+N.sum(localmax(-d))
    numZC = N.sum(N.diff(N.sign(d))!=0)
    return numExtrema,numZC

# % if debug
# %   clf
# %   a1=subplot(2,1,1);
# %   plot(xs,d,'b-',xs,upper,'k-',xs,lower,'k-');
# %   axis tight;
  
# %   a2=subplot(2,1,2);
# %   plot(xs,stopScore,'b-',[0 length(d)],[thresh1 thresh1],'k--',[0 length(d)],[thresh2 ...
# %                       thresh2],'r--');
# %   axis tight;
# %   xlabel(sprintf('score = %.3g',s));  
# %   linkaxes([a1 a2],'x')
# %   keyboard
  
# % end



# function yi=jinterp(x,y,xi);
# if length(x)==1
#   yi=repmat(y,size(xi));
# else
#   yi=interp1(x,y,xi,'spline');
# end

  

def localmax(d):
    """Calculate the local maxima of a vector."""

    # this gets a value of -2 if it is an unambiguous local max
    # value -1 denotes that the run its a part of may contain a local max
    diffvec = N.r_[-N.inf,d,-N.inf]
    diffScore=N.diff(N.sign(N.diff(diffvec)))
                     
    # Run length code with help from:
    #  http://home.online.no/~pjacklam/matlab/doc/mtt/index.html
    # (this is all painfully complicated, but I did it in order to avoid loops...)

    # here calculate the position and length of each run
    runEndingPositions=N.r_[N.nonzero(d[0:-1]!=d[1:])[0],len(d)-1]
    runLengths = N.diff(N.r_[-1, runEndingPositions])
    runStarts=runEndingPositions-runLengths + 1

    # Now concentrate on only the runs with length>1
    realRunStarts = runStarts[runLengths>1]
    realRunStops = runEndingPositions[runLengths>1]
    realRunLengths = runLengths[runLengths>1]

    # save only the runs that are local maxima
    maxRuns=(diffScore[realRunStarts]==-1) & (diffScore[realRunStops]==-1)

    # If a run is a local max, then count the middle position (rounded) as the 'max'
    # CHECK THIS
    maxRunMiddles=N.round(realRunStarts[maxRuns]+realRunLengths[maxRuns]/2.)-1

    # get all the maxima
    maxima=(diffScore==-2)
    maxima[maxRunMiddles.astype(N.int32)] = True

    return maxima

#%make sure beginning & end are not local maxes
#%maxima([1 end])=false;


def calcIF(modes,samplerate):
    """
    Calculate the instantaneous frequency, amplitude, and phase of
    each mode.
    """

    amp=N.zeros(modes.shape,N.float32);
    phase=N.zeros(modes.shape,N.float32);

    for m in range(len(modes)):
        h=scipy.signal.hilbert(modes[m]);
        amp[m,:]=N.abs(h);
        phase[m,:]=N.angle(h);

    # calc the freqs
    f=N.diff(N.unwrap(phase[:,N.r_[0,0:len(modes[0])]]))/(2*N.pi)*samplerate

    # clip the freqs so they don't go below zero
    f = f.clip(0,f.max())

    return f,amp,phase



