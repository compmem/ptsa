from pylab import load, rand, figure, xlim, ylim, show
from ptsa.plotting.topoplot import topoplot

def getElecs():
    # read in testLocs.dat that was generated in Matlab as follows:
    # locs_orig=readlocs('GSN129.sfp');
    # locs=locs_orig(4:end); %ignore orig locations 1-3, these are frontal ones we dont have
    # tmp = [locs.theta; locs.radius];
    # save testLocs.dat tmp -ascii
    locs=load("testLocs.dat")
    theta=-locs[0]+90
    
    #theta=deg2rad(theta)
    radius=locs[1]#*(headRad/0.5)
    #x,y=pol2cart(theta,radius,radians=False)
    return theta,radius #x,y

def getPowerVals():
    # read in toPlotDiff
    toPlot = load("toPlotDiff.dat")
    return toPlot

els = getElecs()
toPlot=rand(129)#getPowerVals()

fig=1
figure(fig)
topoplot()
#show()

fig+=1
figure(fig)
topoplot(elecs=els)
#show()

fig+=1
figure(fig)
topoplot(elecs=els,plotHead=False)
#show()

fig+=1
figure(fig)
topoplot(elecs=els,valsToPlot=toPlot,elecsCol=None)

fig+=1
figure(fig)
topoplot(elecs=els,valsToPlot=toPlot,contWidth=0)

fig+=1
figure(fig)
topoplot(elecs=els,valsToPlot=toPlot,elecsCol=None,plotHead=False)


fig+=1
figure(fig)
topoplot(headCenter=(0,0),elecs=els,valsToPlot=toPlot,plotMask='linear')
topoplot(headCenter=(2,0),elecs=els,valsToPlot=toPlot,plotMask='circular')
topoplot(headCenter=(4,0),elecs=els,valsToPlot=toPlot,plotMask='square')
xlim(-1,5)

grid=100
fig+=1
figure(fig)
topoplot(headCenter=(0,0),headRad=0.2,elecs=els,valsToPlot=toPlot,gridRes=grid)
topoplot(headCenter=(1.5,0),headRad=0.5,elecs=els,valsToPlot=toPlot,gridRes=grid)
topoplot(headCenter=(4,0),headRad=1,elecs=els,valsToPlot=toPlot,gridRes=grid)
topoplot(headCenter=(8.5,0),headRad=2,elecs=els,valsToPlot=toPlot,gridRes=grid)
xlim(-0.5,12)
#axis('on')
#show()

fig+=1
figure(fig)
topoplot(headCenter=(0,0),noseDir=-45,elecs=els,valsToPlot=toPlot,elecsCol=None,gridRes=grid)
topoplot(headCenter=(0,2),noseDir=-135,elecs=els,valsToPlot=toPlot,elecsCol=None,gridRes=grid)
topoplot(headCenter=(2,0),noseDir=135,elecs=els,valsToPlot=toPlot,elecsCol=None,gridRes=grid)
topoplot(headCenter=(2,2),noseDir=45,elecs=els,valsToPlot=toPlot,elecsCol=None,gridRes=grid)
xlim(-1,3)
ylim(-1,3)
#show()

#topoplot


show()
