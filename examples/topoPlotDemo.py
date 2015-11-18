from pylab import loadtxt, rand, figure, xlim, ylim, show
from ptsa.plotting.topo import topoplot

def getElecs():
    # read in testLocs.dat that was generated in Matlab as follows:
    # locs_orig=readlocs('GSN129.sfp');
    # locs=locs_orig(4:end); %ignore orig locations 1-3, these are frontal ones we dont have
    # tmp = [locs.theta; locs.radius];
    # save testLocs.dat tmp -ascii
    locs=loadtxt("testLocs.dat")
    theta=-locs[0]+90
    
    #theta=deg2rad(theta)
    radius=locs[1]#*(headRad/0.5)
    #x,y=pol2cart(theta,radius,radians=False)
    return theta,radius #x,y

def getPowerVals():
    # read in toPlotDiff
    toPlot = loadtxt("toPlotDiff.dat")
    return toPlot

els = getElecs()
toPlot=rand(129)#getPowerVals()

fig=1
figure(fig)
topoplot()
#show()

fig+=1
figure(fig)
topoplot(sensors=els)
#show()

fig+=1
figure(fig)
topoplot(sensors=els,values=toPlot)

grid=100
fig+=1
figure(fig)
topoplot(center=(0,0),radius=0.2,sensors=els,values=toPlot,resolution=grid)
topoplot(center=(1.5,0),radius=0.5,sensors=els,values=toPlot,resolution=grid)
topoplot(center=(4,0),radius=1,sensors=els,values=toPlot,resolution=grid)
topoplot(center=(8.5,0),radius=2,sensors=els,values=toPlot,resolution=grid)
xlim(-0.5,12)
#axis('on')
#show()
#show()
