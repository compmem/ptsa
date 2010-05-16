from pylab import loadtxt, rand, figure, xlim, ylim, show
from ptsa.plotting.topoplot import topoplot

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
topoplot(sensors=els,colors=[None,'black','black'])
#show()

fig+=1
figure(fig)
topoplot(values=toPlot,sensors=els,colors=['black',None,'black'])

fig+=1
figure(fig)
topoplot(sensors=els,values=toPlot,colors=['black','black',None])

fig+=1
figure(fig)
topoplot(sensors=els,values=toPlot,colors=[None,None,'black'])


fig+=1
figure(fig)
topoplot(center=(0,0),sensors=els,values=toPlot,plot_mask='linear')
topoplot(center=(2,0),sensors=els,values=toPlot,plot_mask='circular')
topoplot(center=(4,0),sensors=els,values=toPlot,plot_mask='square')
xlim(-1,5)

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

fig+=1
figure(fig)
topoplot(center=(0,0),nose_dir=-45,sensors=els,values=toPlot,colors=['black',None,'black'],resolution=grid)
topoplot(center=(0,2),nose_dir=-135,sensors=els,values=toPlot,colors=['black',None,'black'],resolution=grid)
topoplot(center=(2,0),nose_dir=135,sensors=els,values=toPlot,colors=['black',None,'black'],resolution=grid)
topoplot(center=(2,2),nose_dir=45,sensors=els,values=toPlot,colors=['black',None,'black'],resolution=grid)
xlim(-1,3)
ylim(-1,3)
#show()

#topoplot


show()
