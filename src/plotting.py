#from numpy import *
from pylab import *
#from matplotlib.patches import Circle
from helper import pol2cart,cart2pol,deg2rad
from griddata import griddata

def topoplot(headCenter=(0,0),noseDir=0.,noseDirRadians=False,headRad=0.5,plotHead=True,elecs=(0,0),headCol='black',headLineWidth=3,noseLineWidth=2,earLineWidth=2,contCols='black',gridRes=250,colmap=None,elecsCol='black',numConts=15,contWidth=0.5,contStyle='-'):
    """Plot a topographic map of the scalp in a 2-D circular view (looking down at the top of the head).
    Nose is at top of plot; left is left; right is right. More to come ...."""
    
    if colmap is None: colmap = get_cmap()
    
    if plotHead:
        # Set up head
        head = Circle(headCenter,headRad,fill=False,linewidth=headLineWidth,edgecolor=headCol)
        # Nose:
        noseWidth = 0.18*headRad
        # Distance from the center of the head to the point where the nose touches the outline of the head: 
        noseDist = math.cos(math.asin((noseWidth/2)/headRad))*headRad
        # Distance from the center of the head to the tip of the nose:
        noseTipDist = 1.15*headRad
        nosePolarTheta,nosePolarRadius=cart2pol(array([-noseWidth/2+headCenter[0],headCenter[0],noseWidth/2+headCenter[0]]),array([noseDist+headCenter[1],noseTipDist+headCenter[1],noseDist+headCenter[1]]))
        if noseDirRadians:
            nosePolarTheta=nosePolarTheta+noseDir
        else:
            nosePolarTheta=nosePolarTheta+deg2rad(noseDir)
        noseX,noseY=pol2cart(nosePolarTheta,nosePolarRadius)
        nose = Line2D(noseX,noseY,color=headCol,linewidth=noseLineWidth,solid_joinstyle='round',solid_capstyle='round')
        # Ears:
        q = .04 # ear lengthening
        earX = array([.497-.005,.510,.518,.5299,.5419,.54,.547,.532,.510,.489-.005])*(headRad/0.5)#+headCenter[0]
        earY = array([q+.0555,q+.0775,q+.0783,q+.0746,q+.0555,-.0055,-.0932,-.1313,-.1384,-.1199])*(headRad/0.5)#+headCenter[1]
        earPolarThetaRight,earPolarRadiusRight=cart2pol(earX,earY)
        earPolarThetaLeft,earPolarRadiusLeft=cart2pol(-earX,earY)
        if noseDirRadians:
            earPolarThetaRight=earPolarThetaRight+noseDir
            earPolarThetaLeft=earPolarThetaLeft+noseDir
        else:
            earPolarThetaRight=earPolarThetaRight+deg2rad(noseDir)
            earPolarThetaLeft=earPolarThetaLeft+deg2rad(noseDir)
        earXRight,earYRight=pol2cart(earPolarThetaRight,earPolarRadiusRight)
        earXLeft,earYLeft=pol2cart(earPolarThetaLeft,earPolarRadiusLeft)
        earRight = Line2D(earXRight+headCenter[0],earYRight+headCenter[1],color=headCol,linewidth=earLineWidth,solid_joinstyle='round',solid_capstyle='round')
        earLeft = Line2D(earXLeft+headCenter[0],earYLeft+headCenter[1],color=headCol,linewidth=earLineWidth,solid_joinstyle='round',solid_capstyle='round')
    
    # Set up Electrodes
    #x,y = getElectrodeCoords()
    #print size(elecs)
    #print elecs
    if size(elecs) < 4 or len(elecs) !=2: return
    
    theta,radius = cart2pol(elecs[0],elecs[1])
    radius = radius*(headRad/0.5)
    theta = theta + deg2rad(noseDir)
    # if plotRad is None:
    plotRad = max(radius)
    # print plotRad
    # plotRad = max(plotRad,headRad)
    
    x,y = pol2cart(theta,radius)
    z = getPowerVals()
    nx = round(gridRes*plotRad*2)
    ny = round(gridRes*plotRad*2)
    xi, yi = meshgrid(linspace(-plotRad,plotRad,nx),linspace(-plotRad,plotRad,ny))
    zi = griddata(x,y,z,xi,yi,masked=True)
    
    mask = (sqrt(pow(xi,2) + pow(yi,2)) > headRad*1.5) # mask outside the plotting circle
    #ii = find(not mask)
    zi[mask] = 0 # mask non-plotting voxels with NaNs
    #grid = plotrad;                       % unless 'noplot', then 3rd output arg is plotrad
    
    #zi[60:,60:]=0

    #nx = 50; ny = 50
    #xi, yi = meshgrid(linspace(min(x),max(x),nx),linspace(min(y),max(y),ny))
    a=subplot(1,1,1, aspect='equal')
    #Xi,Yi,Zi =
    #zi = griddata(y,x,rand(129),yi,xi,masked=True)
    #CS = contour(xi,yi,zi,15,linewidths=0.5,colors=['gray'])#['k'])
    #CS = contourf(xi,yi,zi,15,cmap=colmap)
    CS = contour(xi,yi,zi,numConts,linewidths=contWidth,linestyle=contStyle,colors=contCols)#['k'])
    CS = contourf(xi,yi,zi,numConts,cmap=colmap)
    #test=imshow(zi)
    #cm.autoscale()

    #imshow((zi),interpolation="nearest")
    
    #print shape(xi)
    
    #print 'min/max = ',min(zi.compressed()),max(zi.compressed())
    
    #ax = axes([0.1,0.1,0.75,0.75])
    # Contour the gridded data, plotting dots at the nonuniform data points.
    # CS = p.contour(xi,yi,zi,15,linewidths=0.5,colors=['k'])
    # CS = p.contourf(xi,yi,zi,15,cmap=p.cm.jet)
    # cax = p.axes([0.875, 0.1, 0.05, 0.75]) # setup colorbar axes
    # p.colorbar(tickfmt='%4.2f', cax=cax) # draw colorbar
    # p.axes(ax)  # make the original axes current again
    # p.scatter(x,y,marker='o',c='b',s=5)
    # p.xlim(-1.9,1.9)
    # p.ylim(-1.9,1.9)
    # p.title('griddata test (%d points)' % npts)
    # p.savefig('griddata.png')
    # p.show()
    
    
    
    #X, Y = meshgrid(x,y)
    #Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    #Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    # difference of Gaussians
    #Z = random.rand(129,129) #10.0 * (Z2 - Z1)
    
    
    
    # Create a simple contour plot with labels using default colors.  The
    # inline argument to clabel will control whether the labels are draw
    # over the line segments of the contour, removing the lines beneath
    # the label
    #figure()
    #im = imshow(Z, interpolation='bilinear', origin='lower',
    #        cmap=cm.gray)#, extent=(-1,3,-2,2))
    
    
    
    if elecsCol is not None:
        plot(x,y,markerfacecolor=elecsCol,marker='o',linestyle=None)
    
    #a=subplot(1,1,1, aspect='equal')
    xlim(headCenter[0]-plotRad,headCenter[0]+plotRad)
    ylim(headCenter[1]-plotRad,headCenter[1]+plotRad)
    a.add_artist(head)
    a.add_artist(nose)
    a.add_artist(earRight)
    a.add_artist(earLeft)
    #plot(x,y,'bo')
    #a.add_artist(CS)
    show()
    #figure()


def getElectrodeCoords():
    # read in testLocs.dat that was generated in Matlab as follows:
    # locs_orig=readlocs('GSN129.sfp');
    # locs=locs_orig(4:end); %ignore orig locations 1-3, these are frontal ones we dont have
    # tmp = [locs.theta; locs.radius];
    # save testLocs.dat tmp -ascii
    locs=load("testLocs.dat")
    theta=locs[0]+90
    radius=locs[1]#*(headRad/0.5)
    x,y=pol2cart(theta,radius,radians=False)
    return x,y

def getPowerVals():
    # read in toPlotDiff
    toPlot = load("toPlotDiff.dat")
    return toPlot

  

