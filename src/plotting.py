from pylab import *
from helper import pol2cart,cart2pol,deg2rad
from griddata import griddata

def topoplot(splot=None,headCenter=(0,0),noseDir=0.,noseDirRadians=False,headRad=0.5,plotHead=True,elecs=(0,0),elecsRadians=False,valsToPlot=None,headCol='black',headLineWidth=3,noseLineWidth=2,earLineWidth=2,contCols='black',gridRes=250,colmap=None,elecsCol='black',numConts=15,contWidth=0.5,contStyle='-',axisProp='off',plotMask='linear'):
    """Plot a topographic map of the scalp in a 2-D circular view (looking down at the top of the head).
    Nose is at top of plot; left is left; right is right. More to come ...."""

    # If no colormap is specified, use default colormap:
    if colmap is None: colmap = get_cmap()
    if splot is not None: # subplot to add the topoplot to is given
        a=splot
    else: # a new subplot is created
        a=subplot(1,1,1, aspect='equal')
    axis(axisProp)
    if plotHead:
        # Set up head
        head = Circle(headCenter,headRad,fill=False,linewidth=headLineWidth,edgecolor=headCol)
        # Nose:
        noseWidth = 0.18*headRad
        # Distance from the center of the head to the point where the nose touches the outline of the head: 
        noseDist = math.cos(math.asin((noseWidth/2)/headRad))*headRad
        # Distance from the center of the head to the tip of the nose:
        noseTipDist = 1.15*headRad
        # Convert to polar coordinates for rotating:
        nosePolarTheta,nosePolarRadius=cart2pol(array([-noseWidth/2,0,noseWidth/2]),array([noseDist,noseTipDist,noseDist]))
        if noseDirRadians:
            nosePolarTheta=nosePolarTheta+noseDir
        else:
            nosePolarTheta=nosePolarTheta+deg2rad(noseDir)
        # And back to cartesian coordinates for plotting:
        noseX,noseY=pol2cart(nosePolarTheta,nosePolarRadius)
        # Move nose with head:
        noseX = noseX + headCenter[0]
        noseY = noseY + headCenter[1]
        nose = Line2D(noseX,noseY,color=headCol,linewidth=noseLineWidth,solid_joinstyle='round',solid_capstyle='round')
        # Ears:
        q = .04 # ear lengthening
        earX = array([.497-.005,.510,.518,.5299,.5419,.54,.547,.532,.510,.489-.005])*(headRad/0.5)#+headCenter[0]
        earY = array([q+.0555,q+.0775,q+.0783,q+.0746,q+.0555,-.0055,-.0932,-.1313,-.1384,-.1199])*(headRad/0.5)#+headCenter[1]
        # Convert to polar coordinates for rotating:
        earPolarThetaRight,earPolarRadiusRight=cart2pol(earX,earY)
        earPolarThetaLeft,earPolarRadiusLeft=cart2pol(-earX,earY)
        if noseDirRadians:
            earPolarThetaRight=earPolarThetaRight+noseDir
            earPolarThetaLeft=earPolarThetaLeft+noseDir
        else:
            earPolarThetaRight=earPolarThetaRight+deg2rad(noseDir)
            earPolarThetaLeft=earPolarThetaLeft+deg2rad(noseDir)
        # And back to cartesian coordinates for plotting:
        earXRight,earYRight=pol2cart(earPolarThetaRight,earPolarRadiusRight)
        earXLeft,earYLeft=pol2cart(earPolarThetaLeft,earPolarRadiusLeft)
        
        # Move ears with head:
        earXRight = earXRight + headCenter[0]
        earYRight = earYRight + headCenter[1]
        
        earXLeft = earXLeft + headCenter[0]
        earYLeft = earYLeft + headCenter[1]
        
        earRight = Line2D(earXRight,earYRight,color=headCol,linewidth=earLineWidth,solid_joinstyle='round',solid_capstyle='round')
        earLeft = Line2D(earXLeft,earYLeft,color=headCol,linewidth=earLineWidth,solid_joinstyle='round',solid_capstyle='round')
        
        a.add_artist(head)
        a.add_artist(nose)
        a.add_artist(earRight)
        a.add_artist(earLeft)

    if size(elecs) < 4 or len(elecs) !=2:
        if splot is None:
            xlim(-headRad*1.2+headCenter[0],headRad*1.2+headCenter[0])
            ylim(-headRad*1.2+headCenter[1],headRad*1.2+headCenter[1]) 
        return("No electrode locations specified!")
    
    # Convert & rotate electrode locations:
    theta=elecs[0]
    if not elecsRadians:
        theta=theta+noseDir
        theta = deg2rad(theta)
    else:
        theta = theta + deg2rad(noseDir)
    radius=elecs[1]
    # expand or shrink electrode locations with radius of head:
    radius = radius*(headRad/0.5)
    # plotting radius is determined by largest electrode radius:
    plotRad = max(radius)
    
    # convert electrode locations to cartesian coordinates for plotting:
    x,y = pol2cart(theta,radius)
    x = x + headCenter[0]
    y = y + headCenter[1]

    if elecsCol is not None: # plot electrodes
        plot(x,y,markerfacecolor=elecsCol,marker='o',linestyle=None)

    if valsToPlot is None:
        return('No values to plot specified!')
    if size(valsToPlot) != size(elecs,1):
        return('Numer of values to plot is different from number of electrodes -- no values have been plotted!')
    
    z = valsToPlot
    
    # gridRes determines the number of interpolated points per unit
    # it is based on the standard head radius of 0.5
    # it scales with the actual head radius used
    gridRes=gridRes*(headRad/0.5)
    nx = round(gridRes*plotRad*2)
    ny = round(gridRes*plotRad*2)
    # now set up the grid:
    xi, yi = meshgrid(linspace(-plotRad,plotRad,nx),linspace(-plotRad,plotRad,ny))
    # and move the center to coincide with the center of the head:
    xi = xi + headCenter[0]
    yi = yi + headCenter[1]
    # interploate points:
    if plotMask=='linear':
        # masked = True means that no extrapolation outside the electrode boundaries is made
        # this effectively creates a mask with a linear boundary
        # (connecting the outer electrode locations)
        zi = griddata(x,y,z,xi,yi,masked=True)
    else:
        # we need a custom mask:
        zi = griddata(x,y,z,xi,yi,ext=1,masked=False)
        if plotMask=='circular':
            # the interpolated array doesn't know about its position in space
            # hence we need to subtract head center from xi & xi to calculate the mask
            mask = (sqrt(pow(xi-headCenter[0],2) + pow(yi-headCenter[1],2)) > plotRad)
            zi[mask] = 0
        # other masks may be added here and can be defined as shown for the circular mask
        # all other plotMask values result in no mask which results in showing interpolated
        # values for the square surrounding the head.
    
    # make contour lines:
    contour(xi,yi,zi,numConts,linewidths=contWidth,linestyle=contStyle,colors=contCols)#['k'])
    # make countour color patches:
    contourf(xi,yi,zi,numConts,cmap=colmap)
    

def showTopo(a=None,headCenter=(0,0),noseDir=0.,noseDirRadians=False,headRad=0.5,plotHead=True,elecs=(0,0),elecsRadians=False,valsToPlot=None,headCol='black',headLineWidth=3,noseLineWidth=2,earLineWidth=2,contCols='black',gridRes=250,colmap=None,elecsCol='black',numConts=15,contWidth=0.5,contStyle='-'):
    topoplot(a,headCenter,noseDir,noseDirRadians,headRad,plotHead,elecs,elecsRadians,valsToPlot,headCol,headLineWidth,noseLineWidth,earLineWidth,contCols,gridRes,colmap,elecsCol,numConts,contWidth,contStyle)
    show()


  

