from numpy import *
from pylab import *
from matplotlib.patches import Circle
import csv
from helper import pol2cart
from griddata import griddata

def topoplot():
    """Plot a topographic map of the scalp in a 2-D circular view (looking down at the top of the head).
    Nose is at top of plot; left is left; right is right. More to come ...."""
    
    ########################################################
    # Set up head:
    #
    # Basics:
    # Coordinates of middle of head:
    headCenter = (0,0)
    #
    # Angle of direction where the nose points in degrees:
    # 0 deg is top, 90 deg is right, etc.
    noseDeg = 0 # not yet used!
    #
    # Outline:
    headRad = 0.5 # radius of head
    headLineWidth = 3
    head = Circle(headCenter,headRad,fill=False,linewidth=headLineWidth)
    #
    # Nose:    
    noseLineWidth = earLineWidth = 2
    noseWidth = 0.18*headRad
    # Distance from the center of the head to the point where the nose touches the outline of the head: 
    noseDist = math.cos(math.asin((noseWidth/2)/headRad))*headRad
    # Distance from the center of the head to the tip of the nose:
    noseTipDist = 1.15*headRad
    nose = Line2D([-noseWidth/2+headCenter[0],0+headCenter[0],noseWidth/2+headCenter[0]],[noseDist+headCenter[1],noseTipDist+headCenter[1],noseDist+headCenter[1]],color='black',linewidth=noseLineWidth,solid_joinstyle='round',solid_capstyle='round')
    #
    # Ears:
    q = .04 # ear lengthening
    earX = array([.497-.005,.510,.518,.5299,.5419,.54,.547,.532,.510,.489-.005])*(headRad/0.5)#+headCenter[0]
    earY = array([q+.0555,q+.0775,q+.0783,q+.0746,q+.0555,-.0055,-.0932,-.1313,-.1384,-.1199])*(headRad/0.5)#+headCenter[1]
    earRight = Line2D(earX+headCenter[0],earY+headCenter[1],color='black',linewidth=earLineWidth,solid_joinstyle='round',solid_capstyle='round')
    earLeft = Line2D(headCenter[0]-earX,earY+headCenter[1],color='black',linewidth=earLineWidth,solid_joinstyle='round',solid_capstyle='round')
    #
    ########################################################
    
    x,y = getElectrodeCoords(headRad)
    
    #xi = linspace(min(x),max(x),10)
    #yi = linspace(min(y),max(y),10)
    
    
    #x = random.uniform(-2,2,100);  y = random.uniform(-2,2,100)
    #z = x*exp(-x**2-y**2)
    z = rand(129)
    
    # x, y, and z are now vectors containing nonuniformly sampled data.
    # Define a regular grid and grid data to it.
    nx = 51; ny = 41
    xi, yi = meshgrid(linspace(-2,2,nx),linspace(-2,2,ny))
    # masked=True mean no extrapolation, output is masked array.
    zi = griddata(x,y,z,xi,yi,masked=True)
    
    #nx = 50; ny = 50
    #xi, yi = meshgrid(linspace(min(x),max(x),nx),linspace(min(y),max(y),ny))
    
    #Xi,Yi,Zi =
    #zi = griddata(y,x,rand(129),yi,xi,masked=True)
    #CS = contour(xi,yi,zi,15,linewidths=0.5,colors=['k'])
    CS = contourf(xi,yi,zi,15,cmap=cm.jet)
    #test=imshow(zi)
    
    print 'min/max = ',min(zi.compressed()),max(zi.compressed())

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
    
    
    
    
    plot(x,y,'bo')
    
    #a=subplot(1,1,1, aspect='equal')
    xlim(-(headRad*2)+headCenter[0], (headRad*2)+headCenter[0])
    ylim(-(headRad*2)+headCenter[1], (headRad*2)+headCenter[1])
    #a.add_artist(head)
    #a.add_artist(nose)
    #a.add_artist(earRight)
    #a.add_artist(earLeft)
    #plot(x,y,'bo')
    #a.add_artist(CS)
    show()


def getElectrodeCoords(headRad):
    # read in testLocs.dat that was generated in Matlab as follows:
    # locs_orig=readlocs('GSN129.sfp');
    # locs=locs_orig(4:end); %ignore orig locations 1-3, these are frontal ones we dont have
    # tmp = [locs.theta; locs.radius];
    # save testLocs.dat tmp -ascii
    locs = csv.reader(open("testLocs.dat", "r"), delimiter=' ',skipinitialspace=True)
    theta=locs.next()
    radius=locs.next()
    theta = array([(double(i)+90) for i in theta])
    radius = array([double(i)*(headRad/0.5) for i in radius])
    x,y=pol2cart(theta,radius,radians=False)
    return x,y


    

