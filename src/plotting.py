from numpy import *
from pylab import *
from matplotlib.patches import Circle
import csv


def topoplot():
    """Plot a topographic map of the scalp in a 2-D circular view (looking down at the top of the head).
    Nose is at top of plot; left is left; right is right. More to come ...."""
    
    headCenter = (0,0)
    headRad = 0.5 # radius of head
    headLineWidth = 3
    head = Circle(headCenter,headRad,fill=False,linewidth=headLineWidth)
    
    noseLineWidth = earLineWidth = 2
    noseWidth = 0.18*headRad
    # Distance from the center of the head to the point where the nose touches the outline of the head: 
    noseDist = math.cos(math.asin((noseWidth/2)/headRad))*headRad
    # Distance from the center of the head to the tip of the nose:
    noseTipDist = 1.15*headRad
    nose = Line2D([-noseWidth/2+headCenter[0],0+headCenter[0],noseWidth/2+headCenter[0]],[noseDist+headCenter[1],noseTipDist+headCenter[1],noseDist+headCenter[1]],color='black',linewidth=noseLineWidth,solid_joinstyle='round',solid_capstyle='round')
    
    q = .04 # ear lengthening
    earX = array([.497-.005,.510,.518,.5299,.5419,.54,.547,.532,.510,.489-.005])*(headRad/0.5)#+headCenter[0]
    earY = array([q+.0555,q+.0775,q+.0783,q+.0746,q+.0555,-.0055,-.0932,-.1313,-.1384,-.1199])*(headRad/0.5)#+headCenter[1]
    earRight = Line2D(earX+headCenter[0],earY+headCenter[1],color='black',linewidth=earLineWidth,solid_joinstyle='round',solid_capstyle='round')
    earLeft = Line2D(headCenter[0]-earX,earY+headCenter[1],color='black',linewidth=earLineWidth,solid_joinstyle='round',solid_capstyle='round')
    
    # read in testLocs.dat that was generated in Matlab as follows:
    # locs_orig=readlocs('GSN129.sfp');
    # locs=locs_orig(4:end); %ignore orig locations 1-3, these are frontal ones we dont have
    # tmp = [locs.theta; locs.radius];
    # save testLocs.dat tmp -ascii
    locs = csv.reader(open("testLocs.dat", "r"), delimiter=' ',skipinitialspace=True)
    theta=locs.next()
    radius=locs.next()
    
    # get sin and cos of theta-90 (because nose is on top and not right)
    # convert from deg to rad: 
    cosTheta = array([math.cos(((double(i)-90)/180)*math.pi) for i in theta])
    sinTheta = array([math.sin(((double(i)-90)/180)*math.pi) for i in theta])
    radius = array([double(i) for i in radius])
    
    # convert from polar to cartesian coords:
    x = radius*cosTheta
    y = radius*sinTheta
    
    if size(x) != size(y):
        import sys
        sys.exit("Sizes of x and y arrays differ!\nExiting ...")
    
    points=[()]*size(x)
    for i in xrange(size(x)):
        points[i]=Circle((headCenter[0]+x[i],headCenter[1]+y[i]),headRad/20,fill=True,linewidth=earLineWidth) 
    
    a=subplot(1,1,1, aspect='equal')
    xlim(-(headRad*2)+headCenter[0], (headRad*2)+headCenter[0])
    ylim(-(headRad*2)+headCenter[1], (headRad*2)+headCenter[1])
    a.add_artist(head)
    a.add_artist(nose)
    a.add_artist(earRight)
    a.add_artist(earLeft)
    for p in xrange(size(points)):
        a.add_artist(points[p])
    show()





