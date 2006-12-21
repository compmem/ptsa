from numpy import *
from pylab import *
from matplotlib.patches import Circle
import csv
from helper import pol2cart

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

    a=subplot(1,1,1, aspect='equal')
    xlim(-(headRad*2)+headCenter[0], (headRad*2)+headCenter[0])
    ylim(-(headRad*2)+headCenter[1], (headRad*2)+headCenter[1])
    a.add_artist(head)
    a.add_artist(nose)
    a.add_artist(earRight)
    a.add_artist(earLeft)
    plot(x,y,'bo')
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


    

