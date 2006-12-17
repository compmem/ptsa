from numpy import *
from pylab import *
from matplotlib.patches import Circle

def topoplot():
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
    nose = Line2D([-noseWidth/2,0,noseWidth/2],[noseDist,noseTipDist,noseDist],color='black',linewidth=noseLineWidth,solid_joinstyle='round',solid_capstyle='round')
    
    q = .04 # ear lengthening
    earX = array([.497-.005,.510,.518,.5299,.5419,.54,.547,.532,.510,.489-.005])
    earY = array([q+.0555,q+.0775,q+.0783,q+.0746,q+.0555,-.0055,-.0932,-.1313,-.1384,-.1199])
    earRight = Line2D(earX,earY,color='black',linewidth=earLineWidth,solid_joinstyle='round',solid_capstyle='round')
    earLeft = Line2D((earX*-1),earY,color='black',linewidth=earLineWidth,solid_joinstyle='round',solid_capstyle='round')
    
    
    a=subplot(1,1,1, aspect='equal')
    xlim(-1, 1)
    ylim(-1, 1)
    a.add_artist(head)
    a.add_artist(nose)
    a.add_artist(earRight)
    a.add_artist(earLeft)
    show()





















