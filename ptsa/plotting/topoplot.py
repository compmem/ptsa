#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
import matplotlib.pyplot as plt

from ptsa.helper import pol2cart, cart2pol, deg2rad
from scipy.interpolate import griddata

def topoplot(values=None, axes=None, center=(0,0), nose_dir=0., radius=0.5,
             sensors=None, colors=('black','black','black'),
             linewidths=(3,2,2,0.5), contours_ls='-', contours=15,
             resolution=400,method='cubic',fill_value=np.nan,
             axis_props='off',**kwargs):
    """
    Plot a topographic map of the scalp in a 2-D circular view
    (looking down at the top of the head).

    Parameters
    ----------
    values : {None, array-like}, optional
        Values to plot. There must be one value for each electrode.
    axes : {matplotlib.axes}, optional
        Axes to which the topoplot should be added.
    center : {tuple of floats}, optional
        x and y coordinates of the center of the head.
    nose_dir : {float}, optional
        Angle (in degrees) where the nose is pointing. 0 is
        up, 90 is left, 180 is down, 270 is right, etc.
    radius : {float}, optional
        Radius of the head.    
    sensors : {None, tuple of floats}, optional
        Polar coordinates of the sensor locations. If not None,
        sensors[0] specifies the angle (in degrees) and sensors[1]
        specifies the radius.
    colors : {tuple of str or None}, optional
        Colors for the outline of the head, sensor markers, and contours
        respectively. If any is None, the corresponding feature is
        not plotted. For contours either a single color or
        multiple colors can be specified.
    linewidths : {tuple of floats}, optional
        Line widths for the head, nose, ears, and contours
        respectively. For contours either a single linewith or
        multiple linewidths can be specified.
    contours_ls : {str}, optional
        Line style of the contours.
    contours : {int}, optional
        Number of countours.
    resolution : {int}, optional
        Resolution of the interpolated grid. Higher numbers give
        smoother edges of the plot, but increase memory and
        computational demands.
    method : {'cubic','linear','nearest'}, optional
        Method of interpolation. See scipy.interpolate.griddata for
        details.
    fill_value : float, optional
        Value used to fill in for requested points outside of the
        convex hull of the input points. See
        scipy.interpolate.griddata for details.
    axis_props : {str}, optional
        Axis properties.
    **kwargs : **kwargs, optional
        Key word arguments to be passed on to
        matplotlib.pyplot.contour and matplotlib.pyplot.contourf.
    """

    # If no colormap is specified, use default colormap:
    # if cmap is None:
    #    cmap = plt.get_cmap()

    if axes is not None: # axes are given
        a=axes
    else: # a new subplot is created
        a=plt.subplot(1,1,1, aspect='equal')

    plt.axis(axis_props)
    
    if colors[0]: # head should be plotted
        # Set up head
        head = plt.Circle(center, radius,fill=False, linewidth=linewidths[0],
                          edgecolor=colors[0])

        # Nose:
        nose_width = 0.18*radius
        # Distance from the center of the head to the point where the
        # nose touches the outline of the head:
        nose_dist = np.cos(np.arcsin((nose_width/2.)/radius))*radius
        # Distance from the center of the head to the tip of the nose:
        nose_tip_dist = 1.15*radius
        # Convert to polar coordinates for rotating:
        nose_polar_angle,nose_polar_radius = cart2pol(
            np.array([-nose_width/2,0,nose_width/2]),
            np.array([nose_dist,nose_tip_dist,nose_dist]))
        nose_polar_angle = nose_polar_angle+deg2rad(nose_dir)
        # And back to cartesian coordinates for plotting:
        nose_x,nose_y = pol2cart(nose_polar_angle,nose_polar_radius)
        # Move nose with head:
        nose_x = nose_x + center[0]
        nose_y = nose_y + center[1]
        nose = plt.Line2D(nose_x,nose_y,color=colors[0],linewidth=linewidths[1],
                          solid_joinstyle='round',solid_capstyle='round')

        # Ears:
        q = .04 # ear lengthening
        ear_x = np.array([.497-.005,.510,.518,.5299,
                          .5419,.54,.547,.532,.510,.489-.005])*(radius/0.5)
        ear_y = np.array([q+.0555,q+.0775,q+.0783,q+.0746,q+.0555,
                          -.0055,-.0932,-.1313,-.1384,-.1199])*(radius/0.5)
        # Convert to polar coordinates for rotating:
        rightear_polar_angle,rightear_polar_radius = cart2pol(ear_x,ear_y)
        leftear_polar_angle,leftear_polar_radius = cart2pol(-ear_x,ear_y)
        rightear_polar_angle=rightear_polar_angle+deg2rad(nose_dir)
        leftear_polar_angle=leftear_polar_angle+deg2rad(nose_dir)
        # And back to cartesian coordinates for plotting:
        rightear_x,rightear_y=pol2cart(rightear_polar_angle,
                                       rightear_polar_radius)
        leftear_x,leftear_y=pol2cart(leftear_polar_angle,leftear_polar_radius)
        
        # Move ears with head:
        rightear_x = rightear_x + center[0]
        rightear_y = rightear_y + center[1]
        leftear_x = leftear_x + center[0]
        leftear_y = leftear_y + center[1]
        
        ear_right = plt.Line2D(rightear_x,rightear_y,color=colors[0],
                               linewidth=linewidths[3],solid_joinstyle='round',
                               solid_capstyle='round')
        ear_left = plt.Line2D(leftear_x,leftear_y,color=colors[0],
                             linewidth=linewidths[3],solid_joinstyle='round',
                             solid_capstyle='round')
        
        a.add_artist(head)
        a.add_artist(nose)
        a.add_artist(ear_right)
        a.add_artist(ear_left)

    if sensors is None:
        if axes is None:
            plt.xlim(-radius*1.2+center[0],radius*1.2+center[0])
            plt.ylim(-radius*1.2+center[1],radius*1.2+center[1]) 
        return("No sensor locations specified!")
    
    # Convert & rotate sensor locations:
    angles=sensors[0]
    angles=angles+nose_dir
    angles = deg2rad(angles)
    radii=sensors[1]
    # expand or shrink electrode locations with radius of head:
    radii = radii*(radius/0.5)
    # plotting radius is determined by largest sensor radius:
    plot_radius = max(radii)
    
    # convert electrode locations to cartesian coordinates for plotting:
    x,y = pol2cart(angles,radii)
    x = x + center[0]
    y = y + center[1]

    if colors[1]: # plot electrodes
        plt.plot(x,y,markerfacecolor=colors[1],marker='o',linestyle='')
        
    if values is None:
        return('No values to plot specified!')
    if np.size(values) != np.size(sensors,1):
        return('Numer of values to plot is different from number of sensors!'+
               '\nNo values have been plotted!')
    
    # resolution determines the number of interpolated points per unit
    nx = round(resolution*plot_radius)
    ny = round(resolution*plot_radius)
    # now set up the grid:
    xi, yi = np.meshgrid(np.linspace(-plot_radius,plot_radius,nx),
                         np.linspace(-plot_radius,plot_radius,ny))
    # and move the center to coincide with the center of the head:
    xi = xi + center[0]
    yi = yi + center[1]
    # interploate points:
    points = np.array([x,y]).T
    xyi = np.array([xi,yi]).T
    zi = griddata(points,values,xyi,method).T
    
    # make contour lines:
    plt.contour(xi,yi,zi,contours,linewidths=linewidths[3],
                linestyle=contours_ls,colors=colors[2],**kwargs)
    # make countour color patches:
    plt.contourf(xi,yi,zi,contours,**kwargs)

