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

default_head_props = {'head_linewidth': 3,
                      'head_linecolor': 'black',
                      'nose_linewidth': 2,
                      'ear_linewidth': 2,
                     }
default_label_props = {'ha': 'center',
                       'va': 'center'}
default_sensor_props = {'marker': 'o',
                        'c': 'k', 
                        's': 8}
default_contour_props = {'linewidths': 0,
                         'linestyle': '-',
                         'colors': 'black',}

def topoplot(values=None, labels=None, sensors=None, axes=None, 
             center=(0,0), nose_dir=0., radius=0.5,
             head_props=None, sensor_props=None,
             label_props=None, 
             contours=15, contour_props=None,
             resolution=400, axis_props='off', 
             plot_mask='circular', plot_radius_buffer=.2,
             **kwargs):
    """
    Plot a topographic map of the scalp in a 2-D circular view
    (looking down at the top of the head).

    Parameters
    ----------
    values : {None, array-like}, optional
        Values to plot. There must be one value for each electrode.
    labels : {None, array-like}, optional
        Electrode labels/names to plot. There must be one for each electrode.
    sensors : {None, tuple of floats}, optional
        Polar coordinates of the sensor locations. If not None,
        sensors[0] specifies the angle (in degrees) and sensors[1]
        specifies the radius.
    axes : {matplotlib.axes}, optional
        Axes to which the topoplot should be added.
    center : {tuple of floats}, optional
        x and y coordinates of the center of the head.
    nose_dir : {float}, optional
        Angle (in degrees) where the nose is pointing. 0 is
        up, 90 is left, 180 is down, 270 is right, etc.
    radius : {float}, optional
        Radius of the head.
    head_props : dict
        Dictionary of head properties. See default_head_props for choices.
    sensor_props : dict
        Dictionary of sensor properties. See options for scatter in mpl and
        default_sensor_props.
    label_props : dict
        Dictionary of sensor label properties. See options for text in mpl
        and default_label_props.
    contours : {int}, optional
        Number of contours.
    contour_props : dict
        Dictionary of contour properties. See options for contour in mpl and
        default_contour_props.
    resolution : {int}, optional
        Resolution of the interpolated grid. Higher numbers give
        smoother edges of the plot, but increase memory and
        computational demands.
    axis_props : {str}, optional
        Axis properties.
    plot_mask : {str}, optional
        The mask around the plotted values. 'linear' conects the outer
        electrodes with straight lines, 'circular' draws a circle
        around the outer electrodes (see plot_radius_buffer).
    plot_radius_buffer : float, optional
        Buffer outside the electrode circumference for generating
        interpolated values with a circular mask. 
        This should be greater than zero to aviod interpolation errors.
    **kwargs : optional
        Optional keyword arguments to be passed on to contourf.
    """

    if axes is not None: # axes are given
        a=axes
    else: # a new subplot is created
        a=plt.subplot(1, 1, 1, aspect='equal')

    a.axis(axis_props)
    
    if True: # head should be plotted
        # deal with the head props
        hprops = default_head_props.copy()
        if not head_props is None:
            hprops.update(head_props)

        # Set up head
        head = plt.Circle(center, radius, fill=False, 
                          linewidth=hprops['head_linewidth'],
                          edgecolor=hprops['head_linecolor'],
                          axes=a)

        # Nose:
        nose_width = 0.18*radius
        # Distance from the center of the head to the point where the
        # nose touches the outline of the head:
        nose_dist = np.cos(np.arcsin((nose_width/2.)/radius))*radius
        # Distance from the center of the head to the tip of the nose:
        nose_tip_dist = 1.15*radius
        # Convert to polar coordinates for rotating:
        nose_polar_angle, nose_polar_radius = cart2pol(
            np.array([-nose_width/2, 0, nose_width/2]),
            np.array([nose_dist, nose_tip_dist, nose_dist]))
        nose_polar_angle = nose_polar_angle + deg2rad(nose_dir)
        # And back to cartesian coordinates for plotting:
        nose_x, nose_y = pol2cart(nose_polar_angle, nose_polar_radius)
        # Move nose with head:
        nose_x = nose_x + center[0]
        nose_y = nose_y + center[1]
        nose = plt.Line2D(nose_x, nose_y,
                          solid_joinstyle='round', solid_capstyle='round',
                          color=hprops['head_linecolor'],
                          linewidth=hprops['nose_linewidth'],
                          axes=a)

        # Ears:
        q = .04 # ear lengthening
        ear_x = np.array(
            [.497-.005, .510,.518, .5299, .5419, .54, .547,
             .532, .510, .489-.005])*(radius/0.5)
        ear_y = np.array(
            [q+.0555, q+.0775, q+.0783, q+.0746, q+.0555,
             -.0055, -.0932, -.1313, -.1384, -.1199])*(radius/0.5)
        # Convert to polar coordinates for rotating:
        rightear_polar_angle, rightear_polar_radius = cart2pol(ear_x, ear_y)
        leftear_polar_angle, leftear_polar_radius = cart2pol(-ear_x, ear_y)
        rightear_polar_angle = rightear_polar_angle+deg2rad(nose_dir)
        leftear_polar_angle = leftear_polar_angle+deg2rad(nose_dir)
        # And back to cartesian coordinates for plotting:
        rightear_x, rightear_y = pol2cart(rightear_polar_angle,
                                         rightear_polar_radius)
        leftear_x, leftear_y = pol2cart(leftear_polar_angle,
                                        leftear_polar_radius)
        
        # Move ears with head:
        rightear_x = rightear_x + center[0]
        rightear_y = rightear_y + center[1]
        leftear_x = leftear_x + center[0]
        leftear_y = leftear_y + center[1]
        
        ear_right = plt.Line2D(rightear_x, rightear_y,
                               color=hprops['head_linecolor'],
                               linewidth=hprops['ear_linewidth'],
                               solid_joinstyle='round',
                               solid_capstyle='round',
                               axes=a)
        ear_left = plt.Line2D(leftear_x, leftear_y,
                              color=hprops['head_linecolor'],
                              linewidth=hprops['ear_linewidth'],
                              solid_joinstyle='round',
                              solid_capstyle='round',
                              axes=a)       
        a.add_artist(head)
        a.add_artist(nose)
        a.add_artist(ear_right)
        a.add_artist(ear_left)

    if sensors is None:
        if axes is None:
            a.set_xlim(-radius*1.2+center[0], radius*1.2+center[0])
            a.set_ylim(-radius*1.2+center[1], radius*1.2+center[1]) 
        return("No sensor locations specified!")
    
    # Convert & rotate sensor locations:
    angles = -sensors[0]+90
    angles = angles+nose_dir
    angles = deg2rad(angles)
    radii = sensors[1]
    # expand or shrink electrode locations with radius of head:
    radii = radii*(radius/0.5)
    # plotting radius is determined by largest sensor radius:
    plot_radius = max(radii)*(1.0+plot_radius_buffer)
    
    # convert electrode locations to cartesian coordinates for plotting:
    x, y = pol2cart(angles, radii)
    x = x + center[0]
    y = y + center[1]

    if True: # plot electrodes
        sprops = default_sensor_props.copy()
        if not sensor_props is None:
            sprops.update(sensor_props)

        #a.plot(x,y,markerfacecolor=colors[1],marker='o',linestyle='')
        a.scatter(x, y, zorder=10, **sprops)

    if not labels is None:
        lprops = default_label_props.copy()
        if not label_props is None:
            lprops.update(label_props)

        for i in range(len(labels)):
            a.text(x[i],y[i],labels[i],**lprops)

        
    if values is None:
        return #('No values to plot specified!')
    if np.size(values) != np.size(sensors,1):
        return('Numer of values to plot is different from number of sensors!'+
               '\nNo values have been plotted!')

    # set the values
    z = values

    # resolution determines the number of interpolated points per unit
    nx = round(resolution*plot_radius)
    ny = round(resolution*plot_radius)

    # now set up the grid:
    xi, yi = np.meshgrid(np.linspace(-plot_radius, plot_radius,nx),
                         np.linspace(-plot_radius, plot_radius,ny))
    # and move the center to coincide with the center of the head:
    xi = xi + center[0]
    yi = yi + center[1]
    # interploate points:
    if plot_mask=='linear':
        # masked = True means that no extrapolation outside the
        # electrode boundaries is made this effectively creates a mask
        # with a linear boundary (connecting the outer electrode
        # locations)
        #zi = griddata(x,y,z,xi,yi,masked=True)
        #zi = griddata(x,y,z,xi,yi)
        pass
    elif plot_mask=='circular':
        npts = np.mean((nx,ny))*2
        t = np.linspace(0, 2*np.pi,npts)[:-1]
        x = np.r_[x, np.cos(t)*plot_radius]
        y = np.r_[y, np.sin(t)*plot_radius]
        z = np.r_[z, np.zeros(len(t))]
    else:
        # we need a custom mask:
        #zi = griddata(x,y,z,xi,yi,ext=1,masked=False)
        #zi = griddata(x,y,z,xi,yi)
        # zi = griddata((x,y),z,(xi,yi),method='cubic')
        # if plot_mask=='circular':
        #     # the interpolated array doesn't know about its position
        #     # in space and hence we need to subtract head center from
        #     # xi & xi to calculate the mask
        #     mask = (np.sqrt(np.power(xi-center[0],2) +
        #                     np.power(yi-center[1],2)) > plot_radius)
        #     zi[mask] = 0
        #     zi[np.isnan(zi)] = 0.0
        #     zi[mask] = np.nan
        # other masks may be added here and can be defined as shown
        # for the circular mask. All other plot_mask values result in
        # no mask which results in showing interpolated values for the
        # square surrounding the head.
        pass

    # calc the grid
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # # If no colormap is specified, use default colormap:
    # if cmap is None:
    #     cmap = plt.get_cmap()
        
    # make contours
    cprops = default_contour_props.copy()
    if not contour_props is None:
        cprops.update(contour_props)
 
    if np.any(cprops['linewidths'] > 0):
        a.contour(xi, yi, zi, contours, **cprops)

    # make countour color patches:
    # a.contourf(xi, yi, zi, contours, cmap=cmap, extend='both')
    a.contourf(xi, yi, zi, contours, extend='both', **kwargs)

