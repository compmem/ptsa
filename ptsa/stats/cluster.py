#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

# global imports
import numpy as np
from scipy import stats, sparse, ndimage, spatial

# local imports
from ptsa.helper import pol2cart

# some functions from MNE
def _get_components(x_in, connectivity):
    """get connected components from a mask and a connectivity matrix"""
    cs_graph_components = sparse.cs_graph_components

    mask = np.logical_and(x_in[connectivity.row], x_in[connectivity.col])
    data = connectivity.data[mask]
    row = connectivity.row[mask]
    col = connectivity.col[mask]
    shape = connectivity.shape
    idx = np.where(x_in)[0]
    row = np.concatenate((row, idx))
    col = np.concatenate((col, idx))
    data = np.concatenate((data, np.ones(len(idx), dtype=data.dtype)))
    connectivity = sparse.coo_matrix((data, (row, col)), shape=shape)
    _, components = cs_graph_components(connectivity)
    # print "-- number of components : %d" % np.unique(components).size
    return components


def find_clusters(x, threshold, tail=0, connectivity=None):
    """For a given 1d-array (test statistic), find all clusters which
    are above/below a certain threshold. Returns a list of 2-tuples.

    Parameters
    ----------
    x: 1D array
        Data
    threshold: float
        Where to threshold the statistic
    tail : -1 | 0 | 1
        Type of comparison
    connectivity : sparse matrix in COO format
        Defines connectivity between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Defaut is None, i.e, no connectivity.

    Returns
    -------
    clusters: list of slices or list of arrays (boolean masks)
        We use slices for 1D signals and mask to multidimensional
        arrays.

    sums: array
        Sum of x values in clusters
    """
    if not tail in [-1, 0, 1]:
        raise ValueError('invalid tail parameter')

    x = np.asanyarray(x)

    if tail == -1:
        x_in = x <= threshold
    elif tail == 1:
        x_in = x >= threshold
    else:
        x_in = np.abs(x) >= threshold

    if connectivity is None:
        labels, n_labels = ndimage.label(x_in)

        if x.ndim == 1:
            clusters = ndimage.find_objects(labels, n_labels)
            sums = ndimage.measurements.sum(x, labels,
                                            index=range(1, n_labels + 1))
        else:
            clusters = list()
            sums = np.empty(n_labels)
            for l in range(1, n_labels + 1):
                c = labels == l
                clusters.append(c)
                sums[l - 1] = np.sum(x[c])
    else:
        if x.ndim > 1:
            raise Exception("Data should be 1D when using a connectivity "
                            "to define clusters.")
        if np.sum(x_in) == 0:
            return [], np.empty(0)
        components = _get_components(x_in, connectivity)
        labels = np.unique(components)
        clusters = list()
        sums = list()
        for l in labels:
            c = (components == l)
            if np.any(x_in[c]):
                clusters.append(c)
                sums.append(np.sum(x[c]))
        sums = np.array(sums)
    return clusters, sums

def pval_from_histogram(T, H0, tail):
    """Get p-values from stats values given an H0 distribution

    For each stat compute a p-value as percentile of its statistics
    within all statistics in surrogate data
    """
    if not tail in [-1, 0, 1]:
        raise ValueError('invalid tail parameter')

    # from pct to fraction
    if tail == -1:  # up tail
        pval = np.array([np.sum(H0 <= t) for t in T])
    elif tail == 1:  # low tail
        pval = np.array([np.sum(H0 >= t) for t in T])
    elif tail == 0:  # both tails
        pval = np.array([np.sum(H0 >= abs(t)) for t in T])
        pval += np.array([np.sum(H0 <= -abs(t)) for t in T])

    pval = (pval + 1.0) / (H0.size + 1.0)  # the init data is one resampling
    return pval


def sparse_dim_connectivity(dim_con):
    """
    Create a sparse matrix capturing the connectivity of a conjunction
    of dimensions.
    """
    # get length of each dim by looping over the connectivity matrices
    # passed in
    dlen = [d.shape[0] for d in dim_con]

    # prepare for full connectivity matrix
    nelements = np.prod(dlen)

    # get the indices
    ind = np.indices(dlen)

    # reshape them
    dind = [ind[i].reshape((nelements,1)) for i in range(ind.shape[0])]
    
    # fill the rows and columns
    rows = []
    cols = []

    # loop to create mix of 
    for i in range(len(dind)):
        # get the connected elements for that dimension
        r,c = np.nonzero(dim_con[i])

        # loop over them
        for j in range(len(r)):
            # extend the row/col connections
            rows.extend(np.nonzero(dind[i]==r[j])[0])
            cols.extend(np.nonzero(dind[i]==c[j])[0])

    # create the sparse connectivity matrix
    data = np.ones(len(rows))
    cmat = sparse.coo_matrix((data,(rows,cols)), shape=(nelements,nelements))

    return cmat

def simple_neighbors_1d(n):
    """
    Return connectivity for simple 1D neighbors.
    """
    c = np.zeros((n,n))
    c[np.triu_indices(n,1)] = 1
    c[np.triu_indices(n,2)] = 0
    return c


def sensor_neighbors(sensor_locs):
    """
    Calculate the neighbor connectivity based on Delaunay
    triangulation of the sensor locations.

    sensor_locs should be the x and y values of the 2-d flattened
    sensor locs.
    """
    # see if loading from file
    if isinstance(sensor_locs,str):
        # load from file
        locs = np.loadtxt(sensor_locs)
        theta = -locs[0] + 90
        radius = locs[1]
        x,y = pol2cart(theta,radius,radians=False)
        sensor_locs = np.vstack((x,y)).T

    # get info about the sensors
    nsens = len(sensor_locs)
    
    # do the triangulation
    d = spatial.Delaunay(sensor_locs)

    # determine the neighbors
    n = [np.unique(d.vertices[np.nonzero(d.vertices==i)[0]])
         for i in range(nsens)]

    # make the symmetric connectivity matrix
    cn = np.zeros((nsens,nsens))
    for r in range(nsens):
        cn[r,n[r]] = 1

    # only keep the upper
    cn[np.tril_indices(nsens)] = 0

    # return it
    return cn


def tfce(x, dt=.1, E=2/3., H=2.0, tail=0, connectivity=None):
    """
    Threshold-Free Cluster Enhancement.
    """
    # test tail value
    if not tail in [-1, 0, 1]:
        raise ValueError('Invalid tail parameter.')

    # make sure array
    x = np.asanyarray(x)

    # figure out thresh range based on tail and the data
    trange = []
    if tail == -1:
        sign = -1.0
        if (x<0).sum()>0:
            trange = np.arange(x[x<0].max(),x.min()-dt,-dt)
    elif tail == 1:
        sign = 1.0
        if (x>0).sum()>0:
            trange = np.arange(x[x>0].min(),x.max()+dt,dt)
    else:
        sign = 1.0
        trange = np.arange(np.abs(x).min(),np.abs(x).max()+dt,dt)

    # get starting values for data (reshape it b/c needs to be 1d)
    xt = np.zeros_like(x).reshape(np.prod(x.shape))
    
    # make own connectivity if not provided so that we have consistent return values
    if connectivity is None:
        connectivity = sparse_dim_connectivity([simple_neighbors_1d(n) for n in x.shape])

    # integrate in steps of dt over the threshold
    # do reshaping once
    xr = x.reshape(np.prod(x.shape))
    for thresh in trange:
        # get the clusters (reshape as necessary)
        clusts,sums = find_clusters(xr, thresh, 
                                    tail=tail, connectivity=connectivity)        

        # add to values in clusters
        for c in clusts:
            # take into account direction of test
            xt[c] += sign * np.power(c.sum(),E) * np.power(sign*thresh,H) * dt

    # return the enhanced data, reshaped back
    return xt.reshape(*(x.shape))
