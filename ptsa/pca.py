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

def pca(X, ncomps=None, eigratio=1e6):
    """
    Principal components analysis

%   [W,Y] = pca(X,NBC,EIGRATIO) returns the PCA matrix W and the principal
%   components Y corresponding to the data matrix X (realizations
%   columnwise). The number of components is NBC components unless the
%   ratio between the maximum and minimum covariance eigenvalue is below
%   EIGRATIO. In such a case, the function will return as few components as
%   are necessary to guarantee that such ratio is greater than EIGRATIO.
    
    """

    if ncomps is None:
        ncomps = X.shape[0]
        
    C = np.cov(X)
    D,V = np.linalg.eigh(C)
    val = np.abs(D)
    I = np.argsort(val)[::-1]
    val = val[I]

    while (val[0]/val[ncomps-1])>eigratio:
        ncomps -= 1

    V = V[:,I[:ncomps]]
    D = np.diag(D[I[:ncomps]]**(-.5))
    W = np.dot(D,V.T)
    Y = np.dot(W,X)

    return W,Y


