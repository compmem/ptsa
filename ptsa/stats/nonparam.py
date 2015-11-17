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
from scipy.stats import ttest_ind, ttest_1samp, norm
import sys

def gen_perms(dat, group_var, nperms):
    """
    Generate permutations within a group variable, but across conditions. 

    There is no need to sort your data as this method will shuffle the
    indices properly.

    """
    # grab the unique groups
    ugrp = np.unique(dat[group_var])

    # save indices for each unique group
    grpind = {u:np.nonzero(dat[group_var]==u)[0] for u in ugrp}

    # set the base permutation indices for each unique group
    p_ind = {u:np.arange(len(grpind[u])) for u in ugrp}
    
    # start with actual data
    perms = [np.arange(len(dat))]

    # loop and shuffle for each perm
    for p in xrange(nperms):
        # set the starting indices
        ind = np.arange(len(dat))

        # loop over each group
        for u in ugrp:
            # permute the indices for that group
            perm = np.random.permutation(p_ind[u])

            # insert the permuted group indices into the base index
            np.put(ind,grpind[u],grpind[u][perm])

        # append the shuffled perm to the list of permutations
        perms.append(ind)

    # turn the final perms into an array
    perms = np.array(perms)
    return perms



def ttest_ind_z_one_sided(X,Y):
    # do the test
    t,p = ttest_ind(X,Y)

    # convert the pvals to one-sided tests based on the t
    p = (p/2.)+np.finfo(p.dtype).eps
    p[t>0] = 1-p[t>0]

    # convert the p to a z
    z = norm.ppf(p)
    
    return z


def permutation_test(X, Y=None, parametric=True, iterations=1000):
    """
    Perform a permutation test on paired or non-paired data.

    Observations must be on the first axis.
    """
    # see if paired or not and concat data
    if Y is None:
        paired = True
        data = X
        nX = len(X)
    else:
        paired = False
        data = np.r_[X,Y]
        nX = len(X)
        nY = len(Y)

    # currently no non-parametric
    if not parametric:
        raise NotImplementedError("Currently only parametric stats are supported.")

    # perform stats
    z_boot = []
    if paired:
        # paired stat
        raise NotImplementedError("Currently only non-paired stats are supported.")
        # first on actual data
        #t,p = ttest_1samp(data)
    else:
        # non-paired
        # first on actual data
        z = ttest_ind_z_one_sided(data[:nX],data[nX:])

        # now on random shuffles
        sys.stdout.write('%d: '%iterations)
        for i in xrange(iterations):
            # shuffle it
            sys.stdout.write('%d '%i)
            sys.stdout.flush()
            np.random.shuffle(data)
            z_boot.append(ttest_ind_z_one_sided(data[:nX],data[nX:]))
        sys.stdout.write('\n')
        sys.stdout.flush()

    # convert z_boot to array
    z_boot = np.asarray(z_boot)

    # return those z values
    return z, z_boot
