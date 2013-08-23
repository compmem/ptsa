#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


import numpy as np

# Connect to an R session
import rpy2.robjects
r = rpy2.robjects.r

# For a Pythonic interface to R
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula
from rpy2.robjects.environments import Environment
from rpy2.robjects.vectors import DataFrame,IntVector

# Make it so we can send numpy arrays to R
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

# load some required packages
lme4 = importr('lme4')


"""
0) Determine perms to be used at each feature

1) Loop over features, run LMER on all perms

2) Optionally run TFCE on perms

3) Aggregate results, optionally with windows.
"""

def gen_perms(dat, group_var, nperms):
    """
    Generate permutations within a group variable. 

    There is no need to sort your data as this method will shuffle the
    indices properly.

    """
    ugrp = np.unique(dat[group_var])
    grpind = {u:np.nonzero(dat[group_var]==u)[0] for u in ugrp}
    p_ind = {u:np.arange(len(grpind[u])) for u in ugrp}
    
    # start with actual data
    perms = [np.arange(len(dat))]
    for p in xrange(nperms):
        ind = np.arange(len(dat))
        for u in ugrp:
            perm = np.random.permutation(p_ind[u])
            np.put(ind,grpind[u],grpind[u][perm])
        perms.append(ind)
    perms = np.array(perms)
    return perms


def lmer_feature(formula_str, dat, perms, perm_var, **kwargs):
    # convert the recarray to a DataFrame
    rdf = DataFrame({k:dat[k] for k in dat.dtype.names})

    # get the column index
    col_ind = list(rdf.colnames).index(perm_var)

    # make a formula obj
    rformula = Formula(formula_str)

    # run on each permutation
    tvals = []
    for perm in perms:
        # set the perm
        rdf[col_ind] = rdf[col_ind].rx(perm+1)

        # inside try block to catch convergence errors
        try:
            ms = lme4.lmer(rformula, data=rdf, **kwargs)
            tvals.append(np.array(list(r['data.frame'](lme4.coef(r['summary'](ms))).rx2('t.value'))))
        except:
            tvals.append(np.array([np.nan]))

    return np.array(tvals)

if __name__ == '__main__':
    
    s = np.concatenate([np.array(['subj%02d'%i]*5) for i in range(3)])
    dat = np.rec.fromarrays((np.random.randn(len(s)),
                             np.random.randn(len(s)),s),names='val,beh,subj')
    perms = gen_perms(dat,'subj',10)
    t = lmer_feature('val ~ beh + (1|subj)',dat,perms,'val')
