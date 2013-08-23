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

# Make it so we can send numpy arrays to R
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

# load some required packages
lme4 = importr('lme4')

import pandas as pd
import pandas.rpy.common as com


"""
run_lmer <- function(frm, dat, perms, perm_var, family='gaussian') {
  for ()
  {
    m = lmer(frm, data=dat[], family=family)
    coef(m)
  }
}
"""


"""
0) Determine perms to be used at each feature

1) Loop over features, run LMER on all perms

2) Optionally run TFCE on perms

3) Aggregate results, optionally with windows.
"""

def lmer_feature(formula_str, dat, perms, perm_var):
    # run on each permutation
    for perm in perms:
        
    dat['pet_score'] = z_scores
    rdf = com.convert_to_r_dataframe(dat)
    try:
        fmstr = "pet_score ~ %s %s + (1|subject) + ((%s - 1) | subject)" % \
                (beh_score_str, resp_str, beh_score_str)
        ms = lme4.lmer(Formula(fmstr), data=rdf)
        res = r['data.frame'](lme4.coef(r['summary'](ms)))
        tvals = res.rx2('t.value')
        return np.array(tvals) #np.array(res.rx2('t.value'))
    except:
        return np.array([np.nan]*rterms)
