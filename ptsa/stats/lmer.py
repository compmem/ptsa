#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


import os
import sys
import time
import tempfile
import numpy as np
from numpy.lib.recfunctions import append_fields
from scipy.linalg import diagsvd
from scipy.stats import rankdata

from joblib import Parallel,delayed

# Connect to an R session
import rpy2.robjects
r = rpy2.robjects.r

# For a Pythonic interface to R
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula, FactorVector
from rpy2.robjects.environments import Environment
from rpy2.robjects.vectors import DataFrame, Vector, FloatVector
from rpy2.rinterface import MissingArg,SexpVector

# Make it so we can send numpy arrays to R
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

# load some required packages
# PBS: Eventually we should try/except these to get people 
# to install missing packages
lme4 = importr('lme4')
rstats = importr('stats')
fdrtool = importr('fdrtool')
ssvd = importr('ssvd')
if hasattr(lme4,'coef'):
    r_coef  = lme4.coef
else:
    r_coef = rstats.coef
if hasattr(lme4,'model_matrix'):
    r_model_matrix = lme4.model_matrix
else:
    r_model_matrix = rstats.model_matrix

#import pandas as pd
#import pandas.rpy.common as com

# load ptsa clustering
import cluster


def lmer_feature(formula_str, dat, perms=None, 
                 val=None, factors=None, **kwargs):
    """
    Run LMER on a number of permutations of the predicted data.

    
    """
    # get the perm_var
    perm_var = formula_str.split('~')[0].strip()

    # set the val if necessary
    if not val is None:
        dat[perm_var] = val

    # make factor list if necessary
    if factors is None:
        factors = []

    # convert the recarray to a DataFrame
    rdf = DataFrame({k:(FactorVector(dat[k]) 
                        if (k in factors) or isinstance(dat[k][0],str) 
                        else dat[k]) 
                     for k in dat.dtype.names})
    
    #rdf = com.convert_to_r_dataframe(pd.DataFrame(dat),strings_as_factors=True)
    

    # get the column index
    col_ind = list(rdf.colnames).index(perm_var)

    # make a formula obj
    rformula = Formula(formula_str)

    # just apply to actual data if no perms
    if perms is None:
        #perms = [np.arange(len(dat))]
        perms = [None]

    # run on each permutation
    tvals = None
    for i,perm in enumerate(perms):
        if not perm is None:
            # set the perm
            rdf[col_ind] = rdf[col_ind].rx(perm+1)

        # inside try block to catch convergence errors
        try:
            ms = lme4.lmer(rformula, data=rdf, **kwargs)
        except:
            continue
            #tvals.append(np.array([np.nan]))
        # extract the result

        df = r['data.frame'](r_coef(r['summary'](ms)))
        if tvals is None:
            # init the data
            # get the row names
            rows = list(r['row.names'](df))
            tvals = np.rec.fromarrays([np.ones(len(perms))*np.nan 
                                       for ro in range(len(rows))],
                                      names=','.join(rows))
        tvals[i] = tuple(df.rx2('t.value'))

    return tvals

class LMER():
    """
    Wrapper for the lmer method provided by lme4 in R. 

    This object facilitates fitting the same model multiple times 
    and extracting the associated t-stat.

    """
    def __init__(self, formula_str, df, factors=None, 
                 resid_formula_str=None, **lmer_opts):
        """
        """
        # get the pred_var
        pred_var = formula_str.split('~')[0].strip()

        # add column if necessary
        if not pred_var in df.dtype.names:
            # must add it
            df = append_fields(df, pred_var, [0.0]*len(df), usemask=False)

        # make factor list if necessary
        if factors is None:
            factors = {}
        # add in missingarg for any potential factor not provided
        for k in df.dtype.names:
            if isinstance(df[k][0],str) and not factors.has_key(k):
                factors[k] = MissingArg
                
        for f in factors:
            if factors[f] is None:
                factors[f] = MissingArg
            # checking for both types of R Vectors for rpy2 variations
            elif (not isinstance(factors[f],Vector) and 
                  not factors[f] == MissingArg):
                factors[f] = Vector(factors[f])

        # convert the recarray to a DataFrame (releveling if desired)
        self._rdf = DataFrame({k:(FactorVector(df[k], levels=factors[k]) 
                                  if (k in factors) or isinstance(df[k][0],str) 
                                  else df[k]) 
                               for k in df.dtype.names})
        #self._rdf = com.convert_to_r_dataframe(pd.DataFrame(df),strings_as_factors=True)


        # get the column index
        self._col_ind = list(self._rdf.colnames).index(pred_var)

        # make a formula obj
        self._rformula = Formula(formula_str)

        # make one for resid if necessary
        if resid_formula_str:
            self._rformula_resid = Formula(resid_formula_str)
        else:
            self._rformula_resid = None

        # save the args
        self._lmer_opts = lmer_opts

        # model is null to start
        self._ms = None

    def run(self, vals=None, perms=None):

        # set the col with the val
        if not vals is None:
            self._rdf[self._col_ind] = vals

        # just apply to actual data if no perms
        if perms is None:
            perms = [None]
            #perms = [np.arange(len(self._rdf[self._col_ind]))]

        # run on each permutation
        tvals = None
        log_likes = None
        for i,perm in enumerate(perms):
            if not perm is None:
                # set the perm
                self._rdf[self._col_ind] = self._rdf[self._col_ind].rx(perm+1)

            # inside try block to catch convergence errors
            try:
                if self._rformula_resid:
                    # get resid first
                    msr = lme4.lmer(self._rformula_resid, data=self._rdf, 
                                    **self._lmer_opts)
                    self._rdf[self._col_ind] = lme4.resid(msr)
                # run the model (possibly on the residuals from above)
                ms = lme4.lmer(self._rformula, data=self._rdf, **self._lmer_opts)
            except:
                continue
                #tvals.append(np.array([np.nan]))
            
            # save the model
            if self._ms is None:
                self._ms = ms
                if self._rformula_resid:
                    self._msr = msr

            # extract the result
            df = r['data.frame'](r_coef(r['summary'](ms)))
            if tvals is None:
                # init the data
                # get the row names
                rows = list(r['row.names'](df))
                tvals = np.rec.fromarrays([np.ones(len(perms))*np.nan 
                                           for ro in range(len(rows))],
                                          names=','.join(rows))
                log_likes = np.zeros(len(perms))

            # set the values
            tvals[i] = tuple(df.rx2('t.value'))
            log_likes[i] = float(r['logLik'](ms)[0])

        return tvals, log_likes




"""LMER-PLS/MELD Notes

We can optionally have this run on rank data, too, so that it's
non-parametric. In this case we would rank the continuous variables
and the predicted data, but not the factors.

I'm not sure, yet, how this will work for whole-brain RSA. In that
case all the pairwise comparisons are our observations. We can
then correlate them with the behavior distances, just like in
behavior-PLS and then perform the SVD on those correlations. That
will give us a component weighted based on the strength of the
correlation (which could be performed non-parametrically). Then we
perform the across-subject lmer on the whole-brain data
transformed by the components, repeating this process for each
permutation. Then we do the same for the bootstrap to get the
stable weights. 

Note that this would have to run on neural distances calculated at all
the spheres over the entire brain and data would have to be in
template space.

For RSA-PLS analyses we'd also need to provide a way to do custom
permutations. The bootstrap would be the same :)

It would be great to figure out how to integrate this within a
joint-modeling framework.

"""

def procrustes(orig_lv, boot_lv):
    # define coordinate space between orignal and bootstrap LVs
    temp=np.dot(orig_lv,boot_lv.T)

    # orthogonalze space
    U,s,V = np.linalg.svd(temp)

    # determine procrustean transform
    #rotatemat=U*V'    
    return np.dot(U,V)

def pick_stable_features(R, shape, nboot=500, connectivity=None):
    # generate the boots
    boots = [np.random.random_integers(0,len(R)-1,len(R))
             for i in xrange(nboot)]

    # run tfce on each subj and cond
    if True:
        Rt = np.zeros([R.shape[0],R.shape[1]]+list(shape))
        for i in range(Rt.shape[0]):
            for j in range(Rt.shape[1]):
                Rt[i,j] = cluster.tfce(np.arctanh(R[i,j]).reshape(*shape),
                                       dt=.05,tail=1,connectivity=connectivity, 
                                       E=1.0,H=2.0)
                Rt[i,j] += cluster.tfce(np.arctanh(R[i,j]).reshape(*shape),
                                        dt=.05,tail=-1,connectivity=connectivity, 
                                        E=1.0,H=2.0)
    else:
        # convert to Z
        Rt = np.arctanh(R)

    # calc bootstrap ratio
    Rtb = np.array([Rt[boots[b]].mean(0) for b in range(len(boots))])
    Rtbr = Rt.mean(0)/Rtb.std(0)
    return Rtbr

def sparse_stable_svd(R, nboot=50):
    # generate the boots
    boots = [np.random.random_integers(0,len(R)-1,len(R))
             for i in xrange(nboot)]

    # calc the original SVD
    U, s, Vh = np.linalg.svd(np.concatenate(R), full_matrices=False)
    
    # do the boots
    rVs = []
    for i in range(len(boots)):
        Ub, sb, Vhb = np.linalg.svd(np.concatenate(R[boots[i]]), full_matrices=False)

        rmat = procrustes(U,Ub)

        rVs.append(np.dot(rmat,np.dot(diagsvd(sb,len(sb),len(sb)),Vhb)))
        
    # get the bootstrap ratios
    rVs = np.array(rVs)
    Vs = np.dot(diagsvd(s,len(s),len(s)),Vh)
    boot_ratio = Vs/rVs.std(0)
    
    # pass the boot ratios through fdrtool to pick stable features
    fachist = np.histogram(boot_ratio.flatten(),bins=500)
    peak = fachist[1][fachist[0]==np.max(fachist[0])][0]
    results = fdrtool.fdrtool(FloatVector(boot_ratio.flatten()-peak), statistic='normal', 
                              plot=False, verbose=False)
    qv = np.array(results.rx('qval')).reshape(boot_ratio.shape)
    #qv = None
    # apply the thresh
    return U,s,Vh,qv,boot_ratio

# global container so that we can use joblib with a smaller memory
# footprint (i.e., we don't need to duplicate everything)
_global_meld = {}

def _eval_model(model_id, perm=None, boot=None):
    # arg check
    if not perm is None and not boot is None:
        raise ValueError("Both and perm and boot should not be non-None." +
                         "It's either or neither.")

    # set vars from model
    mm = _global_meld[model_id]
    _R = mm._R

    # Calculate R
    R = []

    # set the boot shuffle
    if boot is None:        
        ind_b = np.arange(len(mm._groups))
    else:
        ind_b = boot

    # loop over group vars
    ind = {}
    for i,k in enumerate(mm._groups[ind_b]):
        # grab the A and M
        A = mm._A[k]
        M = mm._M[k]

        # gen a perm for that subj
        if perm is None:
            ind[k] = np.arange(len(A))
        else:
            ind[k] = perm[k]            

        if perm is None and not _R is None:
            # reuse R we already calculated
            R.append(mm._R[ind_b[i]])
        else:
            # calc the correlation
            R.append(np.inner(A.T,M[ind[k]].T))

    # save R before concatenating if not a permutation
    if perm is None and boot is None and _R is None:
        _R = np.array(R)

    # concatenate R for SVD
    R_nocat = np.array(R)
    R = np.concatenate(R)
    # if not boot is None:
    #     # get the R for that boot
    #     R = np.concatenate(_R[boot][:])
    # else:
    #     # concatenate all groups
    #     R = np.concatenate(R)

    # zero invariant features
    feat_mask = np.isnan(R)
    R[feat_mask] = 0.0

    # pick only stable features
    Rtbr = pick_stable_features(R_nocat, mm._feat_shape, connectivity = None)
    # conjunction across conditions
    #stable_ind = ((np.abs(Rtbr)>2.575).sum(0)>0).flatten()
    stable_ind = np.abs(Rtbr)>2.575
    stable_ind = stable_ind.reshape((stable_ind.shape[0],-1))

    # zero out non-stable
    R_nocat[:,~stable_ind] = 0.0
    R = np.concatenate([R_nocat[i] for i in range(len(R_nocat))])

    # perform SVD
    if mm._use_ssvd:
        #res = ssvd.ssvd(R, method='method', r=len(R))
        #Vh = np.concatenate(res.rx('v')).T
        #s = np.concatenate(res.rx('d'))
        #U = np.concatenate(res.rx('u')).T

        U, s, Vh, qv, boot_ratio = sparse_stable_svd(R_nocat)
        ssV_full = np.sqrt((Vh**2).sum(1))
        #Vh[np.abs(boot_ratio)<2.95] = 0.0
        qvt_br = qv[np.abs(boot_ratio) > 
                    np.percentile(np.abs(boot_ratio).flat, 99.5)].max()
        qvt = max(qv.min()+.01, .5)
        qvt = min(qvt,qvt_br)
        Vh[qv>qvt] = 0.0
        ssV_sparse = np.sqrt((Vh**2).sum(1))
        s[s>0] *= ssV_sparse[s>0]/ssV_full[s>0]
        #s[np.abs(Vh).sum(1)==0.0] = 0.0
    else:
        U, s, Vh = np.linalg.svd(R, full_matrices=False)

    # calc prop of variance accounted for
    if mm._ss is None:
        #_ss = np.sqrt((s*s).sum())
        _ss = s.sum()
    else:
        _ss = mm._ss
    #ss /= ss.sum()
    ss = s

    # set up lmer
    O = None
    lmer = None
    if mm._mer is None or not boot is None:
        O = [mm._O[i].copy() for i in ind_b]
        if not boot is None:
            # replace the group
            for i,k in enumerate(mm._groups):
                O[i][mm._re_group] = k

        lmer = LMER(mm._formula_str, np.concatenate(O),
                    factors=mm._factors, 
                    resid_formula_str=mm._resid_formula_str, **mm._lmer_opts)
        mer = None
    else:
        mer = mm._mer

    if perm is None and boot is None and mm._null_formula_str:
        run_null = True
        if mm._mer_null is None:
            lmer_null = LMER(mm._null_formula_str, np.concatenate(O),
                             factors=mm._factors, 
                             resid_formula_str=mm._resid_formula_str, **mm._lmer_opts)
        mer_null is None
    else:
        run_null = False
        mer_null = mm._mer_null
    
    # loop over LVs performing LMER
    #Dw = []
    res = []
    res_null = []
    ll_null_boot = []
    ll_full_boot = []
    for i in range(len(Vh)):
        if ss[i] <= 0.0:
            #print 'skipped ',str(i)
            continue

        # flatten then weigh features via dot product
        Dw = np.concatenate([np.dot(mm._D[k][ind[k]],Vh[i])
                             for g,k in enumerate(mm._groups[ind_b])])

        # run the main model
        if mer is None:
            res.append(lmer.run(vals=Dw))
            mer = lmer._ms
        else:
            mer = r['refit'](mer, FloatVector(Dw))
            df = r['data.frame'](r_coef(r['summary'](mer)))
            rows = list(r['row.names'](df))
            new_tvals = np.rec.fromarrays([[tv] for tv in tuple(df.rx2('t.value'))],
                                          names=','.join(rows))
            new_ll = float(r['logLik'](mer)[0])
            res.append((new_tvals,np.array([new_ll])))

        # see if run null model
        if perm is None and boot is None and mm._null_formula_str:
            res_null.append(lmer_null.run(vals=Dw))
            m_null = lmer_null._ms
            m_full = lmer._ms
            ll_null = []
            ll_full = []
            for bi in xrange(mm._num_null_boot):
                # sim new data
                #Dw_sim = np.array(r['simulate'](m_null))[0]
                Dw_sim = r['simulate'](m_null)
                # run full and null models, saving the loglike
                #ll_null.append(lmer_null.run(vals=Dw_sim)[1])
                #ll_full.append(lmer.run(vals=Dw_sim)[1])
                ll_null.append(float(r['logLik'](r['refit'](m_null,Dw_sim))[0]))
                ll_full.append(float(r['logLik'](r['refit'](m_full,Dw_sim))[0]))
            # append to full list
            ll_null_boot.append(ll_null)
            ll_full_boot.append(ll_full)

    if len(res) == 0:
        # must make dummy data
        if lmer is None:
            O = [mm._O[i].copy() for i in ind_b]
            if not boot is None:
                # replace the group
                for i,k in enumerate(mm._groups):
                    O[i][mm._re_group] = k

            lmer = LMER(mm._formula_str, np.concatenate(O),
                        factors=mm._factors, 
                        resid_formula_str=mm._resid_formula_str, **mm._lmer_opts)

        Dw = np.random.randn(len(np.concatenate(O)))
        temp_t,temp_ll = lmer.run(vals=Dw)

        for n in temp_t.dtype.names: temp_t[n] = 0.0
        temp_ll[0] = 0.0
        res.append((temp_t,temp_ll))

    tvals,log_likes = zip(*res)
    tvals = np.concatenate(tvals)
    log_likes = np.concatenate(log_likes)
    if perm is None and boot is None and mm._null_formula_str:
        tvals_null,log_likes_null = zip(*res_null)
        log_likes_null = np.concatenate(log_likes_null)

        print "Like Ratio: ", 2*(log_likes.sum() - log_likes_null.sum())
        print "Like Ratio Scaled: ", 2*(np.dot(log_likes,ss[ss>0.0]/_ss) - 
                                        np.dot(log_likes_null,ss[ss>0.0]/_ss))
        print "Like Ratio Comb: ", np.dot(2*(log_likes-log_likes_null),ss[ss>0.0]/_ss)

        ll_null_boot = np.vstack(ll_null_boot)
        ll_full_boot = np.vstack(ll_full_boot)

        lr = 2*(log_likes-log_likes_null)
        lr_boot = 2*(ll_full_boot-ll_null_boot)
        print "Like Boot: ", np.dot(lr_boot, ss[ss>0.0]/_ss)
        sscale = ss/ss.sum()
        mg = np.dot([((lr_boot[:,ii]-lr[ii])>0).sum() for ii in range(20)],sscale)
        print "prop: ", mg/float(mm._num_null_boot)
        1/0

    # # get the term names (skip the intercept)
    # names = res.dtype.names[1:]

    # # calc the SSU for each component
    # nterms = len(names)
    # #ssU = {n:np.array([s[c]*np.sqrt((U[c][i::nterms]**2).sum()) 
    # #                   for c in range(len(s))])
    # #       for i,n in enumerate(names)}
    # ssU = {}
    # ssU_sum = np.zeros(len(s))
    # for i,n in enumerate(names):
    #     ssU[n] = np.array([np.sqrt((U[c][i::nterms]**2).sum())
    #                        for c in range(len(s))])
    #     ssU_sum += ssU[n]
    # for n in names:
    #     ssU[n] = (ssU[n]*s)/ssU_sum

    # recombine and scale the tvals across components
    if boot is None:
        ts = np.rec.fromarrays([np.dot(tvals[k],ss[ss>0.0]/_ss) #/(ss>0.).sum()
                                for k in tvals.dtype.names],
                               names= ','.join(tvals.dtype.names))
        # tvals = np.rec.fromarrays([np.dot(tvals[k],ssU[k][ss>0.0]) #/(ss>0.).sum()
        #                            for k in names],
        #                           names= ','.join(names))

    # see if calc tfs for first or bootstrap
    if perm is None:
        tfs = []
        for k in tvals.dtype.names:
            tfs.append(np.dot(tvals[k],
                              np.dot(diagsvd(ss[ss>0], len(ss[ss>0]),len(ss[ss>0])),
                                     Vh[ss>0,...]))) #/(ss>0).sum())
        tfs = np.rec.fromarrays(tfs, names=','.join(tvals.dtype.names))
        # for k in names:
        #     tfs.append(np.dot(res[k],
        #                       np.dot(diagsvd(ssU[k][ss>0], len(ssU[k][ss>0]),len(ssU[k][ss>0])),
        #                              Vh[ss>0,...]))) #/(ss>0).sum())
        # tfs = np.rec.fromarrays(tfs, names=','.join(names))


    # decide what to return
    if perm is None and boot is None:
        # return tvals, tfs, and R for actual non-permuted data
        out = (ts,tfs,_R,feat_mask,_ss,mer,mer_null)
    elif perm is None:
        # return the boot features
        out = tfs
    else:
        # return the tvals for the terms
        out = ts

    return out

def lmer_feat(mer, Dw):
    mer = r['refit'](mer, FloatVector(Dw))
    df = r['data.frame'](r_coef(r['summary'](mer)))
    rows = list(r['row.names'](df))
    new_tvals = np.rec.fromarrays([[tv] for tv in tuple(df.rx2('t.value'))],
                                  names=','.join(rows))
    return new_tvals



def _memmap_array(x, memmap_dir=None):
    if memmap_dir is None:
        memmap_dir = tempfile.gettempdir()
    filename = os.path.join(memmap_dir,str(id(x))+'.npy')
    np.save(filename,x)
    return np.load(filename,'r')


class MELD(object):
    """Mixed Effects for Large Data (MELD)

    me = MELD('val ~ beh + rt', '(beh|subj) + (rt|subj)', 'subj'
              dep_data, ind_data, factors = ['beh', 'subj'])
    me.run_perms(200)
    me.run_boots(200)

    If you provide ind_data as a dict with a separate recarray for
    each group, you must ensure the columns match.


    """
    def __init__(self, fe_formula, re_formula,
                 re_group, dep_data, ind_data, 
                 factors=None, row_mask=None,
                 use_ranks=False, use_norm=True,
                 memmap=False, memmap_dir=None,
                 resid_formula=None,
                 null_formula=None, num_null_boot=0,
                 svd_terms=None, use_ssvd=False,
                 #nperms=500, nboot=100, 
                 n_jobs=1, verbose=10,
                 lmer_opts=None):
        """
        """
        if verbose>0:
            sys.stdout.write('Initializing...')
            sys.stdout.flush()
            start_time = time.time()

        # save the formula
        self._formula_str = fe_formula + ' + ' + re_formula

        # see if there's a resid formula
        if resid_formula:
            # the random effects are the same
            self._resid_formula_str = resid_formula + ' + ' + re_formula
        else:
            self._resid_formula_str = None

        # see if there's a null formula
        if null_formula:
            # the random effects are the same
            self._null_formula_str = null_formula + ' + ' + re_formula
        else:
            self._null_formula_str = None
        self._num_null_boot = num_null_boot

        # save whether using ranks
        self._use_ranks = use_ranks

        # see whether to use sparse svd
        self._use_ssvd = use_ssvd

        # see if memmapping
        self._memmap = memmap

        # save job info
        self._n_jobs = n_jobs
        self._verbose = verbose

        # eventually fill the feature shape
        self._feat_shape = None

        # fill A,M,O,D
        self._A = {}
        self._M = {}
        self._O = {}
        self._D = {}
        O = []

        # loop over unique grouping var
        self._re_group = re_group
        if isinstance(ind_data, dict):
            # groups are the keys
            self._groups = np.array(ind_data.keys())
        else:
            # groups need to be extracted from the recarray
            self._groups = np.unique(ind_data[re_group])
        for g in self._groups:
            # get that subj inds
            if isinstance(ind_data,dict):
                # the index is just the group into that dict
                ind_ind = g
            else:
                # select the rows based on the group
                ind_ind = ind_data[re_group]==g

            # process the row mask
            if row_mask is None:
                # no mask, so all good
                row_ind = np.ones(len(ind_data[ind_ind]),dtype=np.bool)
            elif isinstance(row_mask, dict):
                # pull the row_mask from the dict
                row_ind = row_mask[g]
            else:
                # index into it with ind_ind
                row_ind = row_mask[ind_ind]
            
            # extract that group's A,M,O
            # first save the observations (rows of A)
            self._O[g] = ind_data[ind_ind][row_ind]
            if use_ranks:
                # loop over non-factors and rank them
                for n in self._O[g].dtype.names:
                    if (n in factors) or isinstance(self._O[g][n][0],str):
                        continue
                    self._O[g][n] = rankdata(self._O[g][n])
            O.append(self._O[g])

            # eventually allow for dict of data files for dep_data
            if isinstance(dep_data,dict):
                # the index is just the group into that dict
                dep_ind = g
            else:
                # select the rows based on the group
                dep_ind = ind_ind

            # save feature shape if necessary
            if self._feat_shape is None:
                self._feat_shape = dep_data[dep_ind].shape[1:]

            # Save D index into data
            self._D[g] = dep_data[dep_ind][row_ind]
            # reshape it
            self._D[g] = self._D[g].reshape((self._D[g].shape[0],-1))
            if use_ranks:
                if verbose>0:
                    sys.stdout.write('Ranking %s...'%(str(g)))
                    sys.stdout.flush()

                for i in xrange(self._D[g].shape[1]):
                    self._D[g][:,i] = rankdata(self._D[g][:,i])

            # reshape M, so we don't have to do it repeatedly
            self._M[g] = self._D[g].copy() #dep_data[ind].reshape((dep_data[ind].shape[0],-1))
                
            # normalize M
            if use_norm:
                self._M[g] -= self._M[g].mean(0)
                self._M[g] /= np.sqrt((self._M[g]**2).sum(0))

            # determine A from the model.matrix
            rdf = DataFrame({k:(FactorVector(self._O[g][k]) 
                                if k in factors else self._O[g][k]) 
                             for k in self._O[g].dtype.names})
            
            # model spec as data frame
            ms = r['data.frame'](r_model_matrix(Formula(fe_formula), data=rdf))

            cols = list(r['names'](ms))
            if svd_terms is None:
                self._svd_terms = [c for c in cols if not 'Intercept' in c]
            else:
                self._svd_terms = svd_terms
            self._A[g] = np.concatenate([np.array(ms.rx(c)) 
                                         for c in self._svd_terms]).T
                                         #for c in cols if not 'Intercept' in c]).T

            if use_ranks:
                for i in xrange(self._A[g].shape[1]):
                    self._A[g][:,i] = rankdata(self._A[g][:,i])

            # normalize A
            if True: #use_norm:
                self._A[g] -= self._A[g].mean(0)
                self._A[g] /= np.sqrt((self._A[g]**2).sum(0))

            # memmap if desired
            if self._memmap:
                self._M[g] = _memmap_array(self._M[g], memmap_dir)
                self._D[g] = _memmap_array(self._D[g], memmap_dir)

        # concat the Os together and make an LMER instance
        #O = np.concatenate(O)
        #self._O = np.vstack(O)
        #self._O = np.array(O)
        self._O = O
        if lmer_opts is None:
            lmer_opts = {}
        self._lmer_opts = lmer_opts
        self._factors = factors
        #self._lmer = LMER(self._formula_str, O, factors=factors, **lmer_opts)

        # prepare for the perms and boots
        self._perms = []
        self._boots = []
        self._tp = []
        self._tb = []

        if verbose>0:
            sys.stdout.write('Done (%.2g sec)\n'%(time.time()-start_time))
            sys.stdout.write('Processing actual data...')
            sys.stdout.flush()
            start_time = time.time()

        global _global_meld
        _global_meld[id(self)] = self

        # run for actual data (returns both perm and boot vals)
        self._R = None
        self._ss = None
        self._mer = None
        self._mer_null = None
        tp,tb,R,feat_mask,ss,mer,mer_null = _eval_model(id(self),None, None)
        self._R = R
        self._tp.append(tp)
        self._tb.append(tb)
        self._feat_mask = feat_mask
        self._ss = ss
        self._mer = mer
        self._mer_null = mer_null

        if verbose>0:
            sys.stdout.write('Done (%.2g sec)\n'%(time.time()-start_time))
            sys.stdout.flush()

    def __del__(self):
        # get self id
        my_id = id(self)

        # clean up memmapping files
        if self._memmap:
            for g in self._M.keys():
                try:
                    filename = self._M[g].filename 
                    del self._M[g]
                    os.remove(filename)
                except OSError:
                    pass
            for g in self._D.keys():
                try:
                    filename = self._D[g].filename 
                    del self._D[g]
                    os.remove(filename)
                except OSError:
                    pass

        # clean self out of global model list
        global _global_meld
        if _global_meld and _global_meld.has_key(my_id):
            del _global_meld[my_id]



    def run_perms(self, perms, n_jobs=None, verbose=None):
        """Run the specified permutations.

        This method will append to the permutations you have already
        run.

        """
        if n_jobs is None:
            n_jobs = self._n_jobs
        if verbose is None:
            verbose = self._verbose

        if not isinstance(perms,list):
            # perms is nperms
            nperms = perms

            # gen the perms ahead of time
            perms = []
            for p in xrange(nperms):
                ind = {}
                for k in self._groups:
                    # gen a perm for that subj
                    ind[k] = np.random.permutation(len(self._A[k]))

                perms.append(ind)
        else:
            # calc nperms
            nperms = len(perms)

        if verbose>0:
            sys.stdout.write('Running %d permutations...\n'%nperms)
            sys.stdout.flush()
            start_time = time.time()

        # save the perms
        self._perms.extend(perms)

        # res = []
        # for i,perm in enumerate(perms):
        #     if verbose>0:
        #         sys.stdout.write('%d '%i)
        #         sys.stdout.flush()
        #     res.append(self._eval_model(perm=perm, boot=None,
        #                                 n_jobs=n_jobs, verbose=0))
        res = Parallel(n_jobs=n_jobs, 
                       verbose=verbose)(delayed(_eval_model)(id(self),perm,None)
                                        for perm in perms)
        self._tp.extend(res)

        if verbose>0:
            sys.stdout.write('Done (%.2g sec)\n'%(time.time()-start_time))
            sys.stdout.flush()


    def run_boots(self, boots, n_jobs=None, verbose=None):
        """Run the specified bootstraps.

        This method will append to the bootstraps you have already
        run.

        """
        if n_jobs is None:
            n_jobs = self._n_jobs
        if verbose is None:
            verbose = self._verbose

        if isinstance(boots,list):
            # get the nboots
            nboots = len(boots)
        else:
            # boots is nboots
            nboots = boots

            # calculate the boots with replacement
            boots = [np.random.random_integers(0,len(self._R)-1,len(self._R))
                     for i in xrange(nboots)]

        if verbose>0:
            sys.stdout.write('Running %d bootstraps...\n'%nboots)
            sys.stdout.flush()
            start_time = time.time()

        # save the boots
        self._boots.extend(boots)

        # run in parallel if desired
        # res = []
        # for i,boot in enumerate(boots):
        #     if verbose>0:
        #         sys.stdout.write('%d '%i)
        #         sys.stdout.flush()
        #     res.append(self._eval_model(perm=None, boot=boot,
        #                                 n_jobs=n_jobs, verbose=0))
        res = Parallel(n_jobs=n_jobs, 
                      verbose=verbose)(delayed(_eval_model)(id(self),None,boot)
                                       for boot in boots)
        self._tb.extend(res)

        if verbose>0:
            sys.stdout.write('Done (%.2g sec)\n'%(time.time()-start_time))
            sys.stdout.flush()


    @property
    def terms(self):
        return self._tp[0].dtype.names

    @property
    def t_terms(self):
        return self._tp[0]

    @property
    def t_features(self):
        names = [n for n in self.terms
                 if n != '(Intercept)']
        tfeat = [self._tb[0][n].reshape(self._feat_shape)
                 for n in names]
        return np.rec.fromarrays(tfeat, names=','.join(names))

    @property
    def pvals_uncorrected(self):
        """Return p-value of each LMER term.
        """
        # fancy way to stack recarrays
        tvals = self._tp[0].__array_wrap__(np.hstack(self._tp))
        #allt = np.abs(np.vstack([tvals[n] 
        #                         for n in tvals.dtype.names
        #                         if n != '(Intercept)']))
        #pvals = {n:np.mean(allt.flatten()>=np.abs(tvals[n][0]))
        names = [n for n in tvals.dtype.names
                 if n != '(Intercept)']
        pvals = [np.mean(np.abs(tvals[n])>=np.abs(tvals[n][0]))
                 for n in names]
        pvals = np.rec.fromarrays(pvals, names=','.join(names))
        return pvals

    @property
    def pvals(self):
        """Return Holm-corrected p-values of each LMER term.
        """
        pvals = self.pvals_uncorrected
        names = pvals.dtype.names
        # scale the pvals
        ind = np.argsort([1-pvals[n] for n in names]).argsort()+1
        for i,n in enumerate(names):
            pvals[n] = (pvals[n]*ind[i]).clip(0,1)
        # ensure monotonicity
        # reorder ind to go from smallest to largest p
        ind = (-ind).argsort()
        for i in range(1,len(ind)):
            if pvals[names[ind[i]]] < pvals[names[ind[i-1]]]:
                pvals[names[ind[i]]] = pvals[names[ind[i-1]]]
            
        return pvals
        
    @property
    def boot_ratio(self):
        """Return the bootstrap ratio for each feature.

        This can be treated as a Z and thresholded to determine
        whether each feature contributed to the term-level result
        (e.g., a ratio >= 2.57 would be equivalent to a two-tailed p
        <= .01).

        """
        # fancy way to stack recarrays
        tfs = self._tb[0].__array_wrap__(np.hstack(self._tb))
        names = [n for n in tfs.dtype.names
                 if n != '(Intercept)']
        brs = []
        for n in names:
            # reshape back to number of bootstraps
            tf = tfs[n].reshape((len(self._tb),-1))
            # calculate the bootstrap ratio and shape to feature space
            brs.append((tf[0]/tf.std(0)).reshape(self._feat_shape))

        brs = np.rec.fromarrays(brs, names=','.join(names))

        return brs

    @property
    def fdr_boot(self):
        """Calculate the False Discovery Rate on the bootstrap ratios.

        Makes use of the fdrtool package in R, which estimates the
        signal and null distributions across your features.
        """
        # get the boot ratios
        brs = self.boot_ratio
        names = brs.dtype.names
        qvals = []
        for n in names:
            # get R vector of bootstrap ratios
            br = brs[n].flatten()
            good_ind = ~np.isnan(br)
            qv = np.ones_like(br)
            br = FloatVector(br[good_ind])

            # calc the fdr
            results = fdrtool.fdrtool(br, statistic='normal', 
                                      plot=False, verbose=False)

            # append the qvals
            qv[good_ind] = np.array(results.rx('qval'))
            
            qvals.append(qv.reshape(self._feat_shape))

        # convert to recarray
        qvals = np.rec.fromarrays(qvals, names=','.join(names))

        # grab the qs
        return qvals


if __name__ == '__main__':

    # test some MELD
    n_jobs = -1

    # generate some fake data
    nobs = 100
    nsubj = 10
    nfeat = (50,100)
    nperms = 50
    nboots = 50
    use_ranks = False
    s = np.concatenate([np.array(['subj%02d'%i]*nobs) for i in range(nsubj)])
    beh = np.concatenate([np.array([1]*(nobs/2) + [0]*(nobs/2)) 
                          for i in range(nsubj)])
    # observations (data frame)
    ind_data = np.rec.fromarrays((np.zeros(len(s)),
                                  beh,
                                  np.random.rand(len(s)),s),
                                 names='val,beh,beh2,subj')

    # data with observations in first dimension and features on the remaining
    dep_data = np.random.rand(len(s),*nfeat)
    print 'Data shape:',dep_data.shape

    # now with signal
    # add in some signal
    dep_data_s = dep_data.copy()
    for i in range(0,20,2):
        for j in range(2):
            dep_data_s[:,4,i+j] += (ind_data['beh'] * (i+1)/200.)

    # smooth the data
    import scipy.ndimage
    dep_data = scipy.ndimage.gaussian_filter(dep_data, [0,1,1])
    dep_data_s = scipy.ndimage.gaussian_filter(dep_data_s, [0,1,1])
    

    # # run without signal
    # # set up the lmer_pht
    # me = MELD('val ~ beh+beh2', '(1|subj)', 'subj',
    #           dep_data, ind_data, factors = {'subj':None},
    #           use_ranks=use_ranks, use_ssvd=False,
    #           n_jobs=n_jobs)
    # me.run_perms(nperms)
    # #me.run_boots(nboots)

    # run with signal
    # set up the lmer_pht
    me_s = MELD('val ~ beh+beh2', '(1|subj)', 'subj',
                dep_data_s, ind_data, factors = {'subj':None},
                use_ranks=use_ranks, use_ssvd=False,
                n_jobs=n_jobs)
    me_s.run_perms(nperms)
    #me.run_boots(nboots)


    # # explore the results
    # print
    # print "No signal!"
    # print "----------"
    # print "Terms:",me.terms
    # print "t-vals:",me.t_terms
    # print "term p-vals:",me.pvals
    # #brs = me.boot_ratio
    # #print "Bootstrap ratio shape:",brs.shape
    # #print "BR num sig:",[(n,(me.fdr_boot[n]<=.05).sum())
    # #                     for n in brs.dtype.names]

    print
    print "Now with signal!"
    print "----------------"
    print "Terms:",me_s.terms
    print "t-vals:",me_s.t_terms
    print "term p-vals:",me_s.pvals
    #brs_s = me_s.boot_ratio
    #print "Bootstrap ratio shape:",brs_s.shape
    #print "BR num sig:",[(n,(me_s.fdr_boot[n]<=.05).sum())
    #                     for n in brs_s.dtype.names]


