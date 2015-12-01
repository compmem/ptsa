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
import scipy.stats.distributions as dists

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
#import rpy2.robjects as ro
#import  rpy2.robjects.numpy2ri as numpy2ri
#ro.conversion.py2ri = numpy2ri
#numpy2ri.activate()

# load some required packages
# PBS: Eventually we should try/except these to get people 
# to install missing packages
lme4 = importr('lme4')
rstats = importr('stats')
if hasattr(lme4,'coef'):
    r_coef  = lme4.coef
else:
    r_coef = rstats.coef
if hasattr(lme4,'model_matrix'):
    r_model_matrix = lme4.model_matrix
else:
    r_model_matrix = rstats.model_matrix

# load ptsa clustering
import cluster
from stat_helper import fdr_correction

# deal with warnings for bootstrap
import warnings

class InstabilityWarning(UserWarning):
    """Issued when results may be unstable."""
    pass

# On import, make sure that InstabilityWarnings are not filtered out.
warnings.simplefilter('always',InstabilityWarning)


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
                ms = lme4.lmer(self._rformula, data=self._rdf,
                               **self._lmer_opts)
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

def R_to_tfce(R, connectivity=None, shape=None, 
              dt=.01, E=2/3., H=2.0):
    """Apply TFCE to the R values."""
    # allocate for tfce
    Rt = np.zeros_like(R)
    Z = np.arctanh(R)
    # loop
    for i in range(Rt.shape[0]):
        for j in range(Rt.shape[1]):
            # apply tfce in pos and neg direction
            Rt[i,j] += cluster.tfce(Z[i,j].reshape(*shape),
                                   dt=dt,tail=1,connectivity=connectivity, 
                                   E=E,H=H).flatten()
            Rt[i,j] += cluster.tfce(Z[i,j].reshape(*shape),
                                    dt=dt,tail=-1,connectivity=connectivity, 
                                    E=E,H=H).flatten()
    return Rt

def pick_stable_features(Z, nboot=500):
    """Use a bootstrap to pick stable features.
    """
    # generate the boots
    boots = [np.random.random_integers(0,len(Z)-1,len(Z))
             for i in xrange(nboot)]

    # calc bootstrap ratio
    Zb = np.array([Z[boots[b]].mean(0) for b in range(len(boots))])
    Zbr = Z.mean(0)/Zb.std(0)

    # ignore any nans
    Zbr[np.isnan(Zbr)]=0.

    # bootstrap ratios are supposedly t-distributed, so test sig
    Zbr = dists.t(len(Z)-1).cdf(-1*np.abs(Zbr))*2.
    Zbr[Zbr>1]=1
    return Zbr

# global container so that we can use joblib with a smaller memory
# footprint (i.e., we don't need to duplicate everything)
_global_meld = {}

def _eval_model(model_id, perm=None):
    # set vars from model
    mm = _global_meld[model_id]
    _R = mm._R

    # Calculate R
    R = []

    ind_b = np.arange(len(mm._groups))

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

        if perm is None and not mm._R is None:
            # reuse R we already calculated
            R.append(mm._R[ind_b[i]])
        else:
            # calc the correlation
            #R.append(np.inner(A.T,M[ind[k]].T))
            R.append(np.dot(A.T,M[ind[k]]))

    # turn R into array
    R_nocat = np.array(R)
    
    # zero invariant features
    feat_mask = np.isnan(R)
    R_nocat[feat_mask] = 0.0

    if mm._do_tfce:
        # turn to Z, then TFCE
        R_nocat = R_to_tfce(R_nocat, connectivity = mm._connectivity, 
                            shape=mm._feat_shape, 
                            dt=mm._dt, E=mm._E, H=mm._H)
    else:
        # turn to Z
        R_nocat = np.arctanh(R_nocat)
                
    # pick only stable features
    # NOTE: R_nocat is no longer R, it's either TFCE or Z
    Rtbr = pick_stable_features(R_nocat, nboot=mm._feat_nboot)

    # apply the thresh
    stable_ind = Rtbr<mm._feat_thresh
    stable_ind = stable_ind.reshape((stable_ind.shape[0],-1))

    # zero out non-stable
    feat_mask[:,~stable_ind] = True
    R_nocat[:,~stable_ind] = 0.0

    # save R before concatenating if not a permutation
    if perm is None and _R is None:
        _R = R_nocat.copy()

    # concatenate R for SVD
    # NOTE: It's really either Z or TFCE now
    R = np.concatenate([R_nocat[i] for i in range(len(R_nocat))])

    # perform svd
    #U, s, Vh = np.linalg.svd(np.arctanh(R), full_matrices=False)
    U, s, Vh = np.linalg.svd(R, full_matrices=False)

    # fix near zero vals from SVD
    Vh[np.abs(Vh)<(.00000001*mm._dt)] = 0.0

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
    if mm._mer is None:
        O = [mm._O[i].copy() for i in ind_b]

        lmer = LMER(mm._formula_str, np.concatenate(O),
                    factors=mm._factors, 
                    resid_formula_str=mm._resid_formula_str, **mm._lmer_opts)
        mer = None
    else:
        mer = mm._mer
    
    # loop over LVs performing LMER
    res = []
    for i in range(len(Vh)):
        if ss[i] <= 0.0:
            #print 'skipped ',str(i)
            continue

        # flatten then weigh features via dot product
        Dw = np.concatenate([np.dot(mm._D[k][ind[k]],Vh[i])
                             for g,k in enumerate(mm._groups[ind_b])])

        # run the main model
        if mer is None:
            # run the model for the first time and save it
            res.append(lmer.run(vals=Dw))
            mer = lmer._ms
        else:
            # use the saved model and just refit it for speed
            mer = r['refit'](mer, FloatVector(Dw))
            df = r['data.frame'](r_coef(r['summary'](mer)))
            rows = list(r['row.names'](df))
            new_tvals = np.rec.fromarrays([[tv] for tv in tuple(df.rx2('t.value'))],
                                          names=','.join(rows))
            new_ll = float(r['logLik'](mer)[0])
            res.append((new_tvals,np.array([new_ll])))

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

        # must make ss, too
        ss = np.array([1.0])
        #print "perm fail"

    # pull out data from all the components
    tvals,log_likes = zip(*res)
    tvals = np.concatenate(tvals)
    log_likes = np.concatenate(log_likes)

    # recombine and scale the tvals across components
    ts = np.rec.fromarrays([np.dot(tvals[k],ss[ss>0.0]/_ss) #/(ss>0.).sum()
                            for k in tvals.dtype.names],
                           names= ','.join(tvals.dtype.names))

    # scale tvals across features
    tfs = []
    for k in tvals.dtype.names:
        tfs.append(np.dot(tvals[k],
                          np.dot(diagsvd(ss[ss>0], len(ss[ss>0]),len(ss[ss>0])),
                                 Vh[ss>0,...]))) #/(ss>0).sum())
    tfs = np.rec.fromarrays(tfs, names=','.join(tvals.dtype.names))

    # decide what to return
    if perm is None:
        # return tvals, tfs, and R for actual non-permuted data
        out = (ts,tfs,_R,feat_mask,_ss,mer)
    
    else:
        # return the tvals for the terms
        out = (ts,tfs,~feat_mask[0])

    return out


def _memmap_array(x, memmap_dir=None):
    if memmap_dir is None:
        memmap_dir = tempfile.gettempdir()
    filename = os.path.join(memmap_dir,str(id(x))+'.npy')
    np.save(filename,x)
    return np.load(filename,'r')


class MELD(object):
    """Mixed Effects for Large Datasets (MELD)

    me = MELD('val ~ beh + rt', '(beh|subj) + (rt|subj)', 'subj'
              dep_data, ind_data, factors = ['beh', 'subj'])
    me.run_perms(200)

    If you provide ind_data as a dict with a separate recarray for
    each group, you must ensure the columns match.


    """
    def __init__(self, fe_formula, re_formula,
                 re_group, dep_data, ind_data, 
                 factors=None, row_mask=None,
                 use_ranks=False, use_norm=True,
                 memmap=False, memmap_dir=None,
                 resid_formula=None,
                 svd_terms=None, feat_thresh=0.05, 
                 feat_nboot=1000, do_tfce=False, 
                 connectivity=None, shape=None, 
                 dt=.01, E=2/3., H=2.0,
                 #nperms=500, nboot=100, 
                 n_jobs=1, verbose=10,
                 lmer_opts=None):
        """

        dep_data can be an array or a dict of arrays (possibly
        memmapped), one for each group.

        ind_data can be a rec_array for each group or one large rec_array
        with a grouping variable.

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

        # save whether using ranks
        self._use_ranks = use_ranks

        # see the thresh for keeping a feature
        self._feat_thresh = feat_thresh
        self._feat_nboot = feat_nboot
        self._do_tfce = do_tfce
        self._connectivity=connectivity
        self._dt=dt
        self._E=E 
        self._H=H

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
                    # rank it
                    self._D[g][:,i] = rankdata(self._D[g][:,i])

                    # normalize it
                    self._D[g][:,i] = (self._D[g][:,i] - 1)/(len(self._D[g][:,i])-1)

            # reshape M, so we don't have to do it repeatedly
            self._M[g] = self._D[g].copy() 
            
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

            #self._A[g] = np.vstack([ms[c] #np.array(ms.rx(c)) 
            self._A[g] = np.concatenate([np.array(ms.rx(c)) 
                                         for c in self._svd_terms]).T

            if use_ranks:
                for i in xrange(self._A[g].shape[1]):
                    # rank it
                    self._A[g][:,i] = rankdata(self._A[g][:,i])

                    # normalize it
                    self._A[g][:,i] = (self._A[g][:,i] - 1)/(len(self._A[g][:,i])-1)

            # normalize A
            if True: #use_norm:
                self._A[g] -= self._A[g].mean(0)
                self._A[g] /= np.sqrt((self._A[g]**2).sum(0))

            # memmap if desired
            if self._memmap:
                self._M[g] = _memmap_array(self._M[g], memmap_dir)
                self._D[g] = _memmap_array(self._D[g], memmap_dir)


        self._O = O
        if lmer_opts is None:
            lmer_opts = {}
        self._lmer_opts = lmer_opts
        self._factors = factors


        # prepare for the perms and boots and jackknife
        self._perms = []
        ##self._boots = []
        self._tp = []
        self._tb = []
        self._tj = []
        self._pfmask = []

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
        tp,tb,R,feat_mask,ss,mer = _eval_model(id(self),None)
        self._R = R
        self._tp.append(tp)
        self._tb.append(tb)
        self._feat_mask = feat_mask
        self._fmask = ~feat_mask[0]
        self._pfmask.append(~feat_mask[0])
        self._ss = ss
        self._mer = mer

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

        # must use the threading backend
        res = Parallel(n_jobs=n_jobs, 
                       verbose=verbose,
                       backend='threading')(delayed(_eval_model)(id(self),perm)
                                            for perm in perms)
        tp,tfs,feat_mask = zip(*res)
        self._tp.extend(tp)
        self._tb.extend(tfs)
        self._pfmask.extend(feat_mask)

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
    def p_features(self):
        tpf = self._tb[0].__array_wrap__(np.hstack(self._tb))
        pfmasks = np.array(self._pfmask).transpose((1,0,2))
        nperms = np.float(len(self._perms)+1)

        pfs = []
        names = [n for n in tpf.dtype.names
                         if n != '(Intercept)']

        # convert all of the ts to ps
        for i,n in enumerate(names):
            tf = tpf[n]
            tf = np.abs(tf.reshape(nperms,-1))
            fmask = pfmasks[i]
            tf[~fmask]=0
            # make null T dist within term
            nullTdist = tf.max(1)
            nullTdist.sort()
            # use searchsorted to get indicies for turning ts into ps, then divide by number of perms
            # got this from http://stackoverflow.com/questions/18875970/comparing-two-numpy-arrays-of-different-length
            pf = ((nperms-np.searchsorted(nullTdist,tf.flatten(),'left'))/nperms).reshape((nperms,-1))
            pfs.append(pf)

        # pfs is terms by perms by features
        pfs = np.array(pfs)

        # make null p distribution
        nullPdist = pfs.min(2).min(0)
        nullPdist.sort()

        # get pvalues for each feature for each term
        pfts = np.searchsorted(nullPdist,pfs[:,0,:].flatten(),'right').reshape(((-1,)+self._feat_shape))/nperms

        # reconstruct the recarray
        pfts = np.rec.fromarrays(pfts, names=','.join(names))
        return pfts

    

if __name__ == '__main__':
    np.random.RandomState(seed = 42)

    # test some MELD
    n_jobs = -1
    verbose = 20

    # generate some fake data
    nobs = 100
    nsubj = 10
    nfeat = (10,30)
    nperms = 50
    use_ranks = False
    smoothed = False
    memmap = False
    
    s = np.concatenate([np.array(['subj%02d'%i]*nobs) for i in range(nsubj)])
    beh = np.concatenate([np.array([1]*(nobs/2) + [0]*(nobs/2)) 
                          for i in range(nsubj)])
    # observations (data frame)
    ind_data = np.rec.fromarrays((np.zeros(len(s)),
                                  beh,
                                  np.random.rand(len(s)),s),
                                 names='val,beh,beh2,subj')

    # data with observations in first dimension and features on the remaining
    dep_data = np.random.randn(len(s),*nfeat)
    print 'Data shape:',dep_data.shape

    # now with signal
    # add in some signal
    dep_data_s = dep_data.copy()
    for i in range(0,20,2):
        for j in range(2):
            dep_data_s[:,4,i+j] += (ind_data['beh'] * (i+1)/50.)
            dep_data_s[:,5,i+j] += (ind_data['beh'] * (i+1)/50.)
    
    # smooth the data
    if smoothed:
        import scipy.ndimage
        dep_data = scipy.ndimage.gaussian_filter(dep_data, [0,1,1])
        dep_data_s = scipy.ndimage.gaussian_filter(dep_data_s, [0,1,1])


    print "Starting MELD test"
    print "beh has signal, beh2 does not"
    me_s = MELD('val ~ beh+beh2', '(1|subj)', 'subj',
                dep_data_s, ind_data, factors = {'subj':None},
                use_ranks=use_ranks, 
                feat_nboot=1000, feat_thresh=0.05,
                do_tfce=True,
                connectivity=None, shape=None, 
                dt=.01, E=2/3., H=2.0,
                n_jobs=n_jobs,verbose=verbose,
                memmap=memmap,
                #lmer_opts={'control':lme4.lmerControl(#optimizer="nloptwrap",
                #                                      optimizer="Nelder_Mead",
                #                                      optCtrl=r['list'](maxfun=100000))
                #       }
    )
    me_s.run_perms(nperms)
    pfts = me_s.p_features
    print "Number of signifcant features:",[(n,(pfts[n]<=.05).sum())
                                 for n in pfts.dtype.names]



