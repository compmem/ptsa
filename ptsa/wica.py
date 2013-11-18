#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
import pywt
import sys

from ptsa.pca import pca
from ptsa.iwasobi import iwasobi
from ptsa.wavelet import iswt,swt

try:
    import multiprocessing as mp
    has_mp = True
except ImportError:
    has_mp = False


def find_blinks(dat, L, fast_rate=.5, slow_rate=.975, thresh=None):
    """
    Identify eyeblinks with fast and slow running averages.
    """
    # make the range to go around an eyeblink
    #L = np.int32(np.round(samplerate*0.1))*2
    
    # process the data
    #if thresh is None:
    #    zd = (dat-np.mean(dat))/np.std(dat)
    #else:
    #    zd = dat
    zdf = dat
    zdb = dat[::-1]
    
    # initialize the fast and slow running averages
    fastf = np.zeros(len(dat)+1)
    slowf = np.zeros(len(dat)+1)
    slowf[0] = np.mean(zdf[:10])
    fastb = np.zeros(len(dat)+1)
    slowb = np.zeros(len(dat)+1)
    slowb[0] = np.mean(zdb[:10])
    
    # params for running averages
    a = fast_rate
    b = 1-a
    c = slow_rate
    d = 1-c

    # first forward
    # calc running average
    for i in xrange(len(zdf)):
        fastf[i+1] = a*fastf[i] + b*(zdf[i]-slowf[i])
        slowf[i+1] = c*slowf[i] + d*(zdf[i])

    # remove the first value
    fastf = fastf[1:]
    slowf = slowf[1:]

    # then backward
    # calc running average
    for i in xrange(len(zdb)):
        fastb[i+1] = a*fastb[i] + b*(zdb[i]-slowb[i])
        slowb[i+1] = c*slowb[i] + d*(zdb[i])

    # remove the first value
    fastb = fastb[1:]
    slowb = slowb[1:]

    # combine
    fast = (fastf*fastb)/2.

    # determine the thresh
    if thresh is None:
        thresh = np.std(np.abs(fast))
    
    # determine the artifact indices
    
    # first apply a thresh
    idx = np.nonzero(np.abs(fast)>thresh)[0]
    inds = np.arange(len(dat), dtype=np.int32)

    # make sure to connect contiguous artifacts
    idx_ext = np.zeros(len(idx)*(2*L+1), dtype=np.int32)
    for k in xrange(len(idx)):
        idx_ext[(2*L+1)*(k):(2*L+1)*(k+1)-1] = np.arange(idx[k]-L,idx[k]+L)
    id_noise = np.setdiff1d(inds, idx_ext)
    id_artef = np.setdiff1d(inds, id_noise)
    
    return id_artef,id_noise

def _clean_find_thresh(Y,Kthr,wavelet,L):
    # init
    xn = None
    thld = 0.0

    N = len(Y)
    
    # find the outliers
    # need to replace this with blink-finding code

    if False:
        # Sig = median(abs(Y)/0.6745);
        #Sig = np.median(np.abs(Y)/0.6745)
        #Sig = np.median(np.abs(icaEEG[Comp[c],pure_range[0]:pure_range[1]])/0.6745)
        Sig = np.median(np.abs(Y)/0.6745)
        # Thr = 4*Sig;
        Thr = 3*Sig
        # idx = find(abs(Y) > Thr);
        idx = np.nonzero(np.abs(Y) > Thr)[0]
        # idx_ext = zeros(1,length(idx)*(2*L+1));
        idx_ext = np.zeros(len(idx)*(2*L+1), dtype=np.int32)
        # for k=1:length(idx),
        #     idx_ext((2*L+1)*(k-1)+1:(2*L+1)*k) = [idx(k)-L:idx(k)+L];
        # end
        for k in xrange(len(idx)):
            idx_ext[(2*L+1)*(k):(2*L+1)*(k+1)-1] = np.arange(idx[k]-L,idx[k]+L)
        # id_noise=setdiff((1:N), idx_ext);
        id_noise = np.setdiff1d(range(N), idx_ext)
        # id_artef=setdiff((1:N), id_noise);
        id_artef = np.setdiff1d(range(N), id_noise)
    else:
        id_artef,id_noise = find_blinks(Y,L)

    # make sure it's not all noise or artifact
    print len(id_artef),len(id_noise)

    # if isempty(id_artef),
    #     disp(['The component #' num2str(Comp(c)) ' has passed unchanged']);
    #     continue;
    # end
    if len(id_artef) == 0:
        #sys.stdout.write("passed unchanged\n")
        #sys.stdout.flush()
        return xn, thld
    # KK = 100;
    KK = 100.
    # LL = floor(log2(length(Y)));
    LL = np.int32(np.floor(np.log2(len(Y))))
    # [xl, xh] = mrdwt(Y, h, LL);
    wres = swt(Y,wavelet,level=LL)
    # make it editable
    wres = [list(wres[i]) for i in range(len(wres))]

    # start with a low high-pass threshold and zero out wavelet
    # components below that value, test to see if the artifacts look
    # like the noise, if not, step up the threshold. At some point it
    # should stop, but perhaps not after removing a good bit of
    # signal.

    # while KK > Kthr,
    #     thld = thld + 0.5;
    #     xh = HardTh(xh, thld); % x = (abs(y) > thld).*y;
    #     xd = mirdwt(xl,xh,h,LL); 
    #     xn = Y - xd;
    #     cn=corrcoef(Y(id_noise),xn(id_noise));
    #     ca=corrcoef(Y(id_artef),xd(id_artef));
    #     KK = ca(1,2)/cn(1,2);   
    # end
    # thld = 3.6;
    # not sure where this 3.6 number came from, so I'm dropping it down to get
    # more low-freq cleaning
    thld = 1.1 #3.6
    while KK > Kthr:
        # update what's going on
        #sys.stdout.write('.')
        #sys.stdout.flush()
        # bump up the thresh
        thld += 0.5
        # zero out everything below threshold in each wavelet coef
        for i in xrange(len(wres)):
            wres[i][1] = (np.abs(wres[i][1]) > thld) * wres[i][1]
        # invert the wavelet back
        xd = iswt(wres, wavelet)
        # check if clean based on the ratio of correlations for noise
        # and artifact data
        xn = Y-xd
        # cn measures the correlation between the cleaned and original
        # data in the non-artifactual regions
        cn = np.corrcoef(Y[id_noise],xn[id_noise])[0,1]
        # ca measures the corr b/t the signal removed from the
        # artifacts and the original artifacts
        ca = np.corrcoef(Y[id_artef],xd[id_artef])[0,1]
        # must not go negative, it should just be a small positive
        # number if that happens
        if cn <= 0.0:
            cn = .000001
        if ca <= 0.0:
            ca = .000001
        # we want the ratio of the bad things getting cleaned to the
        # good things sticking around to be small, ideally both very
        # close to 1.0
        KK = ca/cn
        sys.stdout.write('(%.2f,%.2f,%.2f) '%(ca,cn,KK))
        sys.stdout.flush()
    # return the cleaned data and the thresh
    return xn, thld

def _clean_use_thresh(Y,thld,wavelet):
    LL = np.int32(np.floor(np.log2(len(Y))))
    wres = swt(Y,wavelet,level=LL)
    wres = [list(wres[i]) for i in range(len(wres))]
    # xh = HardTh(xh, thld);
    for i in xrange(len(wres)):
        wres[i][1] = (np.abs(wres[i][1]) > thld) * wres[i][1]
    # xd = mirdwt(xl,xh,h,LL);
    xd = iswt(wres, wavelet)
    # xn = Y - xd;
    xn = Y - xd
    # return the cleaned data
    return xn

def _clean_comp(comp, Kthr, L, thld=None):
    wavelet = pywt.Wavelet('db3')
    N = np.int32(2**np.floor(np.log2(len(comp))))

    # Y = icaEEG(Comp(c),1:N);
    Y = comp[:N]
        
    if thld is None:
        # opt(c) = thld;
        xn,thld = _clean_find_thresh(Y,Kthr,wavelet,L)
        if thld == 0.0:
            return comp, thld
    else:
        # just apply the thresh
        if thld == 0.0:
            # it was skipped, so skip it
            return comp, thld
        xn = _clean_use_thresh(Y,thld,wavelet)

    # icaEEG(Comp(c),1:N) = xn;
    comp[:N] = xn
    
    # clean the second half
    # Y = icaEEG(Comp(c),end-N+1:end);
    Y = comp[-N:]
    xn = _clean_use_thresh(Y,thld,wavelet)

    # icaEEG(Comp(c),N+1:end) = xn(end-(Nobser-N)+1:end);
    comp[N:] = xn[-(len(comp)-N):]

    return comp, thld

    
def remove_strong_artifacts(data, A, icaEEG, Comp, Kthr=1.25, F=256,
                            Cthr=None, num_mp_procs=0):
    """
    % This function denoise high amplitude artifacts (e.g. ocular) and remove them from the
    % Independent Components (ICs).
    %

    Ported and enhanced from Matlab code distributed by the authors of:
    
    N.P. Castellanos, and V.A. Makarov (2006). 'Recovering EEG brain signals: Artifact 
    suppression with wavelet enhanced independent component analysis'
    J. Neurosci. Methods, 158, 300--312.
    
    % INPUT:
    %
    % icaEEG - matrix of ICA components (Nchanel x Nobservations)
    %
    % Comp   - # of ICs to be denoised and cleaned (can be a vector)
    %
    % Kthr   - threshold (multiplayer) for denoising of artifacts
    %          (default Kthr = 1.15)
    %
    % F      - acquisition frequency
    %          (default F = 256 Hz)
    %
    % OUTPUT:
    % 
    % opt    - vector of threshold values used for filtering of corresponding
    %          ICs
    %
    % NOTE: If a component has no artifacts of a relatively high amplitude
    %       the function will skip this component (no action), dispaly a
    %       warning and the corresponding output "opt" will be set to zero.

    """
    # make sure not to modify data
    #icaEEG = data.copy()
    # allow to modify to save memory
    #icaEEG = data
    
    # L = round(F*0.1);
    L = np.int32(np.round(F*0.1))
    # [Nchan, Nobser] = size(icaEEG);
    Nchan, Nobser = icaEEG.shape
    # if Nchan > Nobser, 
    #     error('Problem with data orientation, try to transpose the matrix!'); 
    # end
    # N = 2^floor(log2(Nobser));
    #N = np.int32(2**np.floor(np.log2(Nobser)))
    # h = daubcqf(6);
    #wavelet = pywt.Wavelet('db3')
    #h = wavelet.rec_lo

    # eventually make the artifact identification and determination of
    # the filter threshold thld based on only the range provided, then
    # apply it to the entire component.  This will make the algorithm
    # faster (and possibly more accurate) by not having to repeatedly
    # run the wavelet transform on the entire dataset.

    # opt = zeros(1,length(Comp));
    if Cthr is None:
        opt = np.zeros(len(Comp))
        find_thresh = True
    else:
        opt = Cthr
        find_thresh = False


    if has_mp and num_mp_procs != 0:
        po = mp.Pool(num_mp_procs)
        mp_res = []
        
    # for c=1:length(Comp),
    for c in xrange(len(Comp)):
        if find_thresh:
            thld = None
        else:
            thld = opt[c]
        if has_mp and num_mp_procs != 0:
            # call with mp
            mp_res.append(po.apply_async(_clean_comp,
                                         (icaEEG[Comp[c]], Kthr,
                                          L, thld)))
        else:
            sys.stdout.write("Component #%d: "%(Comp[c]))
            sys.stdout.flush()
            comp,thld = _clean_comp(icaEEG[Comp[c]], Kthr, L, thld=thld)
            icaEEG[Comp[c]] = comp
            if find_thresh:
                opt[c] = thld
            if opt[c] > 0.0:
                # disp(['The component #' num2str(Comp(c)) ' has been filtered']);
                sys.stdout.write("was filtered at %f\n"%(opt[c]))
                sys.stdout.flush()
            else:
                sys.stdout.write("passed unchanged\n")
                sys.stdout.flush()

    if has_mp and num_mp_procs != 0:
        # collect results
        po.close()
        po.join()
        for c in xrange(len(Comp)):
            sys.stdout.write("Component #%d: "%(Comp[c]))
            sys.stdout.flush()
            comp,thld = mp_res[c].get()
            icaEEG[Comp[c]] = comp
            if find_thresh:
                opt[c] = thld
            if opt[c] > 0.0:
                # disp(['The component #' num2str(Comp(c)) ' has been filtered']);
                sys.stdout.write("was filtered at %f\n"%(opt[c]))
                sys.stdout.flush()
            else:
                sys.stdout.write("passed unchanged\n")
                sys.stdout.flush()

    # end
    return opt
    #return icaEEG, opt

class WICA(object):
    """
    Clean data with the Wavelet-ICA method described here:

    N.P. Castellanos, and V.A. Makarov (2006). 'Recovering EEG brain signals: Artifact 
    suppression with wavelet enhanced independent component analysis'
    J. Neurosci. Methods, 158, 300--312.

    Instead of using the Infomax ICA algorithm, we use the (much much
    faster) IWASOBI algorithm.

    We also pick components to clean by only cleaning components that
    weigh heavily on the EOG electrodes.

    This ICA algorithm works better if you pass in data that have been
    high-pass filtered to remove big non-neural fluctuations and
    drifts.

    You do not have to run the ICA step on your entire dataset.
    Instead, it is possible to provide the start and end indicies for
    a continguous chunk of data that is 'clean' except for having lots
    of eyeblink examples.  This range will also be used inside the
    wavelet-based artifact correction code to determine the best
    threshold for identifying artifacts.  You do, however, want to try
    and make sure you provide enough samples for a good ICA
    decomposition.  A good rule of thumb is 3*(N^2) where N is the
    number of channels/sources.
    """
    ICA_weights = property(lambda self: self._ICA_weights)

    def __init__(self, data, samplerate, pure_range=None):
        """
        """
        # process the pure range
        if pure_range is None:
            pure_range = (None,None)
        self._pure_range = pure_range

        # run pca
        sys.stdout.write("Running PCA...")
        Wpca,pca_data = pca(data[:,pure_range[0]:pure_range[1]]) #, ncomps, eigratio)

        # Run iwasobi
        sys.stdout.write("Running IWASOBI ICA...")
        sys.stdout.flush()
        #(W,Winit,ISR,signals) = iwasobi(data[:,pure_range[0]:pure_range[1]])
        (W,Winit,ISR,signals) = iwasobi(pca_data)

        # combine the iwasobi weights with the pca weights
        W = np.dot(W,Wpca)
        A = np.linalg.pinv(W)

        # reorder the signals by loading (reordered from high to low)
        ind = np.argsort(np.abs(A).sum(0))[::-1]
        A = A[:,ind]
        W = W[ind,:]
        signals = signals[ind,:]

        self._ICA_weights = A
        #A = np.linalg.inv(W)
        sys.stdout.write("DONE!\n")
        sys.stdout.flush()

        # expand signals to span the entire dataset if necessary
        if (not pure_range[0] is None) or (not pure_range[1] is None):
            #Xmean=data[:,pure_range[0]:pure_range[1]].mean(1)
            #signals = np.add(np.dot(W,data).T,np.dot(W,Xmean)).T
            signals = np.dot(W,data)

        self._components = signals
        self._samplerate = samplerate
        self._data = data

    def pick(self, EOG_elecs=[0,1], std_fact=1.5):
        # pick which signals to clean (ones that weigh on EOG elecs)
        # vals = np.sum(np.abs(A[EOG_elecs,:]),0)
        # std_thresh = std_fact*np.std(vals)
        # comp_ind = np.nonzero(vals>=std_thresh)[0]
        A = self.ICA_weights
        comp_ind = []
        # loop over EOG elecs
        for e in EOG_elecs:
            # get the weights of each component onto that electrode
            vals = np.abs(A[e,:])
            # calculate the threshold that the std must exceed for that
            # component to be considered
            std_thresh = std_fact*np.std(vals)
            #comp_ind.extend(np.nonzero(vals>=std_thresh)[0].tolist())
            # loop over potential components
            for s in np.nonzero(vals>=std_thresh)[0].tolist():
                # calculate the weights of all electrodes into that component
                sweights = np.abs(A[:,s])
                # get threshold based on the std across those weights
                sthresh2 = std_fact*sweights.std()
                # see if that component crosses this second threshold
                if np.abs(A[e,s]) >= sthresh2:
                    # yes, so append to the list to clean
                    comp_ind.append(s)
        # Instead, try and make sure the weights are above the STD thresh
        # AND bigger for EOG elecs than for an elec like Pz

        comp_ind = np.unique(comp_ind)

        return comp_ind


    def get_loading(self, comp):
        return self.ICA_weights[:,comp]


    def clean(self, comp_inds=None, Kthr=2.5, num_mp_procs=0):
        if comp_inds is None:
            comp_inds = self.pick()
        if not isinstance(comp_inds, list):
            comp_inds = [comp_inds]

        # remove strong artifacts
        if (not self._pure_range[0] is None) or (not self._pure_range[1] is None):
            # figure out the thresh for the range
            Cthr = remove_strong_artifacts(self._data[:,self._pure_range[0]:self._pure_range[1]], self.ICA_weights,
                                           self._components[:,self._pure_range[0]:self._pure_range[1]],
                                           comp_inds, Kthr,
                                           self._samplerate,
                                           num_mp_procs=num_mp_procs)
        else:
            Cthr = None
        Cthr = remove_strong_artifacts(self._data,self.ICA_weights,self._components, 
                                       comp_inds, Kthr,
                                       self._samplerate, Cthr,
                                       num_mp_procs=num_mp_procs)
        pass

    def get_corrected(self):
        # return cleaned data back in EEG space
        return np.dot(self.ICA_weights,self._components)

    

def wica_clean(data, samplerate=None, pure_range=(None,None),
               EOG_elecs=[0,1], std_fact=1.5, Kthr=2.5,num_mp_procs=0):
    """
    Clean data with the Wavelet-ICA method described here:

    N.P. Castellanos, and V.A. Makarov (2006). 'Recovering EEG brain signals: Artifact 
    suppression with wavelet enhanced independent component analysis'
    J. Neurosci. Methods, 158, 300--312.

    Instead of using the Infomax ICA algorithm, we use the (much much
    faster) IWASOBI algorithm.

    We also pick components to clean by only cleaning components that
    weigh heavily on the EOG electrodes.

    This ICA algorithm works better if you pass in data that have been
    high-pass filtered to remove big non-neural fluctuations and
    drifts.

    You do not have to run the ICA step on your entire dataset.
    Instead, it is possible to provide the start and end indicies for
    a continguous chunk of data that is 'clean' except for having lots
    of eyeblink examples.  This range will also be used inside the
    wavelet-based artifact correction code to determine the best
    threshold for identifying artifacts.  You do, however, want to try
    and make sure you provide enough samples for a good ICA
    decomposition.  A good rule of thumb is 3*(N^2) where N is the
    number of channels/sources.
    """
    # run pca
    sys.stdout.write("Running PCA...")
    Wpca,pca_data = pca(data[:,pure_range[0]:pure_range[1]]) #, ncomps, eigratio)
    
    # Run iwasobi
    sys.stdout.write("Running IWASOBI ICA...")
    sys.stdout.flush()
    #(W,Winit,ISR,signals) = iwasobi(data[:,pure_range[0]:pure_range[1]])
    (W,Winit,ISR,signals) = iwasobi(pca_data)
    W = np.dot(W,Wpca)
    A = np.linalg.pinv(W)
    #A = np.linalg.inv(W)
    sys.stdout.write("DONE!\n")
    sys.stdout.flush()

    # expand signals to span the entire dataset if necessary
    if (not pure_range[0] is None) or (not pure_range[1] is None):
        #Xmean=data[:,pure_range[0]:pure_range[1]].mean(1)
        #signals = np.add(np.dot(W,data).T,np.dot(W,Xmean)).T
        signals = np.dot(W,data)

    # pick which signals to clean (ones that weigh on EOG elecs)
    # vals = np.sum(np.abs(A[EOG_elecs,:]),0)
    # std_thresh = std_fact*np.std(vals)
    # comp_ind = np.nonzero(vals>=std_thresh)[0]
    comp_ind = []
    # loop over EOG elecs
    for e in EOG_elecs:
        # get the weights of each component onto that electrode
        vals = np.abs(A[e,:])
        # calculate the threshold that the std must exceed for that
        # component to be considered
        std_thresh = std_fact*np.std(vals)
        #comp_ind.extend(np.nonzero(vals>=std_thresh)[0].tolist())
        # loop over potential components
        for s in np.nonzero(vals>=std_thresh)[0].tolist():
            # calculate the weights of all electrodes into that component
            sweights = np.abs(A[:,s])
            # get threshold based on the std across those weights
            sthresh2 = std_fact*sweights.std()
            # see if that component crosses this second threshold
            if np.abs(A[e,s]) >= sthresh2:
                # yes, so append to the list to clean
                comp_ind.append(s)
    # Instead, try and make sure the weights are above the STD thresh
    # AND bigger for EOG elecs than for an elec like Pz

    comp_ind = np.unique(comp_ind)

    sys.stdout.write("Cleaning these components: " + str(comp_ind) + '\n')
    sys.stdout.flush()

    # remove strong artifacts
    if (not pure_range[0] is None) or (not pure_range[1] is None):
        # figure out the thresh for the range
        Cthr = remove_strong_artifacts(signals[:,pure_range[0]:pure_range[1]],
                                       comp_ind,Kthr,
                                       samplerate,
                                       num_mp_procs=num_mp_procs)
    else:
        Cthr = None
    Cthr = remove_strong_artifacts(signals,comp_ind,Kthr,
                                   samplerate,Cthr,
                                   num_mp_procs=num_mp_procs)
    
    # return cleaned data back in EEG space
    return np.dot(A,signals).astype(data.dtype)

