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

def _clean_find_thresh(Y,Kthr,wavelet,L):
    # init
    xn = None
    thld = 0.0

    N = len(Y)
    
    # Sig = median(abs(Y)/0.6745);
    #Sig = np.median(np.abs(Y)/0.6745)
    #Sig = np.median(np.abs(icaEEG[Comp[c],pure_range[0]:pure_range[1]])/0.6745)
    Sig = np.median(np.abs(Y)/0.6745)
    # Thr = 4*Sig;
    Thr = 4*Sig
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
    # if isempty(id_artef),
    #     disp(['The component #' num2str(Comp(c)) ' has passed unchanged']);
    #     continue;
    # end
    if len(id_artef) == 0:
        #sys.stdout.write("passed unchanged\n")
        #sys.stdout.flush()
        return xn, thld
    # thld = 3.6;
    # not sure where this 3.6 number came from, so I'm dropping it down
    thld = 2.1 #3.6
    # KK = 100;
    KK = 100.
    # LL = floor(log2(length(Y)));
    LL = np.int32(np.floor(np.log2(len(Y))))
    # [xl, xh] = mrdwt(Y, h, LL);
    wres = swt(Y,wavelet,level=LL)
    # make it editable
    wres = [list(wres[i]) for i in range(len(wres))]
    # while KK > Kthr,
    #     thld = thld + 0.5;
    #     xh = HardTh(xh, thld); % x = (abs(y) > thld).*y;
    #     xd = mirdwt(xl,xh,h,LL); 
    #     xn = Y - xd;
    #     cn=corrcoef(Y(id_noise),xn(id_noise));
    #     ca=corrcoef(Y(id_artef),xd(id_artef));
    #     KK = ca(1,2)/cn(1,2);   
    # end
    while KK > Kthr:
        sys.stdout.write('.')
        sys.stdout.flush()
        for i in xrange(len(wres)):
            wres[i][1] = (np.abs(wres[i][1]) > thld) * wres[i][1]
        xd = iswt(wres, wavelet)
        xn = Y-xd
        cn = np.corrcoef(Y[id_noise],xn[id_noise])
        ca = np.corrcoef(Y[id_artef],xd[id_artef])
        KK = ca[0,1]/cn[0,1]
        thld += 0.5
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

    
def remove_strong_artifacts(data, Comp, Kthr=1.25, F=256,
                            Cthr=None,num_mp_procs=0):
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
    % icaEEG - matrix of cleaned indenpendent components
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
    icaEEG = data
    
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
                sys.stdout.write("was filtered\n"%(Comp[c]))
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
                sys.stdout.write("was filtered\n")
                sys.stdout.flush()
            else:
                sys.stdout.write("passed unchanged\n")
                sys.stdout.flush()

    # end
    return icaEEG, opt


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
        clean_signals,Cthr = remove_strong_artifacts(signals[:,pure_range[0]:pure_range[1]],
                                                     comp_ind,Kthr,
                                                     samplerate,
                                                     num_mp_procs=num_mp_procs)
    else:
        Cthr = None
    clean_signals,Cthr = remove_strong_artifacts(signals,comp_ind,Kthr,
                                                 samplerate,Cthr,
                                                 num_mp_procs=num_mp_procs)
        
    
    # return cleaned data back in EEG space
    return np.dot(A,clean_signals).astype(data.dtype)

