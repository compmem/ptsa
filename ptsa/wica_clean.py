#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
import math
import pywt
import sys

#from ptsa.pca import pca
from ptsa.iwasobi import iwasobi
from ptsa.wavelet import iswt,swt

def remove_strong_artifacts(data, Comp, Kthr=1.25, F=256,
                            pure_range=(None,None)):
    """
    % This function denoise high amplitude artifacts (e.g. ocular) and remove them from the
    % Independent Components (ICs).
    %

    Ported from Matlab code distributed by the authors of:
    
    N.P. Castellanos, and V.A. Makarov (2006). 'Recovering EEG brain signals: Artifact 
    suppression with wavelet enhanced independent component analysis'
    J. Neurosci. Methods, 158, 300–312.
    
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
    icaEEG = data.copy()
    
    # L = round(F*0.1);
    L = np.int32(np.round(F*0.1))
    # [Nchan, Nobser] = size(icaEEG);
    Nchan, Nobser = icaEEG.shape
    # if Nchan > Nobser, 
    #     error('Problem with data orientation, try to transpose the matrix!'); 
    # end
    # N = 2^floor(log2(Nobser));
    N = np.int32(2**np.floor(np.log2(Nobser)))
    # h = daubcqf(6);
    wavelet = pywt.Wavelet('db3')
    h = wavelet.rec_lo
    
    # opt = zeros(1,length(Comp));
    opt = np.zeros(len(Comp), dtype=icaEEG.dtype)
    # for c=1:length(Comp),
    for c in xrange(len(Comp)):
        sys.stdout.write("Component #%d: "%(Comp[c]))
        sys.stdout.flush()
        # Y = icaEEG(Comp(c),1:N);
        Y = icaEEG[Comp[c],:N]
        # Sig = median(abs(Y)/0.6745);
        #Sig = np.median(np.abs(Y)/0.6745)
        Sig = np.median(np.abs(icaEEG[Comp[c],pure_range[0]:pure_range[1]])/0.6745)
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
            sys.stdout.write("passed unchanged\n"%(Comp[c]))
            sys.stdout.flush()
            continue
        # thld = 3.6;
        # not sure where this number came from
        thld = 3.6
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
            thld += 0.5
            for i in xrange(len(wres)):
                wres[i][1] = (np.abs(wres[i][1]) > thld) * wres[i][1]
            xd = iswt(wres, wavelet)
            xn = Y-xd
            cn = np.corrcoef(Y[id_noise],xn[id_noise])
            ca = np.corrcoef(Y[id_artef],xd[id_artef])
            KK = ca[0,1]/cn[0,1]
        # opt(c) = thld;
        opt[c] = thld
        # Y = icaEEG(Comp(c),end-N+1:end);
        Y = icaEEG[Comp[c],-N:]
        # icaEEG(Comp(c),1:N) = xn;
        icaEEG[Comp[c],:N] = xn
        # LL = floor(log2(length(Y)));
        LL = np.int32(np.floor(np.log2(len(Y))))
        # [xl, xh] = mrdwt(Y, h, LL);
        wres = swt(Y,wavelet,level=LL)
        wres = [list(wres[i]) for i in range(len(wres))]
        # xh = HardTh(xh, thld);
        for i in xrange(len(wres)):
            wres[i][1] = (np.abs(wres[i][1]) > thld) * wres[i][1]
        # xd = mirdwt(xl,xh,h,LL);
        xd = iswt(wres, wavelet)
        # xn = Y - xd;
        xn = Y - xd
        # icaEEG(Comp(c),N+1:end) = xn(end-(Nobser-N)+1:end);
        icaEEG[Comp[c],N:] = xn[-(Nobser-N):]
        # disp(['The component #' num2str(Comp(c)) ' has been filtered']);
        sys.stdout.write("was filtered\n"%(Comp[c]))
        sys.stdout.flush()
    # end
    return icaEEG, opt


def wica_clean(data, samplerate=None, pure_range=(None,None),
               EOG_elecs=[0,1], std_fact=1.5, Kthr=2.5):
    """
    Clean data with the Wavelet-ICA method described here:

    N.P. Castellanos, and V.A. Makarov (2006). 'Recovering EEG brain signals: Artifact 
    suppression with wavelet enhanced independent component analysis'
    J. Neurosci. Methods, 158, 300–312.

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
    #sys.stdout.write("Running PCA...")
    #Wpca,data = pca.pca(data[:,ica_range[0]:ica_range[1]], ncomps, eigratio)
    
    # Run iwasobi
    sys.stdout.write("Running IWASOBI ICA...")
    sys.stdout.flush()
    (W,Winit,ISR,signals) = iwasobi.iwasobi(data[:,pure_range[0]:pure_range[1]])
    #W = np.dot(W,Wpca)
    #A = np.linalg.pinv(W)
    A = np.linalg.inv(W)
    sys.stdout.write("DONE!\n")
    sys.stdout.flush()

    # expand signals to span the entire dataset if necessary
    if (not pure_range[0] is None) or (not pure_range[1] is None):
        #Xmean=data[:,pure_range[0]:pure_range[1]].mean(1)
        #signals = np.add(np.dot(W,data).T,np.dot(W,Xmean)).T
        signals = np.dot(W,data)

    # pick which signals to clean (ones that weigh on EOG elecs)
    vals = np.sum(np.abs(A[EOG_elecs,:]),0)
    std_thresh = std_fact*np.std(vals)
    ind = np.nonzero(vals>=std_thresh)[0]
    
    # remove strong artifacts
    clean_signals,opt = remove_strong_artifacts(signals,ind,Kthr,
                                                samplerate,pure_range)
    
    # return cleaned data back in EEG space
    return np.dot(A,clean_signals).astype(data.dtype)

