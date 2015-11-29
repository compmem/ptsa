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
from scipy.linalg import toeplitz,hankel

#import pdb
#from IPython.Debugger import Tracer; debug_here = Tracer()

class IWASOBI():
    """
    Implements algorithm WASOBI for blind source separation of
    AR sources in a fast way, allowing separation up to 100 sources
    in the running time of the order of tens of seconds.

    Ported from MATLAB code by Jakub Petkov / Petr Tichavsky
    """
    def __init__(self, ar_max=10, rmax=0.99, eps0=5.0e-7):
        """
        """
        self.ar_max = ar_max
        self.rmax = rmax
        self.eps0 = eps0
        
    def __call__(self, data):

        x = data
        num_iterations = 3

        # get the shape
        d,N = x.shape

        # get mean and remove it
        Xmean=x.astype(np.float64).mean(1)
        # x=x-Xmean*ones(1,N);  %%%%%%%%%  removing the sample mean
        x = np.subtract(x.astype(np.float64).T,Xmean).T
        
        # T=length(x(1,:))-AR_order;
        T = N-self.ar_max
        
        # C0=corr_est(x,T,AR_order);
        C0 = self.corr_est(x.astype(np.float64),T,self.ar_max)
        
        # for k=2:AR_order+1
        #     ik=d*(k-1);
        #     C0(:,ik+1:ik+d)=0.5*(C0(:,ik+1:ik+d)+C0(:,ik+1:ik+d)');
        # end      %%%%%%%%% symmetrization
        for k in xrange(1,self.ar_max+1):
            ik = d*(k)
            C0[:,ik:ik+d] = 0.5*(C0[:,ik:ik+d]+C0[:,ik:ik+d].T)

        # [Winit Ms] = uwajd(C0,20); %%% compute initial separation
        #                                %%% using uniform weights
        Winit,Ms = self.uwajd(C0,20)
 
        # %conver
        # %t1 = cputime-time_start;
        # W=Winit;
        W = Winit.copy()
        
        # for in = 1:num_iterations
        #     [H ARC]=weights(Ms,rmax,eps0);
        #     [W Ms]=wajd(C0,H,W,5);
        # end
        for i in xrange(num_iterations):
            H,ARC = self.weights(Ms,self.rmax,self.eps0)
            W,Ms = self.wajd(C0,H,W,5)

            
        # ISR=CRLB4(ARC)/N;
        ISR = self.CRLB4(ARC)/np.float(N)
        
        # %t1 = [t1 cputime-time_start];
        # signals=W*x+(W*Xmean)*ones(1,N);
        signals = np.add(np.dot(W,x).T,np.dot(W,Xmean)).T

        return (W,Winit,ISR,signals)

    def THinv5(self,phi,K,M,eps):
        """
         function G=THinv5(phi,K,M,eps)
         %
         %%%% Implements fast (complexity O(M*K^2))
         %%%% computation of the following piece of code:
         %
         %C=[];
         %for im=1:M 
         %  A=toeplitz(phi(1:K,im),phi(1:K,im)')+hankel(phi(1:K,im),phi(K:2*K-1,im)')+eps(im)*eye(K);
         %  C=[C inv(A)];
         %end  
         %
         % DEFAULT PARAMETERS: M=2; phi=randn(2*K-1,M); eps=randn(1,2);
         %   SIZE of phi SHOULD BE (2*K-1,M).
         %   SIZE of eps SHOULD BE (1,M).
        """

        # %C=[];
        C = []
        # %for im=1:M 
        # %  A=toeplitz(phi(1:K,im),phi(1:K,im)')+hankel(phi(1:K,im),phi(K:2*K-1,im)')+eps(im)*eye(K);
        # %  C=[C inv(A)];
        # %end
        for im in xrange(M):
            A = (toeplitz(phi[:K,im],phi[:K,im].T) + 
                 hankel(phi[:K,im],phi[K-1:2*K,im].T) +
                 eps[im]*np.eye(K))
            C.append(np.linalg.inv(A))

        return np.concatenate(C,axis=1)
            
        # # phi(2*K,1:M)=0;
        # phi[2*K-1,:M] = 0
        # # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # # almold=2*phi(1,:)+eps;
        # almold = 2*phi[0,:]+eps
        # # C0=1./almold;
        # C0 = 1/almold
        # # x1=zeros(K,M); x2=x1; x3=x1; x4=x1;
        # x1 = np.zeros((K,M))
        # x2 = np.zeros((K,M))
        # x3 = np.zeros((K,M))
        # x4 = np.zeros((K,M))
        # # x1(1,:)=C0; x2(1,:)=C0;
        # x1[0,:] = C0
        # x2[0,:] = C0
        # # x3(1,:)=-C0.*phi(2,:);
        # x3[0,:] = -C0*phi[1,:]
        # # x4(1,:)=-2*C0.*phi(2,:);
        # x4[0,:] = -2.*C0*phi[1,:]
        # # x4old=[];
        # x4old = []
        # # lalold=2*phi(2,:)./almold;
        # lalold = 2*phi[1,:]/almold
        # # for k=1:K-1
        # for k in xrange(K-1):
        #     #     f2o=phi(k+1:-1:2,:)+phi(k+1:2*k,:);
        #     f2o = phi[k+1:0:-1,:] + phi[k+1:2*k+1,:]
        #     #     alm=sum(f2o.*x4(1:k,:),1)+phi(1,:)+eps+phi(2*k+1,:);
        #     #     a0=zeros(1,M); 
        #     #     if k<K-1
        #     #        a0=phi(k+2,:);
        #     #     end   
        #     #     gam1=sum(f2o.*x1(1:k,:),1);
        #     #     gam3=sum(f2o.*x3(1:k,:),1)+a0+phi(k,:);
        #     #     x4(k+1,:)=ones(1,M);
        #     #     b1m=sum(([phi(2:k+1,:); a0]+[zeros(1,M); phi(1:k,:)]).*x4(1:k+1,:));
        #     #     b2m=sum(([a0; phi(k+1:-1:2,:)]+phi(k+2:2*k+2,:)).*x4(1:k+1,:));
        #     #     latemp=b2m./alm;
        #     #     b2m=latemp-lalold; lalold=latemp;
        #     #     bom=alm./almold;
        #     #     ok=ones(k+1,1);
        #     #     x2(1:k+1,:)=x4(1:k+1,:).*(ok*(1./alm));
        #     #     x1(1:k+1,:)=[x1(1:k,:); zeros(1,M)]-(ok*gam1).*x2(1:k+1,:);
        #     #     x3(1:k+1,:)=[x3(1:k,:); zeros(1,M)]-(ok*gam3).*x2(1:k+1,:);
        #     #     x4temp=x4(1:k,:);
        #     #     x4(1:k+1,:)=[zeros(1,M); x4(1:k,:)]+[x4(2:k,:); ones(1,M); zeros(1,M)]...
        #     #        -(ok*bom).*[x4old; ones(1,M); zeros(1,M)]...
        #     #        -(ok*b2m).*x4(1:k+1,:)-(ok*b1m).*x1(1:k+1,:)-(ok*x4(1,:)).*x3(1:k+1,:);
        #     #     x4old=x4temp;
        #     #     almold=alm;
        # # end  % of for
        # # MK=M*K;
        # MK=M*K
        # # G=zeros(K,MK);
        # G = np.zeros((K,MK))
        # # G(:,1:K:MK)=x1; clast=zeros(K,M);
        # # f1=[phi(2:K,:); zeros(1,M)]+[zeros(1,M); phi(1:K-1,:)];
        # # f2=[zeros(1,M); phi(K:-1:2,:)]+[phi(K+1:2*K-1,:); zeros(1,M)];
        # # for k=2:K
        # #     ck=G(:,k-1:K:MK);
        # #     G(:,k:K:MK)=[ck(2:K,:); zeros(1,M)]+[zeros(1,M);  ck(1:K-1,:)]...
        # #           -clast-(ok*sum(f1.*ck)).*x1-(ok*sum(f2.*ck)).*x2-(ok*ck(1,:)).*x3...
        # #           -(ok*ck(K,:)).*x4;
        # #     clast=ck;
        # # end 

        # # end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   of THinv5


    def armodel(self, R, rmax):
        """
        function [AR,sigmy]=armodel(R,rmax)
        %
        % to compute AR coefficients of the sources given covariance functions 
        % but if the zeros have magnitude > rmax, the zeros are pushed back.
        %
        """
        # [M,d]=size(R);
        M,d = R.shape

        # AR = zeros(M,d);
        AR = np.zeros((M,d))
        
        # for id=1:d
        for id in xrange(d):
            # AR(:,id)=[1; -toeplitz(R(1:M-1,id),R(1:M-1,id)')\R(2:M,id)];
            AR[:,id] = np.r_[1,np.linalg.lstsq(-toeplitz(R[:M-1,id],R[:M-1,id].T),
                                               R[1:M,id])[0]]
            # v=roots(AR(:,id)); %%% mimicks the matlab function "polystab"
            v = np.roots(AR[:,id])
            # %    v1(1,id)=max(abs(v));
            #v1[0,id] = np.max(np.abs(v))
            #     vs=0.5*(sign(abs(v)-1)+1);
            vs = 0.5*(np.sign(np.abs(v)-1)+1)
            #     v=(1-vs).*v+vs./conj(v);
            v = (1-vs)*v + vs/np.conj(v)
            #     vmax=max(abs(v));
            vmax = np.max(np.abs(v))
            # %    v2(1,id)=max(abs(v));
            #     if vmax>rmax
            #        v=v*rmax/vmax;
            #     end
            if vmax > rmax:
                v = v*rmax/vmax
            #     AR(:,id)=real(poly(v)'); %%% reconstructs back the covariance function
            AR[:,id] = np.real(np.poly(v).T)
        # end 
        # Rs=ar2r(AR);
        Rs = self.ar2r(AR)
        # sigmy=R(1,:)./Rs(1,:);
        sigmy = R[0,:]/Rs[0,:]
        # % [v1; v2]
        # end %%%%%%%%%%%%%%%%%%%%%%%  of armodel
        return AR,sigmy

    def ar2r(self, a):
        """
        function [ r ] = ar2r( a )
        %%%%%
        %%%%% Computes covariance function of AR processes from 
        %%%%% the autoregressive coefficients using an inverse Schur algorithm 
        %%%%% and an inverse Levinson algorithm (for one column it is equivalent to  
        %%%%%      "rlevinson.m" in matlab)
        %
        """
        #   if (size(a,1)==1)
        #       a=a'; % chci to jako sloupce
        #   end
        if a.shape[0] == 1:
            a = a.T
            
        #   [p m] = size(a);    % pocet vektoru koef.AR modelu
        p,m = a.shape
        #   alfa = a;
        alfa = a.copy()
        #   K=zeros(p,m);
        K = np.zeros((p,m))
        #   p = p-1;
        p -= 1
        #   for n=p:-1:1
        #       K(n,:) = -a(n+1,:);
        #       for k=1:n-1
        #           alfa(k+1,:) = (a(k+1,:)+K(n,:).*a(n-k+1,:))./(1-K(n,:).^2);
        #       end
        #       a=alfa;
        #   end
        # XXX Check here if broken
        for n in range(p)[::-1]: #range(p-1,-1,-1):
            K[n,:] = -a[n+1,:]
            for k in xrange(n):
                alfa[k+1,:] = (a[k+1,:]+K[n,:]*a[n-k,:])/(1-K[n,:]**2)
            a = alfa.copy()
        # %  
        #   r = zeros(p+1,m);
        r = np.zeros((p+1,m))
        #   r(1,:) = 1./prod(1-K.^2);
        r[0,:] = 1/np.prod(1-K**2,0)
        #   f = r;
        f = r.copy()
        #   b=f;
        b = f.copy()
        #   for k=1:p 
        #       for n=k:-1:1
        #           K_n = K(n,:);
        #           f(n,:)=f(n+1,:)+K_n.*b(k-n+1,:);
        #           b(k-n+1,:)=-K_n.*f(n+1,:)+(1-K_n.^2).*b(k-n+1,:);
        #       end
        #       b(k+1,:)=f(1,:);
        #       r(k+1,:) = f(1,:);
        #   end
        # XXX Check here if broken
        for k in xrange(p):
            for n in range(k+1)[::-1]: #range(k-1:-1,-1):
                K_n = K[n,:]
                f[n,:] = f[n+1,:] + K_n*b[k-n,:]
                b[k-n,:] = -K_n*f[n+1,:]+(1-K_n**2)*b[k-n,:]
            b[k+1,:] = f[0,:]
            r[k+1,:] = f[0,:]
        # end %%%%%%%%%%%%%%%%%%%%%%%%%%%  of ar2r
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        return r

    def corr_est(self,x,T,q):
        """
        # function R_est=corr_est(x,T,q)
        # %
        """
        # NumOfSources = size(x,1);
        NumOfSources = x.shape[0]
        # R_est = zeros(NumOfSources,(q+1)*NumOfSources);
        R_est = np.zeros((NumOfSources,(q+1)*NumOfSources))
        # for index=1:q+1
        #     R_est(:,NumOfSources*(index-1) + (1:NumOfSources)) = 1/T*(x(:,1:T)*x(:,index:T+index-1)');
        # end
        for index in xrange(q+1):
            #irange = NumOfSources*(index) + np.arange(NumOfSources)
            i = NumOfSources*(index)
            R_est[:,i:i+NumOfSources] = (1/np.float(T))*(np.dot(x[:,:T],x[:,index:T+index].T))

        return R_est

    def weights(self,Ms,rmax,eps0):
        """
        function [H ARC]=weights(Ms,rmax,eps0)
        %
        """
        # [d,Ld]=size(Ms);
        d,Ld = Ms.shape
        # L=floor(Ld/d);
        L = np.int32(np.floor(Ld/np.float(d)))
        # d2=d*(d-1)/2;
        d2 = np.int32(d*(d-1)/2.)
        # R=zeros(L,d);
        R = np.zeros((L,d))
        # for index=1:L
        #     id=(index-1)*d;
        #     R(index,:)=diag(Ms(:,id+1:id+d)).';  %%% columns of R will contain 
        #                            %%% covariance function of the separated components
        # end
        for index in xrange(L):
            id = index*d
            R[index,:] = np.diag(Ms[:,id:id+d])
        # %
        # [ARC,sigmy]=armodel(R,rmax);      %%% compute AR models of estimated components
        ARC,sigmy = self.armodel(R,rmax)
        # %
        # AR3=zeros(2*L-1,d2);
        AR3 = np.zeros((2*L-1,d2))
        # ll = 1;
        # for i=2:d
        #   for k=1:i-1
        #       AR3(:,ll) = conv(ARC(:,i),ARC(:,k));
        #       ll = ll+1;
        # %    AR3=[AR3 conv(AR(:,i),AR(:,k))];
        #   end  
        # end
        ll = 0
        for i in xrange(1,d):
            for k in xrange(i):
                AR3[:,ll] = np.convolve(ARC[:,i],ARC[:,k])
                ll += 1
        # phi=ar2r(AR3);     %%%%%%%%%% functions phi to evaluate CVinv
        phi = self.ar2r(AR3)
        # H=THinv5(phi,L,d2,eps0*phi(1,:));  %%%% to compute inversions of CV 
        #                                        %%%% It has dimension zeros(M,M*d2).
        H = self.THinv5(phi,L,d2,eps0*phi[0,:])
        # im=1; 
        # for i=2:d
        #   for k=1:i-1
        #      fact=1/(sigmy(1,i)*sigmy(1,k));
        #      imm=(im-1)*L;
        #      H(:,imm+1:imm+L)=H(:,imm+1:imm+L)*fact;
        #      im=im+1;
        #   end
        # end
        im = 0
        for i in xrange(1,d):
            for k in xrange(i):
                fact = 1/(sigmy[i]*sigmy[k])
                imm = im*L
                H[:,imm:imm+L] = H[:,imm:imm+L]*fact
                im+=1

        # end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% of weights
        return (H,ARC)

    def CRLB4(self,ARC):
        """
        function ISR = CRLB4(ARC)
        %
        % CRLB4(ARC) generates the CRLB for gain matrix elements (in term 
        % of ISR) for blind separation of K Gaussian autoregressive sources 
        % whose AR coefficients (of the length M, where M-1 is the AR order)
        % are stored as columns in matrix ARC.
        """
        # [M K]=size(ARC);
        M,K = ARC.shape
        
        # Rs=ar2r(ARC);
        Rs = self.ar2r(ARC)
        
        # sum_Rs_s=zeros(K,K);
        sum_Rs_s = np.zeros((K,K))
        
        # for s=0:M-1
        #     for t=0:M-1
        #         sum_Rs_s=sum_Rs_s+(ARC(s+1,:).*ARC(t+1,:))'*Rs(abs(s-t)+1,:);
        #     end
        # end
        for s in xrange(M):
            for t in xrange(M):
                sum_Rs_s += np.dot((ARC[s,:]*ARC[t,:])[np.newaxis,:].T,
                                  Rs[np.abs(s-t),:][np.newaxis,:])

        # denom=sum_Rs_s'.*sum_Rs_s+eye(K)-1;
        denom = sum_Rs_s.T*sum_Rs_s+np.eye(K)-1
        # ISR=sum_Rs_s'./denom.*(ones(K,1)*Rs(1,:))./(Rs(1,:)'*ones(1,K));
        ISR = sum_Rs_s.T/denom*np.outer(np.ones((K,1)),Rs[0,:])/np.outer(Rs[0,:].T,np.ones((1,K)))
        # ISR(eye(K)==1)=0;
        ISR[np.eye(K)==1] = 0
        # end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% of CRLB4
        return ISR

    def uwajd(self,M,maxnumiter=20,W_est0=None):
        """
        function [W_est Ms]=uwajd(M,maxnumiter,W_est0)
        %
        % my approximate joint diagonalization with uniform weights
        %
        % Input: M .... the matrices to be diagonalized, stored as [M1 M2 ... ML]
        %        West0 ... initial estimate of the demixing matrix, if available
        % 
        % Output: W_est .... estimated demixing matrix
        %                    such that W_est * M_k * W_est' are roughly diagonal
        %         Ms .... diagonalized matrices composed of W_est*M_k*W_est'
        %         crit ... stores values of the diagonalization criterion at each 
        %                  iteration
        %
        """
        # [d Md]=size(M);
        d,Md = M.shape
        # L=floor(Md/d);
        L = np.int32(np.floor(Md/np.float(d)))
        # Md=L*d;
        Md = L*d
        # iter=0;
        iter = 0
        # eps=1e-7;
        eps = 1e-7
        # improve=10;
        improve = 10
        # if nargin<3
        #    [H E]=eig(M(:,1:d));
        #    W_est=diag(1./sqrt(diag(E)))*H';
        # else
        #    W_est=W_est0;
        # end
        if W_est0 is None:
            E,H = np.linalg.eig(M[:,:d])
            H = np.real_if_close(H, tol=100)
            W_est = np.dot(np.diag(1./np.sqrt(E)),H.T)
        else:
            W_est = W_est0

        # if nargin<2
        #    maxnumiter=20;
        # end   
        # Ms=M;
        Ms = M.copy()
        # Rs=zeros(d,L);
        Rs = np.zeros((d,L))
        # for k=1:L
        #       ini=(k-1)*d;
        #       M(:,ini+1:ini+d)=0.5*(M(:,ini+1:ini+d)+M(:,ini+1:ini+d)');
        #       Ms(:,ini+1:ini+d)=W_est*M(:,ini+1:ini+d)*W_est';
        #       Rs(:,k)=diag(Ms(:,ini+1:ini+d));
        # end
        for k in xrange(L):
            ini = k*d
            M[:,ini:ini+d] = 0.5*(M[:,ini:ini+d]+M[:,ini:ini+d].T)
            Ms[:,ini:ini+d] = np.dot(np.dot(W_est,M[:,ini:ini+d]),W_est.T)
            Rs[:,k] = np.diag(Ms[:,ini:ini+d])

        # crit=sum(Ms(:).^2)-sum(Rs(:).^2);
        crit = (Ms**2).sum() - (Rs**2).sum()

        # while improve>eps && iter<maxnumiter
        while improve > eps and iter<maxnumiter:
            #   b11=[]; b12=[]; b22=[]; c1=[]; c2=[];
            b11=[]; b12=[]; b22=[]; c1=[]; c2=[];
            #   for id=2:d 
            #     Yim=Ms(1:id-1,id:d:Md);
            #     b22=[b22; sum(Rs(id,:).^2)*ones(id-1,1)];
            #     b12=[b12; (Rs(id,:)*Rs(1:id-1,:)')'];
            #     b11=[b11; sum(Rs(1:id-1,:).^2,2)];
            #     c2=[c2; (Rs(id,:)*Yim')'];
            #     c1=[c1; sum(Rs(1:id-1,:).*Yim,2)];
            #   end
            for id in xrange(1,d):
                Yim = Ms[0:id,id:Md:d]
                b22.append(np.dot((Rs[id,:]**2).sum(0),np.ones((id,1))))
                b12.append(np.dot(Rs[id,:],Rs[:id,:].T).T)
                b11.append((Rs[:id,:]**2).sum(1))
                c2.append(np.dot(Rs[id,:],Yim.T).T)
                c1.append((Rs[:id,:]*Yim).sum(1))
            b22 = np.squeeze(np.vstack(b22))
            b12 = np.hstack(b12)
            b11 = np.hstack(b11)
            c2 = np.hstack(c2)
            c1 = np.hstack(c1)
            #   det0=b11.*b22-b12.^2;
            det0 = b11*b22-b12**2
            #   d1=(c1.*b22-b12.*c2)./det0;
            d1 = (c1*b22-b12*c2)/det0
            #   d2=(b11.*c2-b12.*c1)./det0;
            d2 = (b11*c2-b12*c1)/det0
            # %    value=norm([d1; d2])
            #   m=0;
            m=0
            #   A0=eye(d);
            A0 = np.eye(d)
            #   for id=2:d
            #       A0(id,1:id-1)=d1(m+1:m+id-1,1)';
            #       A0(1:id-1,id)=d2(m+1:m+id-1,1);
            #       m=m+id-1;
            #   end
            for id in xrange(1,d):
                A0[id,0:id] = d1[m:m+id]
                A0[0:id,id] = d2[m:m+id]
                m += id

            #   Ainv=inv(A0);
            Ainv = np.linalg.inv(A0)
            #   W_est=Ainv*W_est;
            W_est = np.dot(Ainv,W_est)
            #   Raux=W_est*M(:,1:d)*W_est';
            Raux = np.dot(np.dot(W_est,M[:,:d]),W_est.T)
            #   aux=1./sqrt(diag(Raux));
            aux = 1/np.sqrt(np.diag(Raux))
            #   W_est=diag(aux)*W_est;  % normalize the result
            W_est = np.dot(np.diag(aux),W_est)
            #   for k=1:L
            #      ini=(k-1)*d;
            #      Ms(:,ini+1:ini+d) = W_est*M(:,ini+1:ini+d)*W_est';
            #      Rs(:,k)=diag(Ms(:,ini+1:ini+d));
            #   end
            for k in xrange(L):
                ini = k*d
                Ms[:,ini:ini+d] = np.dot(np.dot(W_est,M[:,ini:ini+d]),W_est.T)
                Rs[:,k] = np.diag(Ms[:,ini:ini+d])
            #   critic=sum(Ms(:).^2)-sum(Rs(:).^2);
            critic = (Ms**2).sum() - (Rs**2).sum()
            # %   improve=abs(critic-crit(end));
            # %   crit=[crit critic];
            #   improve=abs(critic-crit);
            improve = np.abs(critic - crit)
            #   crit = critic;
            crit = critic
            #   iter=iter+1;
            iter += 1
        # end  %%%%%% of while
        # end %%%%%%%%%%%%%%%%%%%  of uwajd

        return W_est,Ms

    def wajd(self,M,H,W_est0=None,maxnumit=100):
        """
        function [W_est Ms]=wajd(M,H,W_est0,maxnumit)
        %
        % my approximate joint diagonalization with non-uniform weights
        %
        % Input: M .... the matrices to be diagonalized, stored as [M1 M2 ... ML]
        %        H .... diagonal blocks of the weight matrix stored similarly
        %                     as M, but there is dd2 blocks, each of the size L x L
        %        West0 ... initial estimate of the demixing matrix, if available
        %        maxnumit ... maximum number of iterations
        % 
        % Output: W_est .... estimated demixing matrix
        %                    such that W_est * M_k * W_est' are roughly diagonal
        %         Ms .... diagonalized matrices composed of W_est*M_k*W_est'
        %         crit ... stores values of the diagonalization criterion at each 
        %                  iteration
        %
        %
        """
        # [d Md]=size(M);
        d,Md = M.shape
        # L=floor(Md/d);
        L = np.int32(np.floor(Md/np.float(d)))
        # dd2=d*(d-1)/2;
        dd2 = np.int32(d*(d-1)/2.);
        # Md=L*d;
        Md = L*d
        # if nargin<4
        #    maxnumit=100;
        # end   
        # if nargin<3
        #    [H E]=eig(M(:,1:d));
        #    W_est=diag(1./sqrt(diag(E)))*H';
        # else
        #    W_est=W_est0;
        # end
        if W_est0 is None:
            E,H = np.linalg.eig(M[:,:d])
            H = np.real_if_close(H, tol=100)
            W_est = np.dot(np.diag(1/np.sqrt(E)),H.T)
        else:
            W_est = W_est0

        # Ms=M;
        Ms = M.copy()
        # Rs=zeros(d,L);
        Rs = np.zeros((d,L))
        # for k=1:L
        #       ini=(k-1)*d;
        #       M(:,ini+1:ini+d)=0.5*(M(:,ini+1:ini+d)+M(:,ini+1:ini+d)');
        #       Ms(:,ini+1:ini+d)=W_est*M(:,ini+1:ini+d)*W_est';
        #       Rs(:,k)=diag(Ms(:,ini+1:ini+d));
        # end 
        for k in xrange(L):
            ini = k*d
            M[:,ini:ini+d] = 0.5*(M[:,ini:ini+d]+M[:,ini:ini+d].T)
            Ms[:,ini:ini+d] = np.dot(np.dot(W_est,M[:,ini:ini+d]),W_est.T)
            Rs[:,k] = np.diag(Ms[:,ini:ini+d])

        # for iter=1:maxnumit
        for iter in xrange(maxnumit):
            #  b11=zeros(dd2,1); b12=b11; b22=b11; c1=b11; c2=c1;
            b11 = np.zeros((dd2,1))
            b12 = np.zeros((dd2,1))
            b22 = np.zeros((dd2,1))
            c1 = np.zeros((dd2,1))
            c2 = np.zeros((dd2,1))
            #  m=0;
            m=0
            #  for id=2:d        
            #     for id2=1:id-1
            #         m=m+1; im=(m-1)*L;
            #         Wm=H(:,im+1:im+L);
            #         Yim=Ms(id,id2:d:Md);
            #         Rs_id = Rs(id,:);
            #         Rs_id2 = Rs(id2,:);
            #         Wlam1=Wm*Rs_id';
            #         Wlam2=Wm*Rs_id2';
            #         b11(m)=Rs_id2*Wlam2;
            #         b12(m)=Rs_id*Wlam2;
            #         b22(m)=Rs_id*Wlam1;
            #         c1(m)=Wlam2'*Yim';
            #         c2(m)=Wlam1'*Yim';
            #      end
            #   end
            for id in xrange(1,d):
                for id2 in xrange(id):
                    im = m*L
                    Wm = H[:,im:im+L]
                    Yim = Ms[id,id2:Md:d]
                    Rs_id = Rs[id,:]
                    Rs_id2 = Rs[id2,:]
                    Wlam1 = np.dot(Wm,Rs_id.T)
                    Wlam2 = np.dot(Wm,Rs_id2.T)
                    b11[m] = np.dot(Rs_id2,Wlam2)
                    b12[m] = np.dot(Rs_id,Wlam2)
                    b22[m] = np.dot(Rs_id,Wlam1)
                    c1[m] = np.dot(Wlam2.T,Yim.T)
                    c2[m] = np.dot(Wlam1.T,Yim.T)
                    m+=1
            #   det0=b11.*b22-b12.^2;
            det0 = b11*b22-b12**2
            #   d1=(c1.*b22-b12.*c2)./det0;
            d1 = (c1*b22-b12*c2)/det0
            #   d2=(b11.*c2-b12.*c1)./det0;
            d2 = (b11*c2-b12*c1)/det0
            #   m=0;
            m=0
            #   A0=eye(d);
            A0 = np.eye(d)
            #   for id=2:d
            #       A0(id,1:id-1)=d1(m+1:m+id-1,1)';
            #       A0(1:id-1,id)=d2(m+1:m+id-1,1);
            #       m=m+id-1;
            #   end
            for id in xrange(1,d):
                A0[id,0:id] = d1[m:m+id,0]
                A0[0:id,id] = d2[m:m+id,0]
                m += id
            #   Ainv=inv(A0);
            Ainv = np.linalg.inv(A0)
            #   W_est=Ainv*W_est;
            W_est = np.dot(Ainv,W_est)
            #   Raux=W_est*M(:,1:d)*W_est';
            Raux = np.dot(np.dot(W_est,M[:,:d]),W_est.T)
            #   aux=1./sqrt(diag(Raux));
            aux = 1/np.sqrt(np.diag(Raux))
            #   W_est=diag(aux)*W_est;  % normalize the result
            W_est = np.dot(np.diag(aux),W_est)
            #   for k=1:L
            #      ini=(k-1)*d;
            #      Ms(:,ini+1:ini+d) = W_est*M(:,ini+1:ini+d)*W_est';
            #      Rs(:,k)=diag(Ms(:,ini+1:ini+d));
            #   end
            for k in xrange(L):
                ini = k*d
                Ms[:,ini:ini+d] = np.dot(np.dot(W_est,M[:,ini:ini+d]),W_est.T)
                Rs[:,k] = np.diag(Ms[:,ini:ini+d])
        # end %%%%%%%%%%% of for
        # end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% of wajd
        return W_est,Ms
                                                                     

def iwasobi(data, ar_max=10, rmax=0.99, eps0=5.0e-7):
    """
    """
    return IWASOBI(ar_max=ar_max, rmax=rmax, eps0=eps0)(data)



