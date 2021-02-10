import numpy as np
import mpmath
import torch
import torch.nn as nn

import utils


class vMFLogPartition(torch.autograd.Function):
    
    '''
    Evaluates log C_d(kappa) for vMF density
    Allows autograd wrt kappa
    '''
    
    besseli = np.vectorize(mpmath.besseli)
    log = np.vectorize(mpmath.log)
    nhlog2pi = -0.5 * np.log(2*np.pi)
    
    @staticmethod
    def forward(ctx, *args):
        
        '''
        Args:
            args[0] = d; scalar (> 0)
            args[1] = kappa; (> 0) torch tensor of any shape
            
        Returns:
            logC = log C_d(kappa); torch tensor of the same shape as kappa
        '''
        
        d = args[0]
        kappa = args[1]
        
        s = 0.5*d - 1
        
        # log I_s(kappa)
        mp_kappa = mpmath.mpf(1.0) * kappa.detach().cpu().numpy()
        mp_logI = vMFLogPartition.log( vMFLogPartition.besseli(s, mp_kappa) )
        logI = torch.from_numpy( np.array(mp_logI.tolist(), dtype=float) ).to(kappa)
        
        if (logI!=logI).sum().item() > 0:  # there is nan
            raise ValueError('NaN is detected from the output of log-besseli()')
        
        logC = d * vMFLogPartition.nhlog2pi + s * kappa.log() - logI
        
        # save for backard()
        ctx.s, ctx.mp_kappa, ctx.logI = s, mp_kappa, logI
        
        return logC
        
    @staticmethod
    def backward(ctx, *grad_output):
        
        s, mp_kappa, logI = ctx.s, ctx.mp_kappa, ctx.logI 
    
        # log I_{s+1}(kappa)
        mp_logI2 = vMFLogPartition.log( vMFLogPartition.besseli(s+1, mp_kappa) )
        logI2 = torch.from_numpy( np.array(mp_logI2.tolist(), dtype=float) ).to(logI)
        
        if (logI2!=logI2).sum().item() > 0:  # there is nan
            raise ValueError('NaN is detected from the output of log-besseli()')
        
        dlogC_dkappa = -(logI2 - logI).exp()
        
        return None, grad_output[0] * dlogC_dkappa
        

class vMF(nn.Module):
    
    '''
    vMF(x; mu, kappa)
    '''
    
    def __init__(self, x_dim, reg=1e-6):
        
        super(vMF, self).__init__()
        
        self.x_dim = x_dim
        
        self.mu_unnorm = nn.Parameter(torch.randn(x_dim))
        self.logkappa = nn.Parameter(0.01*torch.randn([]))
        
        self.reg = reg
        
    def set_params(self, mu, kappa):
        
        with torch.no_grad():
            self.mu_unnorm.copy_(mu)
            self.logkappa.copy_(torch.log(kappa+utils.realmin))
    
    def get_params(self):
        
        mu = self.mu_unnorm / utils.norm(self.mu_unnorm)
        kappa = self.logkappa.exp() + self.reg
        
        return mu, kappa
        
    def forward(self, x, utc=False):
        
        '''
        Evaluate logliks, log p(x)
        
        Args:
            x = batch for x
            utc = whether to evaluate only up to constant or exactly 
                if True, no log-partition computed
                if False, exact loglik computed

        Returns:
            logliks = log p(x)
        '''

        mu, kappa = self.get_params()

        dotp = (mu.unsqueeze(0) * x).sum(1)
        
        if utc:
            logliks = kappa * dotp
        else:
            logC = vMFLogPartition.apply(self.x_dim, kappa)
            logliks = kappa * dotp + logC
        
        return logliks
    
    def sample(self, N=1, rsf=10):
    
        '''
        Args:
            N = number of samples to generate
            rsf = multiplicative factor for extra backup samples in rejection sampling 
        
        Returns:
            samples; N samples generated
            
        Notes:
            no autodiff
        '''
        
        d = self.x_dim
        
        with torch.no_grad():
            
            mu, kappa = self.get_params()
        
            # Step-1: Sample uniform unit vectors in R^{d-1} 
            v = torch.randn(N, d-1).to(mu)
            v = v / utils.norm(v, dim=1)
            
            # Step-2: Sample v0
            kmr = np.sqrt( 4*kappa.item()**2 + (d-1)**2 )
            bb = (kmr - 2*kappa) / (d-1)
            aa = (kmr + 2*kappa + d - 1) / 4
            dd = (4*aa*bb)/(1+bb) - (d-1)*np.log(d-1)
            beta = torch.distributions.Beta( torch.tensor(0.5*(d-1)), torch.tensor(0.5*(d-1)) )
            uniform = torch.distributions.Uniform(0.0, 1.0)
            v0 = torch.tensor([]).to(mu)
            while len(v0) < N:
                eps = beta.sample([1, rsf*(N-len(v0))]).squeeze().to(mu)
                uns = uniform.sample([1, rsf*(N-len(v0))]).squeeze().to(mu)
                w0 = (1 - (1+bb)*eps) / (1 - (1-bb)*eps)
                t0 = (2*aa*bb) / (1 - (1-bb)*eps)
                det = (d-1)*t0.log() - t0 + dd - uns.log()
                v0 = torch.cat([v0, torch.tensor(w0[det>=0]).to(mu)])
                if len(v0) > N:
                    v0 = v0[:N]
                    break
            v0 = v0.reshape([N,1])
            
            # Step-3: Form x = [v0; sqrt(1-v0^2)*v]
            samples = torch.cat([v0, (1-v0**2).sqrt()*v], 1)
    
            # Setup-4: Householder transformation
            e1mu = torch.zeros(d,1).to(mu);  e1mu[0,0] = 1.0
            e1mu = e1mu - mu if len(mu.shape)==2 else e1mu - mu.unsqueeze(1)
            e1mu = e1mu / utils.norm(e1mu, dim=0)
            samples = samples - 2 * (samples @ e1mu) @ e1mu.t()
    
        return samples


class MixvMF(nn.Module):
    
    '''
    MixvMF(x) = \sum_{m=1}^M \alpha_m vMF(x; mu_m, kappa_m)
    '''

    def __init__(self, x_dim, order, reg=1e-6):
        
        super(MixvMF, self).__init__()
        
        self.x_dim = x_dim
        self.order = order
        self.reg = reg
        
        self.alpha_logit = nn.Parameter(0.01*torch.randn(order))
        self.comps = nn.ModuleList(
            [ vMF(x_dim, reg) for _ in range(order) ]
        )
        
    def set_params(self, alpha, mus, kappas):
        
        with torch.no_grad():
            self.alpha_logit.copy_(torch.log(alpha+utils.realmin))
            for m in range(self.order):
                self.comps[m].mu_unnorm.copy_(mus[m])
                self.comps[m].logkappa.copy_(torch.log(kappas[m]+utils.realmin))
    
    def get_params(self):
        
        logalpha = self.alpha_logit.log_softmax(0)
        
        mus, kappas = [], []
        for m in range(self.order):
            mu, kappa = self.comps[m].get_params()
            mus.append(mu)
            kappas.append(kappa)
        
        mus = torch.stack(mus, axis=0)
        kappas = torch.stack(kappas, axis=0)
        
        return logalpha, mus, kappas
        
    def forward(self, x):
        
        '''
        Evaluate logliks, log p(x)
        
        Args:
            x = batch for x
            
        Returns:
            logliks = log p(x)
            logpcs = log p(x|c=m)
        '''
        
        logalpha = self.alpha_logit.log_softmax(0)
        
        logpcs = []
        for m in range(self.order):
            logpcs.append(self.comps[m](x))
        logpcs = torch.stack(logpcs, dim=1)
        
        logliks = (logalpha.unsqueeze(0) + logpcs).logsumexp(1)
        
        return logliks, logpcs
    
    def sample(self, N=1, rsf=10):
    
        '''
        Args:
            N = number of samples to generate
            rsf = multiplicative factor for extra backup samples in rejection sampling 
                  (used in sampling from vMF)
        
        Returns:
            samples = N samples generated
            cids = which components the samples come from; N-dim {0,1,...,M-1}-valued
            
        Notes:
            no autodiff
        '''
        
        with torch.no_grad():
            
            alpha = self.alpha_logit.log_softmax(0).exp()
            
            cids = torch.multinomial(alpha, N, replacement=True)
            
            samples = torch.zeros(N, self.x_dim)
            for c in range(self.order):
                Nc = (cids==c).sum()
                if Nc > 0:
                    samples[cids==c,:] = self.comps[c].sample(N=Nc, rsf=rsf)
                    
        return samples, cids

