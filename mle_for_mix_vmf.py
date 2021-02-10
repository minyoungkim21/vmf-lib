import random
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader

import models
import utils


seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    

'''
Define a true vMF mixture model
'''

mix_true = models.MixvMF(x_dim=5, order=3)

mus_true = [
    torch.tensor([0.3, -1.2, 2.3, 0.4, 2.1]),
    torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
    torch.tensor([-0.3, 1.2, -2.3, -0.4, -2.1]),
]
mus_true = [ mu / utils.norm(mu, dim=0) for mu in mus_true ]
kappas_true = [
    torch.tensor(100.0), 
    torch.tensor(50.0), 
    torch.tensor(100.0)
]
alpha_true = torch.tensor([0.3, 0.4, 0.3])

mix_true.set_params(alpha_true, mus_true, kappas_true)

# sample data from true mixture
samples, cids = mix_true.sample(N=1000, rsf=1)


'''
Full-batch EM learning
'''
    
opts = {}
opts['max_iters'] = 100  # maximum number of EM iterations
opts['rll_tol'] = 1e-5  # tolerance of relative loglik improvement

# randomly initialized mixture
mix = models.MixvMF(x_dim=5, order=3)

# EM learning
ll_old = -np.inf
with torch.no_grad():
    
    for steps in range(opts['max_iters']):
        
        # E-step
        logalpha, mus, kappas = mix.get_params()
        logliks, logpcs = mix(samples)
        ll = logliks.sum()
        jll = logalpha.unsqueeze(0) + logpcs
        qz = jll.log_softmax(1).exp()
        
        if steps==0:
            prn_str = '[Before EM starts] loglik = %.4f\n' % ll.item()
        else:
            prn_str = '[Steps %03d] loglik (before M-step) = %.4f\n' % (steps, ll.item())
        print(prn_str)
        
        # tolerance check
        if steps>0:
            rll = (ll-ll_old).abs() / (ll_old.abs()+utils.realmin)
            if rll < opts['rll_tol']:
                prn_str = 'Stop EM since the relative improvement '
                prn_str += '(%.6f) < tolerance (%.6f)\n' % (rll.item(), opts['rll_tol'])
                print(prn_str)
                break
            
        ll_old = ll
                
        # M-step
        qzx = ( qz.unsqueeze(2) * samples.unsqueeze(1) ).sum(0)
        qzx_norms = utils.norm(qzx, dim=1)
        mus_new = qzx / qzx_norms
        Rs = qzx_norms[:,0] / (qz.sum(0) + utils.realmin)
        kappas_new = (mix.x_dim*Rs - Rs**3) / (1 - Rs**2)
        alpha_new = qz.sum(0) / samples.shape[0]
        
        # assign new params
        mix.set_params(alpha_new, mus_new, kappas_new)
        
    logliks, logpcs = mix(samples)
    ll = logliks.sum()
    prn_str = '[Training done] loglik = %.4f\n' % ll.item()
    print(prn_str)

# find the best matching permutations of components
print('Find the best matching permutations of components')
with torch.no_grad():
    logalpha, mus, kappas = mix.get_params()
    alpha = logalpha.exp()
    perms = list(itertools.permutations(range(mix.order)))
    best_perm, best_error = None, 1e10
    for perm in perms:
        perm = np.array(perm)
        error_alpha = (alpha[perm] - alpha_true).abs().sum()
        error_mus = (mus[perm,:] - torch.stack(mus_true, dim=0)).abs().sum()
        error_kappas = (kappas[perm] - torch.stack(kappas_true, dim=0)).abs().sum()
        error = (error_alpha + error_mus + error_kappas).item()
        print('perm = %s: error = %.4f' % (perm, error))
        if error < best_error:
            best_perm, best_error = perm, error
            print('best perm has changed to: %s' % best_perm)
            
print('For the best components permutation:')
print('----------')
print('alpha_true = %s' % alpha_true)
print('alpha = %s' % alpha[best_perm])
print('error in alpha = %.4f' % (alpha[best_perm] - alpha_true).abs().sum().item())
print('----------')
print('mus_true = %s' % torch.stack(mus_true, dim=0))
print('mus = %s' % mus[best_perm])
print('error in mus = %.4f' % (mus[best_perm,:] - torch.stack(mus_true, dim=0)).abs().sum().item())
print('----------')
print('kappas_true = %s' % torch.stack(kappas_true, dim=0))
print('kappas = %s' % kappas[best_perm])
print('error in kappas = %.4f' % (kappas[best_perm] - torch.stack(kappas_true, dim=0)).abs().sum().item())
print('----------')

# save model
mix_em = mix
    

'''
SGD-based ML estimator 
'''
    
B = 64  # batch size
max_epochs = 100  # maximum number of epochs

dataloader = DataLoader(samples, batch_size=B, shuffle=True, drop_last=True)

# create a model
mix = models.MixvMF(x_dim=5, order=3)
mix = mix.cuda()

# create optimizers and set optim params
params = list(mix.parameters())
optim = torch.optim.Adam(params, lr=1e-1, betas=[0.9, 0.99])
lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.95)

# SGD training
for epoch in range(max_epochs):

    ll, nsamps = 0.0, 0
    for ii, data in enumerate(dataloader):
        data = data.cuda()
        logliks, _ = mix(data)
        anll = -logliks.mean()
        optim.zero_grad()
        anll.backward()
        optim.step()
        ll += -anll.item()*data.shape[0]

    lr_sched.step()

    if (epoch+1) % 1 == 0:
        prn_str = '[Epoch %03d] loglik (accumulated) = %.4f\n' % (epoch, ll)
        print(prn_str)

mix = mix.cpu()

# find the best matching permutations of components
print('Find the best matching permutations of components')
with torch.no_grad():
    logalpha, mus, kappas = mix.get_params()  # logalpha = M-dim
    alpha = logalpha.exp()
    perms = list(itertools.permutations(range(mix.order)))
    best_perm, best_error = None, 1e10
    for perm in perms:
        perm = np.array(perm)
        error_alpha = (alpha[perm] - alpha_true).abs().sum()
        error_mus = (mus[perm,:] - torch.stack(mus_true, dim=0)).abs().sum()
        error_kappas = (kappas[perm] - torch.stack(kappas_true, dim=0)).abs().sum()
        error = (error_alpha + error_mus + error_kappas).item()
        print('perm = %s: error = %.4f' % (perm, error))
        if error < best_error:
            best_perm, best_error = perm, error
            print('best perm has changed to: %s' % best_perm)

print('For the best components permutation:')
print('----------')
print('alpha_true = %s' % alpha_true)
print('alpha = %s' % alpha[best_perm])
print('error in alpha = %.4f' % (alpha[best_perm] - alpha_true).abs().sum().item())
print('----------')
print('mus_true = %s' % torch.stack(mus_true, dim=0))
print('mus = %s' % mus[best_perm])
print('error in mus = %.4f' % (mus[best_perm,:] - torch.stack(mus_true, dim=0)).abs().sum().item())
print('----------')
print('kappas_true = %s' % torch.stack(kappas_true, dim=0))
print('kappas = %s' % kappas[best_perm])
print('error in kappas = %.4f' % (kappas[best_perm] - torch.stack(kappas_true, dim=0)).abs().sum().item())
print('----------')

# save model
mix_sgd = mix

