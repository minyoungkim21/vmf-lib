import torch
from torch.utils.data import DataLoader

import models
import utils


'''
Define a true vMF model
'''

mu_true = torch.zeros(100)  # 5 or 20
mu_true[0] = 1.0
mu_true = mu_true / utils.norm(mu_true, dim=0)
kappa_true = torch.tensor(500.0)  # 50.0
vmf_true = models.vMF(x_dim=mu_true.shape[0])
vmf_true.set_params(mu=mu_true, kappa=kappa_true)
vmf_true = vmf_true.cuda()

# sample from true vMF model
samples = vmf_true.sample(N=10000, rsf=1)


'''
Full-batch ML estimator
'''

xm = samples.mean(0)
xm_norm = (xm**2).sum().sqrt()
mu0 = xm / xm_norm
kappa0 = (len(xm)*xm_norm - xm_norm**3) / (1-xm_norm**2)

mu_err = ((mu0.cpu() - mu_true)**2).sum().item()  # relative error
kappa_err = (kappa0.cpu() - kappa_true).abs().item() / kappa_true.item()
prn_str = '== Batch ML estimator ==\n'
prn_str += 'mu = %s (error = %.8f)\n' % (mu0.cpu().numpy(), mu_err)
prn_str += 'kappa = %s (error = %.8f)\n' % (kappa0.cpu().numpy(), kappa_err)
print(prn_str)


'''
SGD-based ML estimator 
'''
   
B = 128  # batch size
max_epochs = 100  # maximum number of epochs

dataloader = DataLoader(samples, batch_size=B, shuffle=True, drop_last=True)

# create a model
vmf = models.vMF(x_dim=mu_true.shape[0])
vmf = vmf.cuda()

# create optimizers and set optim params
params = list(vmf.parameters())
optim = torch.optim.Adam(params, lr=1e-2, betas=[0.9, 0.99])
lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.95)

# error of the initial model
with torch.no_grad():
    mu, kappa = vmf.get_params()
    mu_err = (mu.cpu() - mu_true).abs().mean().item()  # dim-wise absolute error
    kappa_err = (kappa.cpu() - kappa_true).abs().item()
    prn_str = '== Before training starts ==\n'
    prn_str += 'mu = %s (error = %.6f)\n' % (mu.cpu().numpy(), mu_err)
    prn_str += 'kappa = %s (error = %.6f)\n' % (kappa.cpu().numpy(), kappa_err)
    print(prn_str)

# SGD training
for epoch in range(max_epochs):

    obj, nsamps = 0.0, 0
    for ii, data in enumerate(dataloader):
        enll = -vmf(data).mean()
        optim.zero_grad()
        enll.backward()
        optim.step()
        obj += enll.item()*data.shape[0]
        nsamps += data.shape[0]
    obj /= nsamps

    lr_sched.step()

    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            mu, kappa = vmf.get_params()
        mu_err = ((mu.cpu() - mu_true)**2).sum().item()  # relative error
        kappa_err = (kappa.cpu() - kappa_true).abs().item() / kappa_true.item()
        prn_str = '== After epoch %d ==\n' % epoch
        prn_str += 'Expectected negative log-likelihood = %.4f\n' % enll.item()
        prn_str += 'mu = %s (error = %.8f)\n' % (mu.cpu().numpy(), mu_err)
        prn_str += 'kappa = %s (error = %.8f)\n' % (kappa.cpu().numpy(), kappa_err)
        print(prn_str)

