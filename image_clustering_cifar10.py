import os
import random
import numpy as np
import sklearn.metrics
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import models
import utils


'''
Set options
'''

opts = {}

# path to data and result files
opts['data_path'] = os.path.join('data', 'CIFAR10')  # will donwload data if not exist
utils.mkdirs(opts['data_path'])
opts['result_path'] = os.path.join('result', 'CIFAR10') 
utils.mkdirs(opts['result_path'])

# options for autoencoder
opts['cuda'] = True
opts['z_dim'] = 100
opts['batch_size'] = 128
opts['ae_hid_dim'] = 512
opts['ae_max_epoch'] = 100
opts['ae_lr'] = 1e-4

# options for EM
opts_em = {}
opts_em['max_iters'] = 100  # maximum number of EM iterations
opts_em['rll_tol'] = 1e-5  # tolerance of relative loglik improvement
opts_em['batch_size'] = 128  # batch inside E and M steps

# options for SGD
opts_sgd = {}
opts_sgd['batch_size'] = 256  # batch size
opts_sgd['max_epochs'] = 100  # maximum number of epochs


class ConvEncoder(nn.Module):
    
    def __init__(self, z_dim, h_dim=256):
        
        super().__init__()
        
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.fc4 = nn.Linear(64*4*4, h_dim)
        self.fc5 = nn.Linear(h_dim, z_dim)
        
    def forward(self, x):
        
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.fc4(out))
        out = self.fc5(out)
        
        out = out / utils.norm(out, dim=1)
        
        return out

        
class ConvDecoder(nn.Module):

    def __init__(self, z_dim, h_dim=256):
        
        super().__init__()
        
        self.z_dim = z_dim
        
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 4*4*64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(32, 3, 4, 2, 1)
        
    def forward(self, z):
            
        out = F.relu(self.fc1(z))
        out = F.relu(self.fc2(out))
        out = out.view(out.size(0), 64, 4, 4)
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv4(out))
        out = self.deconv5(out)
        
        return out


'''
Train autoencoder: image -> z
'''

class Autoencoder(nn.Module):

    def __init__(self, z_dim, hid_dim):
        
        super().__init__()
        
        self.z_dim = z_dim
        
        self.encoder = ConvEncoder(z_dim, hid_dim)
        self.decoder = ConvDecoder(z_dim, hid_dim)

    def forward(self, x):
        
        x2 = self.decoder(self.encoder(x))
        return ((x2-x)**2).sum() / x.shape[0]

    def encode(self, x):
        
        return self.encoder(x)


ae = Autoencoder(z_dim=opts['z_dim'], hid_dim=opts['ae_hid_dim'])
if opts['cuda']:
    ae = ae.cuda()

# load data (automatically download if not exist)
dset_tr = datasets.CIFAR10(opts['data_path'], train=True, download=True, transform=transforms.ToTensor())
dset_te = datasets.CIFAR10(opts['data_path'], train=False, download=True, transform=transforms.ToTensor())
dl_tr = DataLoader(dset_tr, batch_size=opts['batch_size'], shuffle=True, drop_last=True)
dl_te = DataLoader(dset_te, batch_size=opts['batch_size'], shuffle=False, drop_last=False)

# optimizer
optimizer = optim.Adam(ae.parameters(), lr=opts['ae_lr'])
       
for epoch in range(opts['ae_max_epoch']):

    train_loss = 0
    
    for batch_idx, (XX, YY) in enumerate(dl_tr):
    
        if opts['cuda']:
            XX = XX.cuda()
    
        optimizer.zero_grad()
        loss = ae(XX)
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item() * XX.shape[0]
        
        if batch_idx % 20 == 0:
            prn_str = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*XX.shape[0], len(dl_tr.dataset),
                100.*batch_idx/len(dl_tr), loss.item() )
            print(prn_str)
            
    epoch_loss = train_loss / len(dl_tr.dataset)
    prn_str = '====> Epoch: {} Average loss: {:.4f}'.format(epoch, epoch_loss)
    print(prn_str)

# save the learned autoencoder
torch.save(ae.state_dict(), os.path.join(opts['result_path'], 'ae_cifar10.pth'))


# load autoencoder (if necessary)
ae = Autoencoder(z_dim=opts['z_dim'], hid_dim=opts['ae_hid_dim'])
ae.load_state_dict( torch.load(os.path.join(opts['result_path'], 'ae_cifar10.pth')) )
if opts['cuda']:
    ae = ae.cuda()

# embed all images (and their labels)
dl_tr = DataLoader(dset_tr, batch_size=opts['batch_size'], shuffle=False, drop_last=False)
dl_te = DataLoader(dset_te, batch_size=opts['batch_size'], shuffle=False, drop_last=False)
with torch.no_grad():
    for split in ['tr', 'te']:
        dl = dl_tr if split=='tr' else dl_te
        for ii, (xx, yy) in enumerate(dl):
            if opts['cuda']:
                xx = xx.cuda()
            zz = ae.encoder(xx).cpu()
            if ii==0:
                ZZ, YY = zz, yy
            else:
                ZZ, YY = torch.cat([ZZ, zz], dim=0), torch.cat([YY, yy], dim=0)
        torch.save([ZZ, YY], os.path.join(opts['result_path'], 'embeds_%s_cifar10.pth' % split))


'''
Do mixture learning in the embedded space
'''
    
seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    
# load embedded data and labels
Ztr, Ytr = torch.load(os.path.join(opts['result_path'], 'embeds_tr_cifar10.pth'))
Zte, Yte = torch.load(os.path.join(opts['result_path'], 'embeds_te_cifar10.pth'))


'''
EM (done on CPU)
'''

# randomly initialized mixture
mix = models.MixvMF(x_dim=opts['z_dim'], order=10)

# data loader for training samples
dataloader = DataLoader(Ztr, batch_size=opts_em['batch_size'], shuffle=False, drop_last=False)

# EM learning
ll_old = -np.inf
with torch.no_grad():
    
    for steps in range(opts_em['max_iters']):
        
        # E-step
        logalpha, mus, kappas = mix.get_params()
        logliks, logpcs = mix(Ztr)
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
            if rll < opts_em['rll_tol']:
                prn_str = 'Stop EM since the relative improvement '
                prn_str += '(%.6f) < tolerance (%.6f)\n' % (rll.item(), opts_em['rll_tol'])
                print(prn_str)
                break
            
        ll_old = ll
                
        # M-step
        qzx = ( qz.unsqueeze(2) * Ztr.unsqueeze(1) ).sum(0)
        qzx_norms = utils.norm(qzx, dim=1)
        mus_new = qzx / qzx_norms
        Rs = qzx_norms[:,0] / (qz.sum(0) + utils.realmin)
        kappas_new = (mix.x_dim*Rs - Rs**3) / (1 - Rs**2)
        alpha_new = qz.sum(0) / Ztr.shape[0]
        
        # assign new params
        mix.set_params(alpha_new, mus_new, kappas_new)
        
    # save model
    mix_em = mix
    torch.save(mix_em.state_dict(), os.path.join(opts['result_path'], 'mix_em_cifar10.pth'))
        
    logliks, logpcs = mix_em(Ztr)
    ll = logliks.sum()
    prn_str = '[Training done] loglik = %.4f\n' % ll.item()
    print(prn_str)
    
# cluster label predictions
with torch.no_grad():
    logalpha, mus, kappas = mix_em.get_params()
    clabs_em = ( logalpha.unsqueeze(0) + logpcs ).max(1)[1]
    logliks_te, logpcs_te = mix_em(Zte)
    clabs_te_em = ( logalpha.unsqueeze(0) + logpcs_te ).max(1)[1]

# clustering metrics
metrics_em = {}
metrics_em['ARI'] = sklearn.metrics.adjusted_rand_score(Ytr, clabs_em)
metrics_em['ARI_te'] = sklearn.metrics.adjusted_rand_score(Yte, clabs_te_em)
metrics_em['NMI'] = sklearn.metrics.normalized_mutual_info_score(Ytr, clabs_em)
metrics_em['NMI_te'] = sklearn.metrics.normalized_mutual_info_score(Yte, clabs_te_em)

prn_str = '== EM estimator ==\n'
prn_str += 'Test: ARI = %.4f, NMI = %.4f\n' % (metrics_em['ARI_te'], metrics_em['NMI_te'])
prn_str += 'Train: ARI = %.4f, NMI = %.4f\n' % (metrics_em['ARI'], metrics_em['NMI'])
print(prn_str)


'''
SGD
'''
 
# data loader for training samples
dataloader = DataLoader(Ztr, batch_size=opts_sgd['batch_size'], shuffle=False, drop_last=False)
    
# create a model
mix = models.MixvMF(x_dim=opts['z_dim'], order=10)
mix = mix.cuda()
    
# create optimizers and set optim params
params = list(mix.parameters())
optim = torch.optim.Adam(params, lr=1e-1, betas=[0.9, 0.99])
lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.95)
    
# SGD training
for epoch in range(opts_sgd['max_epochs']):

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

# save model
mix_sgd = mix
mix_sgd = mix_sgd.cpu()
torch.save(mix_sgd.state_dict(), os.path.join(opts['result_path'], 'mix_sgd_cifar10.pth'))

# cluster label predictions
with torch.no_grad():
    
    logalpha, mus, kappas = mix_sgd.get_params()

    # prediction on train data    
    for ii, data in enumerate(dataloader):
        logliks_, logpcs_ = mix_sgd(data)
        if ii==0:
            logliks, logpcs = logliks_, logpcs_
        else:
            logliks = torch.cat([logliks, logliks_], dim=0)
            logpcs = torch.cat([logpcs, logpcs_], dim=0)
    clabs_sgd = ( logalpha.unsqueeze(0) + logpcs ).max(1)[1]
    
    # prediction on test data
    dataloader = DataLoader(Zte, batch_size=opts_sgd['batch_size'], shuffle=False, drop_last=False)
    for ii, data in enumerate(dataloader):
        logliks_, logpcs_ = mix_sgd(data)
        if ii==0:
            logliks_te, logpcs_te = logliks_, logpcs_
        else:
            logliks_te = torch.cat([logliks_te, logliks_], dim=0)
            logpcs_te = torch.cat([logpcs_te, logpcs_], dim=0)
    clabs_te_sgd = ( logalpha.unsqueeze(0) + logpcs_te ).max(1)[1]

# clustering metrics
metrics_sgd = {}
metrics_sgd['ARI'] = sklearn.metrics.adjusted_rand_score(Ytr, clabs_sgd)
metrics_sgd['ARI_te'] = sklearn.metrics.adjusted_rand_score(Yte, clabs_te_sgd)
metrics_sgd['NMI'] = sklearn.metrics.normalized_mutual_info_score(Ytr, clabs_sgd)
metrics_sgd['NMI_te'] = sklearn.metrics.normalized_mutual_info_score(Yte, clabs_te_sgd)

prn_str = '== SGD estimator ==\n'
prn_str += 'Test: ARI = %.4f, NMI = %.4f\n' % (metrics_sgd['ARI_te'], metrics_sgd['NMI_te'])
prn_str += 'Train: ARI = %.4f, NMI = %.4f\n' % (metrics_sgd['ARI'], metrics_sgd['NMI'])
print(prn_str)

