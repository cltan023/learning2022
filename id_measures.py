import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy as np
import glob
from os.path import join
from joblib import delayed, parallel_backend, Parallel

import sys
sys.path.append('..')

#Import python packages
from geomle import geomle, mle
from intrinsic_dimension import id_estimate

#Import R packages
import rpy2
import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import rpy2py, py2rpy
from rpy2.robjects.packages import importr
intdimr = importr('intrinsicDimension')
r_base = importr('base')

import time
def calculate_time(func): 
    def inner_func(*args, **kwargs): 
        begin = time.time() 
        res = func(*args, **kwargs) 
        end = time.time()
        return res, end - begin
    return inner_func

# k1_sigmoid = lambda x: int(round(20 / (1 + np.exp( -0.05 * x))))
# k2_sigmoid = lambda x: int(round(60 / (1 + np.exp( -0.03 * (x + 5)))))

k1_log = lambda x: int(np.round(3 * np.log(1/3 * x) + 10))
k2_log = lambda x: int(np.round(12 * np.log(1/12 * (x + 10)) + 30))

class DimEst():
    def __init__(self):
        self.names = ['ED', 'MLE', 'GeoMLE', 'MIND']
        self.results = {}

    def pre_processing(self, data_py):
        self.data_py = data_py
        self.data_r = py2rpy(data_py.values)
        self.dim = data_py.shape[1]
    
    def estimateByIndex(self, method):
        if method == 'ED':
            val, elapsed_time = self.ed(self.data_py)
        elif method == 'MLE':
            val, elapsed_time = self.mle(self.data_py)
        elif method == 'GeoMLE': # very slow
            val, elapsed_time = self.geomle(self.data_py, self.dim)
        elif method == 'MIND':
            val, elapsed_time = self.mind_mlk(self.data_r, self.dim)
        elif method == 'DANCo': # very slow
            val, elapsed_time = self.danco(self.data_r, self.dim)
        elif method == 'ESS':
            val, elapsed_time = self.ess(self.data_r)
        elif method == 'PCA':
            val, elapsed_time = self.pca(self.data_r)
        
        return {method: (val, elapsed_time)}
    
    @staticmethod
    @calculate_time
    def ed(data):
        data = data.values
        cov = np.cov(data.T, ddof=1)
        _, s, _ = np.linalg.svd(cov)
        return np.sum(s)**2 / np.sum(s**2)
    
    @staticmethod
    @calculate_time
    def mle(data):
        return mle(data, k1=20, k2=55, average=True)[0].mean()
    
    @staticmethod
    @calculate_time
    def geomle(data, dim):
#         k1 =  k1_log(dim)
#         k2 =  k2_log(dim)
        return geomle(data, k1=20, k2=55, nb_iter1=1, nb_iter2=20, alpha=5e-3).mean()
    
    @staticmethod
    @calculate_time
    def mind_mlk(data, dim):
        return intdimr.dancoDimEst(data, k=(20 + 55) // 2, D=dim, ver="MIND_MLk").rx2('dim.est')[0]
    
    @staticmethod
    @calculate_time
    def danco(data, dim):
        return intdimr.dancoDimEst(data, k=(20 + 55) // 2, D=dim, ver="DANCo").rx2('dim.est')[0]
    
    @staticmethod
    @calculate_time
    def ess(data):
        return intdimr.essLocalDimEst(data).rx2('dim.est')[0]
    
    @staticmethod
    @calculate_time
    def pca(data):
        return intdimr.pcaLocalDimEst(data, 'FO').rx2('dim.est')[0]

from models import resnet18
from data import get_dataloader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = resnet18(num_classes=10).to(device)
train_loader, train_loader_eval, test_loader = get_dataloader(data_name='cifar10', data_dir='../../data', num_of_per_class=5000, batch_size_train=512, batch_size_eval=10000, num_workers=4, noise_y=0.0)

runs = []
for seed in [1, 2, 3]:
    runs.append(f'resnet18_cifar10_ADAM_5000_lr=0.001_bs=32_seed={seed}')
    runs.append(f'resnet18_cifar10_ADAM_5000_lr=1e-05_bs=512_seed={seed}')
    runs.append(f'resnet18_cifar10_SGD0.0_5000_lr=0.1_bs=32_seed={seed}')
    runs.append(f'resnet18_cifar10_SGD0.0_5000_lr=0.001_bs=512_seed={seed}')
    runs.append(f'resnet18_cifar10_SGDM0.9_5000_lr=0.1_bs=32_seed={seed}')
    runs.append(f'resnet18_cifar10_SGDM0.9_5000_lr=0.001_bs=512_seed={seed}')

for run in runs:
    save_dir = join('results', run)
    print(save_dir)    
    model_hist = sorted(glob.glob(join(save_dir, '*epoch*.pt')), key=lambda x: int (x.split('.pt')[0].split('epoch=')[-1]))

    for data, _ in test_loader:
        data = data.to(device)

    nld_hist = []
    for epoch, m in enumerate(model_hist):
        # print(epoch)
        model.load_state_dict(torch.load(m, map_location=device))

        with torch.no_grad():
            _, representation = model(data)
        if representation.is_cuda:
            representation = representation.cpu().numpy()
        representation = pd.DataFrame(representation)

        DE = DimEst()
        DE.names = ['ED', 'MLE']
        DE.pre_processing(representation)

        n_jobs = len(DE.names)
        with parallel_backend('multiprocessing'):
            results = Parallel(n_jobs=n_jobs)(delayed(DE.estimateByIndex)(method) for method in DE.names)
        for res in results:
            DE.results.update(res)
        
        DE.names = ['GeoMLE', 'MIND']
        n_jobs = len(DE.names)
        with parallel_backend('multiprocessing'):
            results = Parallel(n_jobs=n_jobs)(delayed(DE.estimateByIndex)(method) for method in DE.names)
        for res in results:
            DE.results.update(res)

        nld_hist.append(DE.results)
        torch.save(nld_hist, join(save_dir, 'nld_hist.pt'))