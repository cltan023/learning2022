from typing import OrderedDict
import matplotlib.pyplot as plt
from sklearn import metrics
import torch
from torch.autograd import Variable
import random
import numpy as np
import glob
import os
from os.path import join
from collections import OrderedDict
from intrinsic_dimension import id_estimate
import wandb

from models import resnet18, vgg16_bn, densenet121
from data import get_dataloader
from utils import IntermediateLayerGetter as MidGetter

# torch.set_num_threads(os.cpu_count())

arch = 'vgg16_bn'
dataset = 'cifar10'
num_per_class = 5000
optimizer = 'SGD'
lr = 0.001
bs = 512
seed = 1
gpu_id = 1

if optimizer == 'SGD':
    instance = f'{arch}_{dataset}_SGD0.0_{num_per_class}/lr={lr}_bs={bs}_seed={seed}'
elif optimizer == 'SGDM':
    instance = f'{arch}_{dataset}_SGDM0.9_{num_per_class}/lr={lr}_bs={bs}_seed={seed}'
elif optimizer == 'ADAM':
    instance = f'{arch}_{dataset}_ADAM_{num_per_class}/lr={lr}_bs={bs}_seed={seed}'
    
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

if arch == 'resnet18': 
    model = resnet18(num_classes=10).to(device)
elif arch == 'vgg16_bn':
    model = vgg16_bn(num_classes=10).to(device)
elif arch == 'densenet121':
    model = densenet121(num_classes=10).to(device)
else:
    raise NotImplementedError

train_loader, train_loader_eval, test_loader = get_dataloader(data_name=dataset, data_dir='../../data', num_of_per_class=num_per_class, batch_size_train=512, batch_size_eval=500, num_workers=4, noise_y=0.0)

wandb_run = wandb.init(project='multi-nlid', name=instance, group=arch)

return_layers = OrderedDict()
num_relu_layer = 0
for _, (name, layer) in enumerate(model.named_modules()):
    if isinstance(layer, torch.nn.ReLU):
        num_relu_layer += 1
        return_layers.update({f'{name}': f'{name}'})

# resnet18
# return_layers = {
#     'layer1': 'layer1',
#     'layer2': 'layer2',
#     'layer3': 'layer3',
#     'layer4': 'layer4'
# }

# return_layers = {
#     'layer4.0.relu':'layer1'
# }

# densenet121
# return_layers = {
#     'features.denseblock1.denselayer6.relu2': 'layer1',
#     'features.denseblock2.denselayer12.relu2': 'layer2',
#     'features.denseblock3.denselayer19.relu2': 'layer3',
#     'features.denseblock4.denselayer16.relu2': 'layer4'
# }

# return_layers = {
#     # 'features.denseblock1.denselayer6.conv1': 'layer1',
#     # 'features.denseblock2.denselayer12.relu2': 'layer2',
#     'features.transition3': 'layer3',
#     # 'fc.0': 'layer4'
# }

# vgg16_bn
# return_layers = {
#     'features.29': 'layer1',
#     'features.32': 'layer2',
#     'features.36': 'layer3',
#     'features.39': 'layer4'
# }

# return_layers = {
#     # 'fc.1': 'layer1',
#     # 'fc.4': 'layer2',
#     'features.38': 'layer3'
# }

save_dir = join('results', instance)
print(save_dir)
model_hist = sorted(glob.glob(join(save_dir, '*epoch*.pt')), key=lambda x: int (x.split('.pt')[0].split('epoch=')[-1]))

multi_nlid_hist = {}
for key in return_layers.values():
    multi_nlid_hist[key] = []
for cnt, m in enumerate(model_hist):
    print(cnt)
    model.load_state_dict(torch.load(m, map_location=device))
    mid_getter = MidGetter(model, return_layers=return_layers, keep_output=False)
    rep_hist = {}
    for key in return_layers.values():
        rep_hist[key] = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            mid_outputs, _ = mid_getter(x)
            for key in return_layers.values():
                if isinstance(mid_outputs[key], list):
                    rep_hist[key].append(mid_outputs[key][0])
                else:
                    rep_hist[key].append(mid_outputs[key])
    for key in return_layers.values():
        rep_hist[key] = torch.vstack(rep_hist[key])
        rep_hist[key] = rep_hist[key].reshape(len(rep_hist[key]), -1)
        if rep_hist[key].is_cuda:
            rep_hist[key] = rep_hist[key].cpu()
        multi_nlid_hist[key].append(id_estimate(rep_hist[key]))
    metrics = {}
    for key in return_layers.values():
        metrics.update({f'{key}': multi_nlid_hist[key][-1]})
    wandb_run.log(metrics)
    
    torch.save(multi_nlid_hist, join(save_dir, 'multi_nlid_hist.pt'))