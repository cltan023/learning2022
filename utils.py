import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
from typing import Dict, Iterable, Callable
from torch import Tensor
from sklearn.decomposition import PCA
import bisect
import shutil
from collections import OrderedDict
import functools

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class IntermediateLayerGetter():
    # https://github.com/sebamenabar/Pytorch-IntermediateLayerGetter/blob/master/torch_intermediate_layer_getter/torch_intermediate_layer_getter.py
    def __init__(self, model, return_layers, keep_output=True):
        """Wraps a Pytorch module to get intermediate values
        
        Arguments:
            model {nn.module} -- The Pytorch module to call
            return_layers {dict} -- Dictionary with the selected submodules
            to return the output (format: {[current_module_name]: [desired_output_name]},
            current_module_name can be a nested submodule, e.g. submodule1.submodule2.submodule3)
        
        Keyword Arguments:
            keep_output {bool} -- If True model_output contains the final model's output
            in the other case model_output is None (default: {True})

        Returns:
            (mid_outputs {OrderedDict}, model_output {any}) -- mid_outputs keys are 
            your desired_output_name (s) and their values are the returned tensors
            of those submodules (OrderedDict([(desired_output_name,tensor(...)), ...).
            See keep_output argument for model_output description.
            In case a submodule is called more than one time, all it's outputs are 
            stored in a list.
        """
        self._model = model
        self.return_layers = return_layers
        self.keep_output = keep_output
        
    def __call__(self, *args, **kwargs):
        ret = OrderedDict()
        handles = []
        for name, new_name in self.return_layers.items():
            layer = rgetattr(self._model, name)
            
            def hook(module, input, output, new_name=new_name):
                if new_name in ret:
                    if type(ret[new_name]) is list:
                        ret[new_name].append(output)
                    else:
                        ret[new_name] = [ret[new_name], output]
                else:
                    ret[new_name] = output
            try:
                h = layer.register_forward_hook(hook)
            except AttributeError as e:
                raise AttributeError(f'Module {name} not found')
            handles.append(h)
            
        if self.keep_output:
            output = self._model(*args, **kwargs)
        else:
            self._model(*args, **kwargs)
            output = None
            
        for h in handles:
            h.remove()
        
        return ret, output

class Logger():
    def __init__(self):
        self.train_acc = []
        self.test_acc = []
        self.train_loss = []
        self.test_loss = []
        self.dist = []
        self.non_linear_id = []
        self.linear_id = []
        self.eigen_values = []
    def update(self, train_acc, test_acc, train_loss, test_loss, dist, non_linear_id, linear_id, eigen_values):
        self.train_acc.append(train_acc)
        self.test_acc.append(test_acc)
        self.train_loss.append(train_loss)
        self.test_loss.append(test_loss)
        self.dist.append(dist)
        self.non_linear_id.append(non_linear_id)
        self.linear_id.append(linear_id)
        self.eigen_values.append(eigen_values)

def pca_decomposition(x):
    pca = PCA()
    pca.fit(x)
    pca_dim = bisect.bisect(np.cumsum(pca.explained_variance_ratio_), 0.9)
    eigen_values = pca.singular_values_
    return pca_dim, eigen_values

def model_test(model, test_loader, loss_function):
    model.eval()
    test_loss = 0
    correct = 0
    device = list(model.parameters())[0].device
    full_feature_map = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, batch_feature_map = model(data)
            if batch_feature_map.is_cuda:
                batch_feature_map = batch_feature_map.cpu()
            full_feature_map.append(batch_feature_map)
            test_loss += loss_function(output, target) * len(output)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        full_feature_map = torch.cat([*full_feature_map])
    return correct / len(test_loader.dataset), test_loss.item() / len(test_loader.dataset), full_feature_map

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self.hooks = []
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            handle = layer.register_forward_hook(self.save_outputs_hook(layer_id))
            self.hooks.append(handle)

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output.data.view(output.shape[0], -1).cpu()
        return fn
    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        y = self.model(x)
        return y, self._features
    
def hist(data=None, bins=20, color='green', cmap='flag'):
    _, _, patches = plt.hist(data, bins, color=color)
    cm = plt.cm.get_cmap(cmap)
    for i, p in enumerate(patches):
        plt.setp(p, 'facecolor', cm(i/bins))
    plt.show()

def get_param(model): 
    # wrt data at the current step
    res = []
    for p in model.parameters():
        if p.requires_grad:
            res.append(p.data.flatten())
    param_flat = torch.cat(res)
    return param_flat

def get_grads(model): 
    # wrt data at the current step
    res = []
    for p in model.parameters():
        if p.requires_grad:
            res.append(p.grad.data.flatten())
    grad_flat = torch.cat(res)
    return grad_flat

def cycle_loader(dataloader):
    while 1:
        for data in dataloader:
            yield data

def covariance_info(model, train_loader, loss_func, num_mini_batch):
    model.train()
    device = list(model.parameters())[0].device
    cycle_train_loader = cycle_loader(train_loader)
    grads = []
    for i, (x, y) in enumerate(cycle_train_loader):
        if i == num_mini_batch:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        ybar, _ = model(x)
        loss = loss_func(ybar, y)
        loss.backward()
        gradient = get_grads(model)
        if gradient.is_cuda:
            gradient = gradient.cpu()
        grads.append(gradient)
    grads = torch.vstack(grads)
    mean_grad = grads.mean(dim=0)
    grads -= mean_grad
    noise_norm = grads.norm(dim=1).mean()
    gram_matrix = torch.matmul(grads, grads.T) / num_mini_batch
    eigen_values, _ = torch.linalg.eigh(gram_matrix)

    return noise_norm.item(), eigen_values[-1].item(), eigen_values[1].item()