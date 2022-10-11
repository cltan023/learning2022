import torch
import numpy as np
import random
from os.path import join
import torchvision.datasets as datasets

data_dir = '../../data'
data_name = 'fashion_mnist'
num_of_per_class = 5000
num_classes = 10
index_to_select = []

if data_name == 'mnist':
    train_set = datasets.MNIST(root=join(data_dir, data_name), train=True, download=True)
elif data_name == 'fashion_mnist':
    train_set = datasets.FashionMNIST(root=join(data_dir, data_name), train=True, download=True)
elif data_name == 'kmnist':
    train_set = datasets.KMNIST(root=join(data_dir, data_name), train=True, download=True)
elif data_name == 'cifar10':
    train_set = datasets.CIFAR10(root=join(data_dir, data_name), train=True, download=True)
elif data_name == 'cifar100':
    train_set = datasets.CIFAR100(root=join(data_dir, data_name), train=True, download=True)
elif data_name == 'svhn':
    train_set = datasets.SVHN(root=join(data_dir, data_name), split='train', download=True)
else:
    raise NotImplementedError

for i in range(num_classes):
    if hasattr(train_set, 'targets'):
        targets = train_set.targets
    elif hasattr(train_set, 'labels'):
        targets = train_set.labels
    else:
        raise NotImplementedError
    if isinstance(targets, list):
        indices = [j for j, ind in enumerate(train_set.targets) if ind==i]
    elif isinstance(targets, torch.Tensor):
        indices = torch.where(train_set.targets==i)[0].tolist()
    elif isinstance(targets, np.ndarray):
        indices = np.argwhere(train_set.labels==i).ravel().tolist()
    else:
        raise NotImplementedError
    random.shuffle(indices)
    # num_of_per_class = min(num_of_per_class, len(indices))
    index_to_select.extend(indices[:num_of_per_class])
random.shuffle(index_to_select)
torch.save(index_to_select, join(data_dir, data_name, f'indices_for_num_of_per_class={num_of_per_class}.pt'))