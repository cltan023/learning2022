import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from os.path import join
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

RS = transforms.Resize(32)
RC = transforms.RandomCrop(32, padding=4)
RHF = transforms.RandomHorizontalFlip()
RVF = transforms.RandomVerticalFlip()
NRM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TT = transforms.ToTensor()

trans_with_aug = transforms.Compose([RC, TT, NRM])
trans_no_aug = transforms.Compose([RS, TT, NRM])

def randomize_feature(x=None, noise_x=0.1):
    noisy_count = int (noise_x * len(x))
    noisy_index = torch.randperm(len(x))[:noisy_count]
    x[noisy_index] = torch.randn(noisy_count, *x.shape[1:])
    return x

def get_dataloader(data_dir='../../data', data_name='cifar10', num_classes=10, num_of_per_class=1000, batch_size_train=64, 
    batch_size_eval=256, num_workers=4, noise_y=0.0):

    if data_name == 'cifar10':
        train_set = datasets.CIFAR10(root=join(data_dir, data_name), train=True, transform=trans_with_aug, download=True)
        train_ids = torch.load(join(data_dir, data_name, f'indices_for_num_of_per_class={num_of_per_class}.pt'))
        train_set = torch.utils.data.Subset(train_set, train_ids)
        test_set = datasets.CIFAR10(root=join(data_dir, data_name), train=False, transform=trans_with_aug, download=True)
    elif data_name == 'cifar100':
        train_set = datasets.CIFAR100(root=join(data_dir, data_name), train=True, transform=trans_with_aug, download=True)
        train_ids = torch.load(join(data_dir, data_name, f'indices_for_num_of_per_class={num_of_per_class}.pt'))
        train_set = torch.utils.data.Subset(train_set, train_ids)
        test_set = datasets.CIFAR100(root=join(data_dir, data_name), train=False, transform=trans_no_aug, download=True)
    elif data_name == 'mnist':
        train_set = datasets.MNIST(root=join(data_dir, data_name), train=True, transform=transforms.ToTensor(), download=True)
        train_ids = torch.load(join(data_dir, data_name, f'indices_for_num_of_per_class={num_of_per_class}.pt'))
        train_set = torch.utils.data.Subset(train_set, train_ids)
        test_set = datasets.MNIST(root=join(data_dir, data_name), train=False, transform=transforms.ToTensor(), download=True)
    elif data_name == 'fashion_mnist':
        train_set = datasets.FashionMNIST(root=join(data_dir, data_name), train=True, transform=transforms.ToTensor(), download=True)
        train_ids = torch.load(join(data_dir, data_name, f'indices_for_num_of_per_class={num_of_per_class}.pt'))
        train_set = torch.utils.data.Subset(train_set, train_ids)
        test_set = datasets.FashionMNIST(root=join(data_dir, data_name), train=False, transform=transforms.ToTensor(), download=True)
    elif data_name == 'kmnist':
        train_set = datasets.KMNIST(root=join(data_dir, data_name), train=True, transform=transforms.ToTensor(), download=True)
        train_ids = torch.load(join(data_dir, data_name, f'indices_for_num_of_per_class={num_of_per_class}.pt'))
        train_set = torch.utils.data.Subset(train_set, train_ids)
        test_set = datasets.KMNIST(root=join(data_dir, data_name), train=False, transform=transforms.ToTensor(), download=True)
    elif data_name == 'svhn':
        NRM = transforms.Normalize(mean=(0.4376821, 0.4437697, 0.47280442), std=(0.19803012, 0.20101562, 0.19703614))
        train_set = datasets.SVHN(root=join(data_dir, data_name), split='train', download=True, transform=transforms.Compose([TT, NRM]))
        train_ids = torch.load(join(data_dir, data_name, f'indices_for_num_of_per_class={num_of_per_class}.pt'))
        train_set = torch.utils.data.Subset(train_set, train_ids)
        test_set = datasets.SVHN(root=join(data_dir, data_name), split='test', download=True, transform=transforms.Compose([TT, NRM]))
    else:
        raise NotImplementedError(f'{data_name} not implemented!!')
    
    if noise_y != 0.0:
        noisy_count = int (noise_y * len(train_set.dataset))
        noisy_index = np.random.permutation(len(train_set.dataset))[:noisy_count]
        noisy_label = np.random.randint(low=0, high=num_classes, size=(noisy_count,))

        if data_name in ['cifar10', 'cifar100']:
            tmp = np.array(train_set.dataset.targets)
            tmp[noisy_index] = noisy_label
            train_set.dataset.targets = tmp.tolist()
        elif data_name in ['mnist', 'fashion_mnist']:
            train_set.dataset.targets[noisy_index] = torch.from_numpy(noisy_label)
        elif data_name == 'svhn':
            train_set.dataset.labels[noisy_index] = noisy_label
        
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    train_loader_eval = DataLoader(train_set, batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader_eval = DataLoader(test_set, batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, train_loader_eval, test_loader_eval