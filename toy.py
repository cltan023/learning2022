from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy as np
import glob
from os.path import join
import random

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from intrinsic_dimension import id_estimate
from nngeometry import FIM, PMatBlockDiag, PMatDiag, PMatDense, FIM_MonteCarlo
from utils import get_grads, get_param

import sys
sys.path.append('..')

class Net(nn.Module):

    def __init__(self, width=512, input=1024, output=10, transfer_func='tanh'):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=input, out_features=width, bias=False)
        self.fc2 = nn.Linear(in_features=width, out_features=output, bias=False)
        if transfer_func == 'tanh':
            self.transfer_func = nn.Tanh()
        elif transfer_func == 'sigmoid':
            self.transfer_func = nn.Sigmoid()
        elif transfer_func == 'relu':
            self.transfer_func = nn.ReLU()
        self.hidden_representaion = None
         
    def forward(self, x):
        # x = x.view(-1, 784)
        out1 = self.fc1(x)
        out2 = self.transfer_func(out1)
        out3 = self.fc2(out2)
        self.hidden_representaion = out2

        return out3

def model_test(model, test_loader, loss_function):
    model.eval()
    test_loss = 0
    correct = 0
    device = list(model.parameters())[0].device
    full_feature_map = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_feature_map = model.hidden_representaion
            if batch_feature_map.is_cuda:
                batch_feature_map = batch_feature_map.cpu()
            full_feature_map.append(batch_feature_map)
            test_loss += loss_function(output, target) * len(output)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        full_feature_map = torch.cat([*full_feature_map])
    return correct / len(test_loader.dataset), test_loss.item() / len(test_loader.dataset), full_feature_map

def create_dataset(train_loader, test_loader, mapping_dim):
    # dimensionality reduction with PCA, available at
    # https://stats.stackexchange.com/questions/125172/pca-on-train-and-test-datasets-should-i-run-one-pca-on-traintest-or-two-separa
    train_x = []
    train_y = []
    for data, target in train_loader:
        train_x.append(data.reshape(len(data), -1))
        train_y.append(target)
    train_x = torch.vstack(train_x)

    pca = PCA(n_components=mapping_dim)
    scaler = StandardScaler()

    train_x = torch.from_numpy(scaler.fit_transform(pca.fit_transform(train_x.numpy())))
    # train_x = torch.from_numpy(pca.fit_transform(train_x.numpy()))
    train_y = torch.hstack(train_y)

    train_set = torch.utils.data.TensorDataset(train_x, train_y)

    test_x = []
    test_y = []
    for data, target in test_loader:
        test_x.append(data.reshape(len(data), -1))
        test_y.append(target)
    test_x = torch.vstack(test_x)
    test_x = torch.from_numpy(scaler.fit_transform(pca.transform(test_x.numpy())))
    # test_x = torch.from_numpy(pca.transform(test_x.numpy()))
    test_y = torch.hstack(test_y)

    test_set = torch.utils.data.TensorDataset(test_x, test_y)

    return train_set, test_set

def plot(logs, save_dir, file_name):
    logs = np.array(logs)
    num_figs = logs.shape[1]
    col = 2
    row = int ((num_figs)/2)

    fig, ax = plt.subplots(row, col, figsize=(3*col, 3*row))
    
    cnt = 0
    for i in range(row):
        for j in range(col):
            ax[i][j].plot(logs.T[cnt], label=cnt)
            ax[i][j].legend()

            cnt += 1
    plt.tight_layout()
    plt.savefig(join(save_dir, f'{file_name}.jpeg'), format='jpeg', dpi=300, bbox_inches=None)
    plt.close()

# create a smaller data set
from data import get_dataloader

from itertools import product

seeds = [1, 2, 3]
data_names = ['fashion_mnist', 'kmnist']
transfer_funcs = ['tanh', 'relu']

for seed, data_name, transfer_func in product(seeds, data_names, transfer_funcs):
    arch = 'fcn'
    optimizer = 'SGD'
    momentum = 0.0
    # num_epochs = 100
    num_epochs = 200
    num_classes = 10
    data_dir = '../../data'
    num_of_per_class = 5000

    learning_rate = 0.001
    batch_size_train = 512
    batch_size_eval = 512
    num_workers = 4
    mapping_dim = 200
    hidden_dim = 100
    gpu_id = 0


    # freeze the initial state
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    _, train_loader_eval, test_loader = get_dataloader(data_name=data_name, data_dir=data_dir, num_of_per_class=num_of_per_class, batch_size_train=batch_size_train, batch_size_eval=batch_size_eval, num_workers=num_workers)
    train_set, test_set = create_dataset(train_loader_eval, test_loader, mapping_dim)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    if optimizer != 'ADAM':
        instance = f'{arch}_{data_name}_{transfer_func}_{optimizer}{momentum}_{num_of_per_class}_lr={learning_rate}_bs={batch_size_train}_seed={seed}'
    else:
        instance = f'{arch}_{data_name}_{transfer_func}_{optimizer}_{num_of_per_class}_lr={learning_rate}_bs={batch_size_train}_seed={seed}'

    save_dir = join('fisher-info', instance)
    print(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    if arch == 'fcn':
        model = Net(input=mapping_dim, output=num_classes, width=hidden_dim, transfer_func=transfer_func).to(device)

    loss_func = torch.nn.CrossEntropyLoss()

    if optimizer == 'ADAM':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum)

    # milestones = list(range(10, num_epochs, 20))
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    covariance_hist = []
    corrcoef_hist = []
    results = []
    for i in range(num_epochs):
        for j, (data, target) in enumerate(train_loader):
            model.train()
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            prediction = model(data)
            loss = loss_func(prediction, target)
            loss.backward()

            optimizer.step()

        # scheduler.step()

        train_acc, train_loss, _ = model_test(model, train_loader, loss_func)
        test_acc, test_loss, hidden_representaion = model_test(model, test_loader, loss_func)
        test_nlid = id_estimate(hidden_representaion)

        gram_matrix = torch.cov(hidden_representaion.T)
        
        diag_first = gram_matrix.diag().mean()
        diag_second = gram_matrix.diag().abs().mean()
        
        off_diag_first = (gram_matrix.sum() - diag_first * len(gram_matrix)) / (len(gram_matrix)*(len(gram_matrix)-1))
        off_diag_second = (gram_matrix.abs().sum() - diag_second * len(gram_matrix)) / (len(gram_matrix)*(len(gram_matrix)-1))
        covariance_hist.append([diag_first.item(), off_diag_first.item(), diag_second.item(), off_diag_second.item(), gram_matrix.mean().item(), gram_matrix.abs().mean().item()])
        plot(covariance_hist, save_dir, 'covariance_hist')

        gram_matrix = torch.corrcoef(hidden_representaion.T)
        gram_matrix = torch.nan_to_num(gram_matrix)
        
        diag_first = gram_matrix.diag().mean()
        diag_second = gram_matrix.diag().abs().mean()
        
        off_diag_first = (gram_matrix.sum() - diag_first * len(gram_matrix)) / (len(gram_matrix)*(len(gram_matrix)-1))
        off_diag_second = (gram_matrix.abs().sum() - diag_second * len(gram_matrix)) / (len(gram_matrix)*(len(gram_matrix)-1))
        corrcoef_hist.append([diag_first.item(), off_diag_first.item(), diag_second.item(), off_diag_second.item(), gram_matrix.mean().item(), gram_matrix.abs().mean().item()])
        plot(corrcoef_hist, save_dir, 'corrcoef_hist')

        model.eval()
        emp_fisher = FIM(model, train_loader, PMatBlockDiag, num_classes, device=device)
        key = list(emp_fisher.data.keys())[0]
        emp_fisher = emp_fisher.data[key]
        diag = emp_fisher.diag().mean()
        off_diag = (emp_fisher.sum() - diag * len(emp_fisher)) / (len(emp_fisher) * (len(emp_fisher) - 1))
        diag_abs = emp_fisher.diag().abs().mean()
        off_diag_abs = (emp_fisher.abs().sum() - diag * len(emp_fisher)) / (len(emp_fisher) * (len(emp_fisher) - 1))

        results.append([train_acc, test_acc, train_loss, test_loss, test_nlid, emp_fisher.mean().item(), diag.item(), off_diag.item(), emp_fisher.abs().mean().item(), off_diag_abs.item()])
        # print(train_acc, test_acc, test_nlid)
        plot(results, save_dir, 'results')

        # torch.save(model.state_dict(), join(save_dir, f'net_dict_epoch={i}.pt'))
    results = np.array(results)
    covariance_hist = np.array(covariance_hist)
    corrcoef_hist = np.array(corrcoef_hist)
    torch.save(results, join(save_dir, 'results.pt'))
    torch.save(covariance_hist, join(save_dir, 'covariance_hist.pt'))
    torch.save(corrcoef_hist, join(save_dir, 'corrcoef_hist.pt'))