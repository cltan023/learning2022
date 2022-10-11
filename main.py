import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from os.path import join
import argparse
from torch.optim import lr_scheduler
from tqdm import tqdm
from colorama import Fore, Style
import wandb

from utils import model_test
from intrinsic_dimension import id_estimate
from models import resnet18, densenet121, vgg16_bn
from data import get_dataloader, randomize_feature

def main():
    parser = argparse.ArgumentParser(description='Intrinsic Dimension Analysis')
    parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'vgg16_bn', 'densenet121'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--data_name', default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--data_dir', default='../../data', type=str)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_of_per_class', default=5000, type=int)
    parser.add_argument('--noise_x', default=0.0, type=float)
    parser.add_argument('--noise_y', default=0.0, type=float)
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'SGDM', 'ADAM'])
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--batch_size_train', default=512, type=int)
    parser.add_argument('--batch_size_eval', default=256, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--log_interval', default=1, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--save_dir', default='grid-search', type=str)
    parser.add_argument('--debug', default=False, type=bool)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    if args.optimizer == 'SGD':
        args.momentum = 0.0
    elif args.optimizer == 'SGDM':
        args.momentum = 0.9
    args.weight_decay = 1.0e-4

    if args.optimizer != 'ADAM':
        args.group = f'{args.arch}_{args.data_name}_{args.optimizer}{args.momentum}_{args.num_of_per_class}'
    else:
        args.group = f'{args.arch}_{args.data_name}_{args.optimizer}_{args.num_of_per_class}'
    
    args.instance = f'lr={args.learning_rate}_bs={args.batch_size_train}_seed={args.seed}'
    if args.noise_x != 0.0:
        args.instance += f'_noise_x={args.noise_x}'
    if args.noise_y != 0.0:
        args.instance += f'_noise_y={args.noise_y}'

    save_dir = join(args.save_dir, args.group, args.instance)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not args.debug:
        wandb_run = wandb.init(config=args, project='test', name=args.instance, group=args.group)
        
    # freeze the initial state
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # define the model
    if args.arch == 'resnet18':
        model = resnet18(num_classes=args.num_classes).to(device)
    elif args.arch == 'vgg16_bn':
        model = vgg16_bn(num_classes=args.num_classes).to(device)
    elif args.arch == 'densenet121':
        model = densenet121(num_classes=args.num_classes).to(device)
    else:
        raise NotImplementedError

    loss_func = nn.CrossEntropyLoss()

    if args.optimizer == 'ADAM':
        optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # load data
    train_loader, train_loader_eval, test_loader_eval = get_dataloader(data_name=args.data_name, data_dir=args.data_dir, num_of_per_class=args.num_of_per_class, batch_size_train=args.batch_size_train, batch_size_eval=args.batch_size_eval, num_workers=args.num_workers, noise_y=args.noise_y)
        
    milestones = list(range(int(args.num_epochs/2), args.num_epochs, 20))
    # milestones = list(range(20, args.num_epochs, 10))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.6)
    
    # variable to record metrics
    logs = []
    
    # maximum number of examples used to estimate the intrinsic dimension
    # maximum_instances = min(len(train_loader_eval.dataset), len(test_loader_eval.dataset))
    maximum_instances = 10000

    # scrutinize the dynamic behavior in the training epochs
    with tqdm(total=args.num_epochs, colour='MAGENTA', ascii=True) as pbar:
        for i in range(args.num_epochs):
            for x, y in train_loader:
                model.train()
                if args.noise_x != 0.0:
                    x = randomize_feature(x, args.noise_x)
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                ybar, _ = model(x)
                loss = loss_func(ybar, y)
                loss.backward()
                optimizer.step()
            # scheduler.step()

            if i % args.log_interval == 0:
                train_acc, train_loss, feature_maps_train = model_test(model, train_loader_eval, loss_func)
                train_nlid = id_estimate(feature_maps_train[0:maximum_instances])
                
                test_acc, test_loss, feature_maps_test = model_test(model, test_loader_eval, loss_func)
                test_nlid = id_estimate(feature_maps_test[0:maximum_instances])
                
                logs.append([train_acc, test_acc, train_loss, test_loss, train_nlid, test_nlid])
                # torch.save(model.state_dict(), join(save_dir, f'net_dict_epoch={i}.pt'))
                
                if not args.debug:
                    wandb_run.log({'train_acc': train_acc*100, 'test_acc': test_acc*100, 'train_loss': train_loss, 'test_loss': test_loss, 'train_nlid': train_nlid, 'test_nlid':test_nlid})
                
                message = ''
                message += f'{Fore.RED}{train_acc*100:.2f}%{Style.RESET_ALL} '
                message += f'{Fore.GREEN}{test_acc*100:.2f}%{Style.RESET_ALL} '
                message += f'{Fore.BLUE}{train_nlid:.2f}{Style.RESET_ALL} '
                message += f'{Fore.YELLOW}{test_nlid:.2f}{Style.RESET_ALL}'
                
                pbar.set_description(message)
                pbar.update()
    
    logs = torch.tensor(logs)
    torch.save(logs, join(save_dir, 'logs.pt'))
    
if __name__ == '__main__':
    main()