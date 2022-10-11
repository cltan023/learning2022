import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import os
import multiprocessing as mp

def lass_csr(model, device, loader, alpha=0.25/255, beta=0.2/255, r=0.3/255, max_iter=10):
    """
    Reference: Langevin Adversarial Sample Search (LASS) algorithm in the paper 'A Closer Look at Memorization in Deep Networks' <https://arxiv.org/abs/1706.05394>.
    """
    model.train()

    count_cs = 0  
    for x, _ in loader:

        x_hat = Variable(x.data, requires_grad=True).to(device)
        x = x.to(device)
        yhat, _ = model(x)
        pred_on_x = yhat.argmax(dim=1, keepdim=False)

        for i in range(max_iter):
            # compute gradient on x_hat
            x_hat = Variable(x_hat.cpu().data, requires_grad=True).to(device)
            x_hat.retain_grad()
            output_on_x_hat, _ = model(x_hat)
            cost = -F.nll_loss(output_on_x_hat, pred_on_x)

            model.zero_grad()
            if x_hat.grad is not None:
                print('fill 0 to grad')
                x_hat.grad.data.fill_(0)
            cost.backward()

            # take a step
            noise = torch.randn_like(x_hat)
            x_hat.grad.sign_()
            x_hat = x_hat - (alpha*x_hat.grad + beta*noise)

            # projec back to the box
            x_hat = torch.max(torch.min(x_hat, x+r), x-r)
            
            # check is adversial
            y_hat, _ = model(x_hat)
            index_not_adv = pred_on_x.view(-1,1).eq(y_hat.argmax(dim=1, keepdim=True)).view(-1)
            num_not_adv = index_not_adv.sum().item()

            # record number of adversial samples
            count_cs = count_cs + (pred_on_x.size(0)-num_not_adv)
            # print('count_cs: {}, num_not_adv: {}'.format(count_cs, num_not_adv))
            if num_not_adv>0:
                x_hat = x_hat[index_not_adv]#.unsqueeze(1)
                x = x[index_not_adv]#.unsqueeze(1)
                pred_on_x = pred_on_x[index_not_adv]#.view(-1)
            else:
                break

    return count_cs/len(loader.dataset)


def fbm_csr(model, device, loader, alpha=0.25/255, beta=0.2/255, r=0.3/255, max_iter=10, hursts=None):
    model.train()

    count_cs = 0
    sequence_length = 1024
    
    from fbm import FBM
    global generate_fgn
    def generate_fgn(args):
        f = FBM(n=sequence_length, hurst=args, length=sequence_length, method='daviesharte')
        fgn_sample = f.fgn()
        return fgn_sample
    pool = mp.Pool(os.cpu_count()-4)
    fgn_noise = pool.map(generate_fgn, hursts)
    pool.close()
    pool.join()
    fgn_noise = torch.from_numpy(np.array(fgn_noise).T).float().to(device)
    
    sample_used = 0
    for x, _ in loader:

        x_hat = Variable(x.data, requires_grad=True).to(device)
        x = x.to(device)
        yhat, _ = model(x)
        pred_on_x = yhat.argmax(dim=1, keepdim=False)
        
        for i in range(max_iter):
            # compute gradient on x_hat
            x_hat = Variable(x_hat.cpu().data, requires_grad=True).to(device)
            x_hat.retain_grad()
            output_on_x_hat, _ = model(x_hat)
            cost = -F.nll_loss(output_on_x_hat, pred_on_x)

            model.zero_grad()
            if x_hat.grad is not None:
                print('fill 0 to grad')
                x_hat.grad.data.fill_(0)
            cost.backward()

            # take a step
            # noise = np.random.normal()
            noise = fgn_noise[sample_used % sequence_length].reshape(x.shape[1:])
            sample_used += 1
            x_hat.grad.sign_()
            x_hat = x_hat - (alpha*x_hat.grad + beta*noise)

            # projec back to the box
            x_hat = torch.max(torch.min(x_hat, x+r), x-r)
            
            # check is adversial
            y_hat, _ = model(x_hat)
            index_not_adv = pred_on_x.view(-1,1).eq(y_hat.argmax(dim=1, keepdim=True)).view(-1)
            num_not_adv = index_not_adv.sum().item()

            # record number of adversial samples
            count_cs = count_cs + (pred_on_x.size(0)-num_not_adv)
            # print('count_cs: {}, num_not_adv: {}'.format(count_cs, num_not_adv))
            if num_not_adv>0:
                x_hat = x_hat[index_not_adv]#.unsqueeze(1)
                x = x[index_not_adv]#.unsqueeze(1)
                pred_on_x = pred_on_x[index_not_adv]#.view(-1)
            else:
                break

    return count_cs/len(loader.dataset)