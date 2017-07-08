
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import os
import itertools
import shutil
import tqdm
import numpy as np
from sklearn.metrics import fbeta_score

def accuracy(output, target, topk=(1,)):
    """Computes the precision for the specified values of k"""
    res = 0
    target = target.cpu().numpy()
    output = output.cpu().numpy()

    total = len(target[:,0])

    for i in range(total):
        for j in range(len(output[0,:])):
            if output[i,j] > 0.25:
                output[i,j] = 1
            else:
                output[i,j] = 0
        res += fbeta_score(target[i,:], output[i,:], beta=2)
    res /= (total+0.0)
    return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer(object):
    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, max_iter):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.top1 = AverageMeter()
        self.losses = AverageMeter()
        self.best_prec1 = 0
        self.criterion = nn.BCELoss()

    def validate(self):
        self.model.eval()
        top1 = AverageMeter()
        losses = AverageMeter()
        for batch_idx, (data) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Testing on epoch %d' % self.epoch, ncols=80, leave=False):
            if self.cuda:
                data = data.cuda()

            data_var = Variable(data, volatile=True)
            # compute output
            output = self.model(data_var)
            if batch_idx == 0:
                output_result = output
            else:
                output_result = torch.cat([output_result,output],0)
        np.savetxt('result.txt',output_result.cpu().data.numpy())

    def train_epoch(self):
        self.model.train()
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            target = target.float()
            if self.cuda:
                data, target = data.cuda(), target.cuda()

            data_var, target_var = Variable(data), Variable(target)
            output = self.model(data_var)

            softmax_loss = self.criterion(output, target_var)
            loss = softmax_loss

            # measure accuracy and record loss
            prec = accuracy(output.data, target, topk=(1,))
            self.losses.update(loss.data[0], data.size(0))
            self.top1.update(prec, data.size(0))

            # compute gradient and do SGD step
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.iteration >= self.max_iter:
                break

        print('Epoch: [{0}][{1}/{2}]\t'
        'Train Loss {3:.2f} ({4:.2f})\t'
        'Train Prec@1 {5:.3f} ({6:.3f})\t'.format(
        self.epoch, batch_idx, len(self.train_loader),
        float(self.losses.val), float(self.losses.avg), float(self.top1.val), float(self.top1.avg)))
        self.losses.reset()
        self.top1.reset()

    def train(self):


        for epoch in itertools.count(self.epoch):
            self.epoch = epoch

            self.train_epoch()
            

            if self.val_loader:
                prec1 = self.validate()
            if self.iteration >= self.max_iter:
                break

            
            

            is_best = prec1 > self.best_prec1
            self.best_prec1 = max(prec1,self.best_prec1)
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
            }, is_best)


