#coding: utf-8

import os
import time
import numpy as np
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
import torch.nn as nn
import torchvision.models as models
import visdom

from torch.autograd import Variable
from torchviz import make_dot, make_dot_from_trace
from graphviz import Digraph
from tensorboardX import SummaryWriter

from models import * #define nets by zhangcc
from tools import *  #define tools by zhangcc

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_root', default='./data', help='dataset path')

def adjust_learning_rate(optimizer, iteration, lr_interval):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (iteration // lr_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

def validate(val_loader, model, criterion):
    """Validate the model on Validation Set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # Evaluate all the validation set
    for i, (input, target) in enumerate(val_loader):
        input, target = input.to(args.device), target.to(args.device)

        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
        
        # compute output
        output = model(input_var)
        # For nets that have multiple outputs such as Inception
        if isinstance(output, tuple):
            loss = sum((criterion(o,target_var) for o in output))
            # print (output)
            for o in output:
                prec1 = accuracy(o.data, target, topk=(1,))
                top1.update(prec1[0], input.size(0))
            losses.update(loss.data[0], input.size(0)*len(output))
        else:
            loss = criterion(output, target_var)
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))
            losses.update(loss.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
            
        # Info log every args.print_freq
        if True:
            args.logger.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1_val:.4f} ({top1_avg:.4f})'.format(
                   i, len(val_loader), batch_time=batch_time,
                   loss=losses,
                   top1_val=np.asscalar(top1.val.cpu().numpy()),
                   top1_avg=np.asscalar(top1.avg.cpu().numpy())))

    args.logger.info(' * Prec@1 {top1}'
          .format(top1=np.asscalar(top1.avg.cpu().numpy())))

    return top1.avg

def save_checkpoint(model, epoch, optimizer, filename):
    #print('=> Saving...')
    args.logger.info('=> Saving...')
    state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': args.best_acc,
            'optimizer' : optimizer.state_dict(),
        }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, filename)

def save_inference(model):
    torch.save(model, "./checkpoint/inference_temp.pth")

    model.to(args.device)
    model.eval()

    input = torch.rand(1, args.channels, args.height, args.width).to(args.device)
    traced_script_module = torch.jit.trace(model, input)
    traced_script_module.save("./checkpoint/inference.pth")

def net_vision(model):
    input = torch.rand(1, args.channels, args.height, args.width)
    g = make_dot(model(input), params=dict(model.named_parameters()))
    g.view()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, test_loader, model, criterion, optimizer, epoch, lr_interval, test_interval):
    """Train the model on Training Set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        #test 
        if 0 == (epoch * len(train_loader) + i) % test_interval:
            args.logger.info("\n\n","=> testing...")
            prec1 = validate(test_loader, model, criterion)

            args.logger.info("=> test accuracy: {}".format(prec1.item()))
            if prec1[0] >= args.best_acc:
                args.best_acc = prec1.item()
                args.logger.info("=> higher accuracy: {} \n\n".format(args.best_acc))
                save_checkpoint(model, epoch, optimizer, args.resume_file)
                save_inference(model)
            else:
                args.logger.info("=> higher accuracy: {} \n\n".format(args.best_acc))
    
        # measure data loading time
        data_time.update(time.time() - end)
        input, target = input.to(args.device), target.to(args.device)
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        #topk = (1,5) if labels >= 100 else (1,) # TO FIX
        # For nets that have multiple outputs such as Inception
        if isinstance(output, tuple):
            loss = sum((criterion(o,target_var) for o in output))
            # print (output)
            for o in output:
                prec1 = accuracy(o.data, target, topk=(1,))
                top1.update(prec1[0], input.size(0))
            losses.update(loss.data[0], input.size(0)*len(output))
        else:
            loss = criterion(output, target_var)
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))
            losses.update(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
            
        # Info log every args.print_freq
        if i % args.print_freq == 0:
        
            adjust_learning_rate(optimizer, epoch * len(train_loader) + i, lr_interval)
            
            for param_group in optimizer.param_groups: 
                learning_rate = param_group['lr']
           
            args.logger.info('Epoch: [{0}][{1}/{2}]\t' 
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1_val:.4f} ({top1_avg:.4f})\t'
                  'lr {lr:.6f} \t'
                  'iter: {iter}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   top1_val=np.asscalar(top1.val.cpu().numpy()),
                   top1_avg=np.asscalar(top1.avg.cpu().numpy()),
                   lr = learning_rate,
                   iter = epoch * len(train_loader) + i))

def demo():
    vis = visdom.Visdom()

    global args, labels
    args = parser.parse_args()

    #log
    log_file = ".\\checkpoint\\log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)
    args.logger = setup_logger("", ".\\checkpoint\\", 0)

    #data_root = "H:\\POY\\poylongjumptopside\\origin\\train\\longjump_cascade_assem"
    data_root = ".\\data"
    args.resume_file = ".\\checkpoint\\resume_training_model.pkl"

    args.channels = 3
    args.height = 224
    args.width = 224

    args.best_acc = 0.0
    args.print_freq = 100
    test_interval = 100

    start_epoch = 0
    epochs = 10000

    lr_interval = 1000
    args.lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4

    resume = 0
    pretrained = 0
    device_ids = [0]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')
    args.logger.info("=> train_dir:", train_dir)
    args.logger.info("=> test_dir:", test_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            normalize,
        ]))

    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            normalize,
        ]))

    labels = len(train_dataset.classes)
    args.logger.info("=> total_class_num: {}".format(labels))
    for i, (input, target) in enumerate(train_dataset):
        args.logger.info("=> train input_size: {} {} {}".format(input.size(0),input.size(1),input.size(2)))
        break
    for i, (input, target) in enumerate(test_dataset):
        args.logger.info("=> test input_size: {} {} {}".format(input.size(0),input.size(1),input.size(2)))
        break
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False)

    args.logger.info("=> using resnet18 network...")
    #model = models.resnet18(pretrained=pretrained)
    #model.fc = nn.Linear(model.fc.in_features, labels)#change total cls

    model = alexnet(pretrained=pretrained)
    model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, labels),
        )
    print(model)
   
    net_vision(model)

    args.logger.info("gpu_id num: {}".format(torch.cuda.device_count()))
    #if torch.cuda.device_count() > 0:
    #   model = nn.DataParallel(model, device_ids=device_ids)
    model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=momentum, weight_decay=weight_decay)

    #resume
    if resume:
        if os.path.isfile(args.resume_file):
            args.logger.info('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume_file)
            args.best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, epochs):
        args.logger.info("=> training...")
        train(train_loader, test_loader, model, criterion, optimizer, epoch, lr_interval, test_interval)

if __name__=="__main__":
    demo()