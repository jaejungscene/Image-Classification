import os
from args import get_args_parser
args = get_args_parser().parse_args()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

import time
import dataset
import create
import numpy as np
import wandb
from datetime import datetime
from log import save_checkpoint, printSave_one_epoch, printSave_start_condition, printSave_end_state
from utils import accuracy, adjust_learning_rate, AverageMeter, get_learning_rate
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

result_folder_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
best_err1 = 100
best_err5 = 100



def run():
    start = time.time()
    global args, best_err1, best_err5

    train_loader, val_loader, numberofclass = dataset.create_dataloader(args)
    model = create.create_model(args, numberofclass)
    printSave_start_condition(args, sum([p.data.nelement() for p in model.parameters()]))
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = create.create_criterion(args, numberofclass)
    optimizer = create.create_optimizer(args, model)
    cudnn.benchmark = True

    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch, args)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1 # if err1 <= best_err1, is_best is True
        if is_best:
            best_err1 = err1
            best_err5 = err5

        if args.wandb == True:
            wandb.log({'top-1 err': err1, 'top-5 err':err5, 'train loss':train_loss, 'validation loss':val_loss})
        print('Current best accuracy (top-1 and 5 error):\t', best_err1, 'and', best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'best_err1': best_err1,
            'best_err5': best_err5,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, args)

    total_time = time.time()-start
    total_time = time.strftime('%H:%M:%S', time.localtime(total_time))
    printSave_end_state(args, best_err1, best_err5, total_time)





def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        end = time.time()

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        # print('train tuple:',type(output))
        # print(len(output))
        if args.distil > 0:
            loss = criterion(input, output, target)
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        if args.distil > 0:
            err1, err5 = accuracy(output[0].data, target, topk=(1, 5))
        else:
            err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
                
    printSave_one_epoch(epoch, args, batch_time, data_time, top1, top5, losses)
    # print('* Epoch[{0}/{1}]\t Top 1-err {top1.avg:.3f}\t  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
    #     epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg




def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        target = target.cuda()

        output = model(input)
        if args.distil > 0:
            loss = criterion(input, output, target, val=True)
            # if args.wandb == True:
                # wandb.log({"roc": wandb.plot.roc_curve(target, output[0])})
                # wandb.log({"pr": wandb.plots.precision_recall(target, output[0])})
        else:
            loss = criterion(output, target)
            # if args.wandb == True:
                # wandb.log({"roc": wandb.plot.roc_curve(target, output)})
                # wandb.log({"pr": wandb.plots.precision_recall(target, output)})

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                data_time=data_time, top1=top1, top5=top5))

    printSave_one_epoch(epoch, args, batch_time, data_time, top1, top5, losses, False)
    # print('* Epoch[{0}/{1}]\t Top 1-err {top1.avg:.3f}\t  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
    #     epoch+1, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg




if __name__ == '__main__':
    if args.wandb == True:
        if args.net_type.endswith('resnet'):
            temp = (args.net_type+str(args.depth)+'_'+args.dataset+'_'+'b'+str(args.batch_size)+'_'+'s'+str(args.insize))
        else:
            temp = (args.net_type+'_'+args.dataset+'_'+'b'+str(args.batch_size)+'_'+'s'+str(args.insize)+'_distil-'+str(args.distil))
        wandb.init(project='self-directed-research', name=temp, entity='jaejungscene')
    
    run()
