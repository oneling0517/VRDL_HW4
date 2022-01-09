'''
import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr import Net
from datasets import SRDataset

# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    #train_set = DatasetFromHdf5("data/train.h5")
    #training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    # Custom dataloaders
    train_dataset = SRDataset(split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',
                              hr_img_type='[-1, 1]')

    val_dataset = SRDataset(split='val',
                            crop_size=0,
                            scaling_factor=scaling_factor,
                            lr_img_type='imagenet-norm',
                            hr_img_type='[-1, 1]')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=workers,
                                             pin_memory=True)

    print("===> Building model")
    model = Net()
    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))  

    print("===> Setting Optimizer")
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(train_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(train_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    for iteration, batch in enumerate(train_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        loss = criterion(model(input), target)
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm(model.parameters(),opt.clip) 
        optimizer.step()

        if iteration%100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(train_loader), loss.data[0]))

def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
'''
import os
import time

import torch.backends.cudnn as cudnn
from skimage.metrics import peak_signal_noise_ratio
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from datasets import SRDataset
from vdsr import Net
from utils import *

# Data parameters
crop_size = 291  # crop size of target HR images
scaling_factor = 3  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

# Model parameters
large_kernel_size = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
n_channels = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks = 16  # number of residual blocks
checkpoint_dir = 'models'
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

# Learning parameters
epoch = 0
checkpoint = None  # path to model checkpoint, None if none
batch_size = 32  # batch size
start_epoch = 0  # start at this epoch
iterations = 1e4  # number of training iterations
workers = 4  # number of workers for loading data in the DataLoader
lr = 1e-3  # learning rate
grad_clip = True  # clip if gradients are exploding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = Net(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                         n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)
        # Initialize the optimizer
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # Custom dataloaders
    train_dataset = SRDataset(split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',
                              hr_img_type='[-1, 1]')

    val_dataset = SRDataset(split='val',
                            crop_size=0,
                            scaling_factor=scaling_factor,
                            lr_img_type='imagenet-norm',
                            hr_img_type='[-1, 1]')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=workers,
                                             pin_memory=True)
    # Total number of epochs to train for
    epochs = int(iterations // len(train_loader) + 1)

    max_psnr = 0
    # Epochs
    for epoch in range(start_epoch, epochs):

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)
        psnr = val(val_loader=val_loader,
                   model=model,
                   epoch=epoch)
        if psnr.avg > max_psnr:
            max_psnr = psnr.avg
            torch.save({'epoch': epoch, 'model': model, 'optimizer': optimizer},
                       os.path.join(checkpoint_dir, 'best_checkpoint_srresnet.pth.tar'))

        # Save checkpoint
        torch.save({'epoch': epoch, 'model': model, 'optimizer': optimizer},
                   os.path.join(checkpoint_dir, 'checkpoint_srresnet.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    PSNRs = AverageMeter()  # PSNR

    start = time.time()

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        # Forward prop.
        sr_imgs = model(lr_imgs)

        # Calculate PSNR
        sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
        hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
        psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().detach().numpy(), sr_imgs_y.cpu().detach().numpy(), data_range=255.)
        PSNRs.update(psnr, lr_imgs.size(0))

        # Loss
        loss = criterion(sr_imgs, hr_imgs)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        # Keep track of loss
        losses.update(loss.item(), lr_imgs.size(0))

        # Keep track of batch time
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

    writer.add_scalar('Loss/train', losses.avg, epoch)
    writer.add_scalar('PSNR/train', PSNRs.avg, epoch)
    print(
        f'Epoch: {epoch} -- '
        f'Batch Time: {batch_time.avg:.3f} -- '
        f'Loss: {losses.avg:.4f} -- '
        f'PSNR: {PSNRs.avg:.4f}'
    )
    del lr_imgs, hr_imgs, sr_imgs  # free some memory since their histories may be stored


def val(val_loader, model, epoch):
    model.eval()
    PSNRs = AverageMeter()
    with torch.no_grad():
        for i, (lr_imgs, hr_imgs) in enumerate(val_loader):
            # Move to default device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)
            sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
            hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
            psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
            PSNRs.update(psnr, lr_imgs.size(0))
    print(f'Epoch: {epoch}, PSNR: {PSNRs.avg}')
    writer.add_scalar('PSNR/val', PSNRs.avg, epoch)
    del lr_imgs, hr_imgs, sr_imgs

    return PSNRs


if __name__ == '__main__':
    main()

