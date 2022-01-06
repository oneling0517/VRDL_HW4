import os
import time

import torch.backends.cudnn as cudnn
from skimage.metrics import peak_signal_noise_ratio
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from datasets import SRDataset
from model import SRResNet
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
        model = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
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
