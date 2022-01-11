from utils import *
from skimage.metrics import peak_signal_noise_ratio
from datasets import SRDataset
import os
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
data_path = '/content/VRDL_HW4/dataset/val'
images = []

# Model checkpoints
srresnet_checkpoint = '/content/VRDL_HW4/models/best_checkpoint_srresnet.pth.tar'

# Load model, either the SRResNet or the SRGAN
# srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
# srgan_generator.eval()
# model = srgan_generator
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()
model = srresnet

# Custom dataloader
val_dataset = SRDataset(split='val', crop_size=0, scaling_factor=3, lr_img_type='imagenet-norm',
                        hr_img_type='[-1, 1]')

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4,
                                         pin_memory=True)

PSNRs = AverageMeter()

with torch.no_grad():
    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(val_loader):
        # Move to default device
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        # Forward prop.
        sr_imgs = model(lr_imgs)

        # Calculate PSNR
        sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
        hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
        psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),data_range=255.)
      
        PSNRs.update(psnr, lr_imgs.size(0))
        
print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
