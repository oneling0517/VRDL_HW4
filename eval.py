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


def rgb2ycbcr(im):
    cbcr = np.empty_like(im)
    r = im[:, :, 0]
    g = im[:, :, 1]
    b = im[:, :, 2]
    # Y
    cbcr[:, :, 0] = .299 * r + .587 * g + .114 * b
    # Cb
    cbcr[:, :, 1] = 128 - .169 * r - .331 * g + .5 * b
    # Cr
    cbcr[:, :, 2] = 128 + .5 * r - .419 * g - .081 * b
    return np.uint8(cbcr)


def cal_psnr(sr_img, hr_img):
    sr_img = cv2.cvtColor(np.asarray(sr_img), cv2.COLOR_RGB2BGR)
    hr_img = cv2.cvtColor(np.asarray(hr_img), cv2.COLOR_RGB2BGR)

    sr_img_y = rgb2ycbcr(sr_img)[:, :, 0]
    hr_img_y = rgb2ycbcr(hr_img)[:, :, 0]
    return peak_signal_noise_ratio(hr_img_y, sr_img_y)

psnrs = []
with torch.no_grad():
    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(val_loader):
        # Move to default device
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        # Forward prop.
        sr_imgs = model(lr_imgs)

        # Calculate PSNR
        sr_img = convert_image(sr_imgs.squeeze(0), source='[-1, 1]', target='pil')
        hr_img = convert_image(hr_imgs.squeeze(0), source='[-1, 1]', target='pil')

        psnrs.append(cal_psnr(hr_img, sr_img))

print(f'Model images PSNR: {np.mean(psnrs): .3f}')
