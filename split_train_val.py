import os
import numpy as np
from PIL import Image

root_total = '/content/VRDL_HW4/dataset/training_hr_images/training_hr_images'

root_train = '/content/VRDL_HW4/dataset/train'
root_val = '/content/VRDL_HW4/dataset/val'

if not os.path.isdir(root_train):
    os.mkdir(root_train)
if not os.path.isdir(root_val):
    os.mkdir(root_val)

images = []

# Training proportion
split_proportion = 0.8

# Read list of image-paths
for dir_path, dir_names, file_names in os.walk(root_total):
    for f in file_names:
        images.append(os.path.join(dir_path, f))
#print(len(images))

nbr_train = int(len(images) * split_proportion)
np.random.shuffle(images)
train_images = images[:nbr_train]
val_images = images[nbr_train:]

#train_img, val_img = train_test_split(images, test_size=0.2)
#print(len(train_img))

for img_path in train_images:
    img = Image.open(img_path)
    img.save(os.path.join(root_train, img_path.split('/')[-1]))
for img_path in val_images:
    img = Image.open(img_path)
    img.save(os.path.join(root_val, img_path.split('/')[-1]))

