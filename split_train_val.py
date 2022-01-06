import os

from sklearn.model_selection import train_test_split
from PIL import Image

train_dir = './data/training_hr_images'

out_train_dir = './data/train'
out_val_dir = './data/val'

if not os.path.isdir(out_train_dir):
    os.mkdir(out_train_dir)
if not os.path.isdir(out_val_dir):
    os.mkdir(out_val_dir)

images = []

# Read list of image-paths
for dir_path, dir_names, file_names in os.walk(train_dir):
    for f in file_names:
        images.append(os.path.join(dir_path, f))

train_img, val_img = train_test_split(images, test_size=0.2)

for img_path in train_img:
    img = Image.open(img_path)
    img.save(os.path.join(out_train_dir, img_path.split('/')[-1]))
for img_path in val_img:
    img = Image.open(img_path)
    img.save(os.path.join(out_val_dir, img_path.split('/')[-1]))
