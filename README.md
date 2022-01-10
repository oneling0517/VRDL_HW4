# VRDL_HW4

## The link of my Colab

Click [My colab link](https://colab.research.google.com/drive/1mrqrHPnOH7Jx5W3o_2td_CifL7wjtpEo?usp=sharing) or just run Super resolution.ipynb

## Git clone my project
```
!git clone https://github.com/oneling0517/VRDL_HW4.git
```
## Dataset Download
```
os.chdir("/content/VRDL_HW4")
!gdown --id '1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb' --output dataset.zip

!apt-get install unzi
!unzip -q 'dataset.zip' -d dataset
```

## Training
Use Mask R-CNN resnet101
```
os.chdir("/content/Mask_RCNN/samples/VRDL_HW3")
python3 nucleus.py train --dataset=/content/Mask_RCNN/dataset/dataset --subset=train --weights=imagenet
```

## Validation
```
os.chdir("/content/Mask_RCNN/samples/VRDL_HW3")
python3 nucleus.py detect --dataset=/content/Mask_RCNN/dataset/dataset --subset=val --weights=/content/Mask_RCNN/log/mask_rcnn_nucleus_0019.h5
```

## Testing
Use the weights from [Google Drive](https://drive.google.com/file/d/1Apj1jhAVYkVR-SDFrIpeDchNBDkPjfMd/view?usp=sharing).
```
os.chdir("/content/Mask_RCNN")
!gdown --id '1Apj1jhAVYkVR-SDFrIpeDchNBDkPjfMd' --output weights19.zip

!apt-get install unzi
!unzip -q 'weights19.zip' -d log
```
```
os.chdir("/content/Mask_RCNN/samples/VRDL_HW3")
python3 nucleus.py detect --dataset=/content/Mask_RCNN/dataset/dataset --subset=test --weights=/content/Mask_RCNN/log/mask_rcnn_nucleus_0019.h5
```

## Inference

You can click [Inference.ipynb](https://colab.research.google.com/drive/13vLcOs_x6R_ALSdEjlYYxuOcER0Xr-gd?usp=sharing).

## Reference
https://github.com/matterport/Mask_RCNN/tree/master/samples/nucleus
https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
