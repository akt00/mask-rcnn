# Mask R-CNN
Instance segmentation model based on [maskrcnn_resnet50_fpn_v2](https://pytorch.org/vision/main/models/mask_rcnn.html)

# Depndencies
All latest
- PyTorch
- Torchvision
- pycocotools
- PyYAML
# How to train the custom model
### 1. Prepare the custom dataset in MS COCO format
```rb
|
|_dataset
|    |_annotation_train.json
|    |_annotation_val.json
|    |_train
|    |   |_0.jpg
|    |   |_...
|    |_val
|    |   |_0.jpg
|    |   |_...
```
### 2. Modify the config file
```rb
./cfg/config.yaml
```
### 3. Run the script
The model with the best mAP@50:95 will be saved
```rb
python3 train.py
```
# TODO
- Integration with an experiment tracker
- Support for online data augmentation