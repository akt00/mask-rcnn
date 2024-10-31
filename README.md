# Mask R-CNN
Instance segmentation model based on [maskrcnn_resnet50_fpn_v2](https://pytorch.org/vision/main/models/mask_rcnn.html).
# Depndencies
All latest
- PyTorch
- Torchvision
- pycocotools
- PyYAML
# How to train and evaluate custom Mask R-CNN
### 1. Prepare the custom dataset in MS COCO format
Or download the MS COCO dataset from [kaggle](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset).
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
### 3. Train the model
The model with the best mAP@50:95 will be saved.
```rb
python3 train.py
```
### 4. Evaluate the model
By default, it will save the visualizations of the model predictions on the validation dataset. Change the config if you only need the evaluation score.
```rb
python3 eval.py
```
# TODO
- Integration with experiment trackers
- Support for online data augmentation
