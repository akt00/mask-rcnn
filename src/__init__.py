"""
import pathlib

import torch
import torch.utils.data

from torchvision import models, datasets, tv_tensors
from torchvision.transforms import v2
from src.utils import plot
import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T

ROOT = pathlib.Path("./coco")
IMAGES_PATH = str(ROOT / "train2017")
ANNOTATIONS_PATH = str(ROOT / "annotations" / "instances_train2017.json")


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


eval_transform = get_transform(train=False)
dataset = datasets.CocoDetection(IMAGES_PATH, ANNOTATIONS_PATH, transform=get_transform(False))
print(len(dataset))
datasetv2 = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=("boxes", "labels", "masks"))
# plot([datasetv2[43]])
model = models.get_model("maskrcnn_resnet50_fpn_v2", weights=None, weights_backbone=None).eval()
count = 0
device = torch.device("cuda")
model.to(device=device)
for (img, target), (image, target2) in zip(dataset, datasetv2):
    try:
        assert "boxes" in target2.keys() and "masks" in target2.keys() and "labels" in target2.keys()
        count += 1
        # img2 = torch.unsqueeze(img2, 0)
        with torch.no_grad():
            # x = eval_transform(image)
            # convert RGBA -> RGB and move to device
            print(type(target2["boxes"]))
            print(type(target2["masks"]))
            print(target2["labels"])
            x = image.to(device)
            x = x[:3, ...]
            predictions = model([x, x])
            print(len(predictions))
            pred = predictions[0]
            print(pred.keys())
            exit(0)
    except Exception as e:
        # print(target2.keys(), target2["image_id"])
        print(e)
        exit(0)
print(count)

"""
