# How to download the weight from trochvision
```rb
import torch
import torchvision


model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
    weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
)
torch.save({"model": model.state_dict()}, "weights/best.pth")
```