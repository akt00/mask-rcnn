# How to download the weight from trochvision
```rb
import torch
import torchvision


model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
    weights=models.detection.MaskRCNN_ResNet50_FPN_V2_Weight.DEFAULT
)
torch.save({"model": model.state_dict()}, "weights/best.pth")
```