from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor, float32
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


class COCODataset(Dataset):

    def __init__(
        self,
        *,
        image_path: Path,
        annotation_path: Path,
        preproc: v2.Compose = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(dtype=float32, scale=True),
            ]
        ),
    ):
        super().__init__()
        self.path = image_path
        self.coco = COCO(annotation_path)
        self.images = self.coco.getImgIds()
        # remove empty labels
        for idx in reversed(range(len(self.images))):
            ann_ids = self.coco.getAnnIds(imgIds=self.images[idx])
            if not ann_ids:
                del self.images[idx]

        self.preproc = preproc

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, dict]:
        # images
        img_id = self.images[index]
        img_path = self.coco.loadImgs(img_id)[0]
        h, w = img_path["height"], img_path["width"]
        img = Image.open(self.path / img_path["file_name"])
        img = self.preproc(img)
        # annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        labels = []
        bboxes = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w < 1. or h < 1:
                continue
            bboxes.append([x, y, x + w, h + y])
            labels.append(ann["category_id"])
            masks.append(self.coco.annToMask(ann))
            areas.append(ann["area"])
            iscrowd.append(ann["iscrowd"])

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                bboxes, format="XYXY", canvas_size=(h, w)
            ),
            "labels": tv_tensors.TVTensor(labels).long(),
            "masks": tv_tensors.Mask(np.array(masks)),
            "area": tv_tensors.TVTensor(np.array(areas)),
            "image_id": ann["image_id"],
            "iscrowd": tv_tensors.TVTensor(np.array(iscrowd)),
        }

        return img, target


def collate_fn(batch: list) -> tuple[list, list]:
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
