from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.dataset import COCODataset, collate_fn
from src.engine import train_one_epoch, evaluate
from src.models import get_mask_rcnn


def train(cfg: dict):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_dataset = COCODataset(
        image_path=Path(cfg["train"]),
        annotation_path=Path(cfg["train_annotation"]),
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["batch"],
        shuffle=True,
        num_workers=cfg["workers"],
        collate_fn=collate_fn,
        persistent_workers=cfg["persistent"],
    )

    val_dataset = COCODataset(
        image_path=Path(cfg["val"]),
        annotation_path=Path(cfg["val_annotation"]),
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg["batch"],
        shuffle=False,
        num_workers=cfg["workers"],
        collate_fn=collate_fn,
        persistent_workers=cfg["persistent"],
    )

    model = get_mask_rcnn(num_classes=cfg["nc"], coco=cfg["coco"])
    model.to(device=device)

    params = [p for p in model.parameters() if p.requires_grad]

    if cfg["optimizer"].lower() == "sgd":
        sgd_cfg = cfg["sgd"]
        optim = torch.optim.SGD(
            params=params,
            lr=sgd_cfg["lr"],
            momentum=sgd_cfg["momentum"],
            weight_decay=sgd_cfg["decay"],
            nesterov=sgd_cfg["nesterov"],
        )
    elif cfg["optimizer"].lower() == "adamw":
        adamw_cfg = cfg["adamw"]
        optim = torch.optim.AdamW(
            params=params,
            lr=adamw_cfg["lr"],
            betas=adamw_cfg["betas"],
            weight_decay=adamw_cfg["decay"],
        )
    else:
        NotImplementedError(f"Unsupported optimizer: {cfg['optimizer']}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optim,
        T_max=cfg["epoch"],
        eta_min=cfg["min_lr"],
    )

    scaler = torch.GradScaler() if cfg["scaler"] else None

    epochs = cfg["epoch"]
    best_map = 0

    for e in range(epochs):
        # print(f"----- Epoch {e} ----- at LR: {scheduler.get_last_lr()[0]:.4f}")
        train_one_epoch(
            model=model,
            optimizer=optim,
            data_loader=train_loader,
            device=device,
            epoch=e,
            print_freq=100,
            scaler=scaler,
        )

        scheduler.step()

        metrics = evaluate(model=model, data_loader=val_loader, device=device)
        # IoU=0.50:0.95
        if best_map < metrics["segm"]["pr"]:
            best_map = metrics["segm"]["pr"]
            torch.save(
                {
                    "epoch": e,
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                },
                Path("weights/best.pth"),
            )

    print("Done!")


if __name__ == "__main__":
    path = Path("cfg/config.yaml")

    with open(path) as fd:
        try:
            cfg = yaml.safe_load(fd)
        except yaml.YAMLError as e:
            print(f"Failed to load the YAML file: {e}")
            exit(1)

    train(cfg=cfg)
