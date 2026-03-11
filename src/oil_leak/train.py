"""
YOLOv8 training for Oil Leak Detection.

Usage:
  python train.py --data path/to/dataset.yaml
  python train.py --data path/to/dataset.yaml --epochs 200 --model yolov8m.pt
  python train.py  (defaults to merged dataset)

This script only handles training.
Run prepare_dataset.py first to build the dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_YAML = REPO_ROOT / "data" / "processed" / "oil_leak_merged" / "oil_leak_merged.yaml"


def train(
    data: str,
    model: str = "yolov8s.pt",
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 8,
    patience: int = 30,
    seed: int = 42,
) -> None:
    from ultralytics import YOLO

    yolo = YOLO(model)

    train_kwargs = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "amp": True,
        "patience": patience,
        "single_cls": True,
        "cos_lr": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.02,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_bias_lr": 0.01,
        "mosaic": 0.3,
        "mixup": 0.05,
        "close_mosaic": 15,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.3,
        "shear": 1.0,
        "perspective": 0.0,
        "fliplr": 0.5,
        "flipud": 0.1,
        "cache": "ram",
        "seed": seed,
        "deterministic": True,
    }

    try:
        import torch

        if torch.cuda.is_available():
            train_kwargs["device"] = 0
    except Exception:  # noqa: BLE001
        pass

    yolo.train(**train_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for oil-leak detection")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_YAML),
        help="Path to dataset YAML",
    )
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Oil Leak Detection — YOLOv8 Training")
    print(f"  Data  : {args.data}")
    print(f"  Model : {args.model}")
    print(f"  Epochs: {args.epochs}")
    print("=" * 60)

    train(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
