"""
End-to-end training script for Oil Leak Detection with YOLOv8.

Pipeline:
1) Prepare YOLO dataset directories
2) Convert PASCAL VOC XML to YOLO label format
3) Split dataset into train/val (8:2)
4) Generate YOLO YAML config
5) Launch YOLOv8n training
"""

from __future__ import annotations

import random
import shutil
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# =========================
# User-configurable paths
# =========================
# Relative to repo root.
# Raw JPG/XML source folder (as requested): @data/active/original_oil_leak
SOURCE_DIR = Path("data/active/original_oil_leak")
YOLO_DATA_DIR = Path("datasets/oil_leak")


# =========================
# Training configuration
# =========================
CLASS_NAME = "oil_leak"
CLASS_ID = 0
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

EPOCHS = 50
IMG_SIZE = 416
BATCH_SIZE = 16


@dataclass
class SamplePair:
    """A source image + xml annotation pair."""

    stem: str
    image_path: Path
    xml_path: Path


def ensure_yolo_dirs(base_dir: Path) -> None:
    """Create standard YOLO directory structure."""
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (base_dir / sub).mkdir(parents=True, exist_ok=True)


def discover_pairs(source_dir: Path) -> List[SamplePair]:
    """
    Discover valid image/xml pairs by file stem.
    Supports .jpg/.jpeg (case-insensitive).
    """
    image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.JPG"))
    image_files += list(source_dir.glob("*.jpeg")) + list(source_dir.glob("*.JPEG"))
    image_by_stem = {}
    for img in image_files:
        image_by_stem[img.stem] = img

    pairs: List[SamplePair] = []
    for stem, img_path in image_by_stem.items():
        xml_path = source_dir / f"{stem}.xml"
        if xml_path.exists():
            pairs.append(SamplePair(stem=stem, image_path=img_path, xml_path=xml_path))

    pairs.sort(key=lambda p: p.stem)
    return pairs


def safe_float(text: Optional[str], default: float = 0.0) -> float:
    """Best-effort float parser."""
    if text is None:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[Tuple[float, float, float, float]]]:
    """
    Parse VOC XML and return:
    - image width
    - image height
    - list of bboxes as (xmin, ymin, xmax, ymax)

    Robust behavior:
    - Missing/invalid object tags => returns empty bbox list
    - Invalid boxes are skipped
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_tag = root.find("size")
    if size_tag is None:
        raise ValueError(f"Missing <size> in XML: {xml_path}")

    width = int(safe_float(size_tag.findtext("width"), default=0))
    height = int(safe_float(size_tag.findtext("height"), default=0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in XML: {xml_path}")

    bboxes: List[Tuple[float, float, float, float]] = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = safe_float(bnd.findtext("xmin"))
        ymin = safe_float(bnd.findtext("ymin"))
        xmax = safe_float(bnd.findtext("xmax"))
        ymax = safe_float(bnd.findtext("ymax"))

        # Clamp to image bounds to avoid out-of-range labels.
        xmin = max(0.0, min(xmin, float(width)))
        xmax = max(0.0, min(xmax, float(width)))
        ymin = max(0.0, min(ymin, float(height)))
        ymax = max(0.0, min(ymax, float(height)))

        if xmax <= xmin or ymax <= ymin:
            continue

        bboxes.append((xmin, ymin, xmax, ymax))

    return width, height, bboxes


def to_yolo_line(
    bbox: Tuple[float, float, float, float], img_w: int, img_h: int, class_id: int
) -> str:
    """Convert VOC absolute bbox to one YOLO label line."""
    xmin, ymin, xmax, ymax = bbox

    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    # Keep values in [0, 1] for safety.
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def split_train_val(
    pairs: List[SamplePair], train_ratio: float, seed: int
) -> Tuple[List[SamplePair], List[SamplePair]]:
    """Shuffle and split sample pairs."""
    if not pairs:
        return [], []

    rng = random.Random(seed)
    pairs_shuffled = pairs[:]
    rng.shuffle(pairs_shuffled)

    split_idx = int(len(pairs_shuffled) * train_ratio)
    if len(pairs_shuffled) >= 2:
        split_idx = min(max(split_idx, 1), len(pairs_shuffled) - 1)
    else:
        split_idx = len(pairs_shuffled)

    return pairs_shuffled[:split_idx], pairs_shuffled[split_idx:]


def write_label_file(label_path: Path, lines: List[str]) -> None:
    """Write YOLO label lines; empty file for negative/background images."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))
            f.write("\n")


def process_split(
    split_name: str,
    samples: List[SamplePair],
    yolo_dir: Path,
) -> Tuple[int, int]:
    """
    Copy images and write labels for one split.
    Returns:
    - processed image count
    - object (bbox) count
    """
    images_dir = yolo_dir / "images" / split_name
    labels_dir = yolo_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    object_count = 0

    for sample in samples:
        dst_image_path = images_dir / sample.image_path.name
        dst_label_path = labels_dir / f"{sample.stem}.txt"

        try:
            img_w, img_h, bboxes = parse_voc_xml(sample.xml_path)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Skip invalid XML ({sample.xml_path}): {exc}")
            continue

        yolo_lines = [to_yolo_line(b, img_w, img_h, CLASS_ID) for b in bboxes]

        shutil.copy2(sample.image_path, dst_image_path)
        write_label_file(dst_label_path, yolo_lines)

        image_count += 1
        object_count += len(yolo_lines)

    return image_count, object_count


def generate_yaml(yolo_dir: Path, class_name: str) -> Path:
    """Generate YOLO dataset yaml file."""
    yaml_path = yolo_dir / "oil_leak.yaml"
    yaml_content = (
        f"path: {yolo_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        f"names: ['{class_name}']\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def train_yolov8(yaml_path: Path) -> None:
    """Launch YOLOv8 training with requested fast settings."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    train_kwargs = {
        "data": str(yaml_path),
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
        "batch": BATCH_SIZE,
        "amp": True,
    }

    # Only set device=0 when CUDA is available, otherwise keep default behavior.
    try:
        import torch

        if torch.cuda.is_available():
            train_kwargs["device"] = 0
    except Exception:  # noqa: BLE001
        pass

    model.train(**train_kwargs)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    source_dir = (repo_root / SOURCE_DIR).resolve()
    yolo_dir = (repo_root / YOLO_DATA_DIR).resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"SOURCE_DIR not found: {source_dir}")

    ensure_yolo_dirs(yolo_dir)

    pairs = discover_pairs(source_dir)
    if not pairs:
        raise RuntimeError(
            f"No valid image/xml pairs found in {source_dir}. "
            "Please ensure each image has the same-stem XML."
        )

    train_pairs, val_pairs = split_train_val(pairs, TRAIN_RATIO, RANDOM_SEED)

    train_images, train_objects = process_split("train", train_pairs, yolo_dir)
    val_images, val_objects = process_split("val", val_pairs, yolo_dir)

    yaml_path = generate_yaml(yolo_dir, CLASS_NAME)

    print("=" * 60)
    print("Dataset preparation finished")
    print(f"Source dir      : {source_dir}")
    print(f"YOLO dataset dir: {yolo_dir}")
    print(f"Total pairs     : {len(pairs)}")
    print(f"Train images    : {train_images}, objects: {train_objects}")
    print(f"Val images      : {val_images}, objects: {val_objects}")
    print(f"YAML path       : {yaml_path}")
    print("=" * 60)

    if train_images == 0:
        raise RuntimeError("No valid training images after XML parsing. Training aborted.")
    if val_images == 0:
        warnings.warn("Validation split is empty after parsing; training will still continue.")

    print("Starting YOLOv8 training...")
    train_yolov8(yaml_path)


if __name__ == "__main__":
    main()
