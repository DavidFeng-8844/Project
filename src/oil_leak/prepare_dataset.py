"""
Dataset preparation for Oil Leak Detection.

Modes:
  single  — Convert original VOC dataset only (103 images)
  merged  — Merge original VOC + CSDN YOLO datasets (~333 images)

Usage:
  python prepare_dataset.py --mode single
  python prepare_dataset.py --mode merged

Output goes to data/processed/oil_leak or data/processed/oil_leak_merged.
"""

from __future__ import annotations

import argparse
import random
import shutil
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------
# Source paths
# ---------------------
VOC_SOURCE = REPO_ROOT / "data" / "active" / "original_oil_leak"
CSDN_SOURCE = REPO_ROOT / "data" / "active" / "oil_leak_yolo"

# ---------------------
# Output paths
# ---------------------
SINGLE_OUTPUT = REPO_ROOT / "data" / "processed" / "oil_leak"
MERGED_OUTPUT = REPO_ROOT / "data" / "processed" / "oil_leak_merged"

# ---------------------
# Dataset config
# ---------------------
CLASS_NAME = "oil_leak"
CLASS_ID = 0
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


# ================================================================== #
#  VOC parsing utilities
# ================================================================== #

@dataclass
class SamplePair:
    stem: str
    image_path: Path
    xml_path: Path


def discover_pairs(source_dir: Path) -> List[SamplePair]:
    image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.JPG"))
    image_files += list(source_dir.glob("*.jpeg")) + list(source_dir.glob("*.JPEG"))
    image_by_stem = {img.stem: img for img in image_files}

    pairs: List[SamplePair] = []
    for stem, img_path in image_by_stem.items():
        xml_path = source_dir / f"{stem}.xml"
        if xml_path.exists():
            pairs.append(SamplePair(stem=stem, image_path=img_path, xml_path=xml_path))

    pairs.sort(key=lambda p: p.stem)
    return pairs


def _safe_float(text: Optional[str], default: float = 0.0) -> float:
    if text is None:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def parse_voc_xml(
    xml_path: Path,
) -> Tuple[int, int, List[Tuple[float, float, float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_tag = root.find("size")
    if size_tag is None:
        raise ValueError(f"Missing <size> in XML: {xml_path}")

    width = int(_safe_float(size_tag.findtext("width"), default=0))
    height = int(_safe_float(size_tag.findtext("height"), default=0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in XML: {xml_path}")

    bboxes: List[Tuple[float, float, float, float]] = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = max(0.0, min(_safe_float(bnd.findtext("xmin")), float(width)))
        ymin = max(0.0, min(_safe_float(bnd.findtext("ymin")), float(height)))
        xmax = max(0.0, min(_safe_float(bnd.findtext("xmax")), float(width)))
        ymax = max(0.0, min(_safe_float(bnd.findtext("ymax")), float(height)))

        if xmax > xmin and ymax > ymin:
            bboxes.append((xmin, ymin, xmax, ymax))

    return width, height, bboxes


def to_yolo_line(
    bbox: Tuple[float, float, float, float], img_w: int, img_h: int,
) -> str:
    xmin, ymin, xmax, ymax = bbox
    xc = min(max(((xmin + xmax) / 2.0) / img_w, 0.0), 1.0)
    yc = min(max(((ymin + ymax) / 2.0) / img_h, 0.0), 1.0)
    w = min(max((xmax - xmin) / img_w, 0.0), 1.0)
    h = min(max((ymax - ymin) / img_h, 0.0), 1.0)
    return f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


# ================================================================== #
#  Splitting
# ================================================================== #

def _split(items: list, ratio: float, seed: int) -> Tuple[list, list]:
    if not items:
        return [], []
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    idx = int(len(shuffled) * ratio)
    if len(shuffled) >= 2:
        idx = min(max(idx, 1), len(shuffled) - 1)
    else:
        idx = len(shuffled)
    return shuffled[:idx], shuffled[idx:]


def split_stratified(
    pairs: List[SamplePair], ratio: float, seed: int,
) -> Tuple[List[SamplePair], List[SamplePair]]:
    positives, negatives = [], []
    for p in pairs:
        try:
            _, _, bboxes = parse_voc_xml(p.xml_path)
            (positives if bboxes else negatives).append(p)
        except Exception:
            negatives.append(p)

    p_tr, p_val = _split(positives, ratio, seed)
    n_tr, n_val = _split(negatives, ratio, seed + 1)

    rng = random.Random(seed + 2)
    train = p_tr + n_tr
    val = p_val + n_val
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ================================================================== #
#  Directory helpers
# ================================================================== #

def _ensure_dirs(base: Path) -> None:
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (base / sub).mkdir(parents=True, exist_ok=True)


def _clear_dir(base: Path) -> None:
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            folder = base / sub / split
            if not folder.exists():
                continue
            for p in folder.iterdir():
                if p.is_file():
                    p.unlink()


def _write_label(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def _safe_filename(prefix: str, name: str, idx: int, suffix: str) -> str:
    candidate = f"{prefix}{name}"
    if len(candidate.encode("utf-8")) <= 240:
        return candidate
    return f"{prefix}{idx:04d}{suffix}"


def generate_yaml(out_dir: Path, yaml_name: str) -> Path:
    yaml_path = out_dir / yaml_name
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        f"names: ['{CLASS_NAME}']\n",
        encoding="utf-8",
    )
    return yaml_path


# ================================================================== #
#  Ingest: VOC source
# ================================================================== #

def ingest_voc(
    source: Path, out_dir: Path, prefix: str = "",
) -> dict:
    pairs = discover_pairs(source)
    if not pairs:
        raise RuntimeError(f"No VOC pairs in {source}")

    train_pairs, val_pairs = split_stratified(pairs, TRAIN_RATIO, RANDOM_SEED)
    stats = {k: 0 for k in ("train_img", "train_obj", "val_img", "val_obj")}

    for split, samples in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split

        for s in samples:
            try:
                w, h, bboxes = parse_voc_xml(s.xml_path)
            except Exception as exc:
                warnings.warn(f"Skip {s.xml_path}: {exc}")
                continue

            dst_img = img_dir / f"{prefix}{s.image_path.name}"
            dst_lbl = lbl_dir / f"{prefix}{s.stem}.txt"
            shutil.copy2(s.image_path, dst_img)
            _write_label(dst_lbl, [to_yolo_line(b, w, h) for b in bboxes])

            stats[f"{split}_img"] += 1
            stats[f"{split}_obj"] += len(bboxes)

    return stats


# ================================================================== #
#  Ingest: pre-labeled YOLO source
# ================================================================== #

CSDN_SPLIT_MAP = {"train": "train", "valid": "val"}


def ingest_yolo(
    source: Path, out_dir: Path, prefix: str = "csdn_",
) -> dict:
    stats = {k: 0 for k in ("train_img", "train_obj", "val_img", "val_obj")}

    for src_split, dst_split in CSDN_SPLIT_MAP.items():
        src_img = source / src_split / "images"
        src_lbl = source / src_split / "labels"
        dst_img = out_dir / "images" / dst_split
        dst_lbl = out_dir / "labels" / dst_split

        if not src_img.exists():
            warnings.warn(f"Split not found: {src_img}")
            continue

        for idx, img in enumerate(sorted(src_img.glob("*.*"))):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            safe_name = _safe_filename(prefix, img.name, idx, img.suffix)
            safe_stem = Path(safe_name).stem
            shutil.copy2(img, dst_img / safe_name)

            lbl_path = src_lbl / f"{img.stem}.txt"
            n = 0
            if lbl_path.exists():
                content = lbl_path.read_text().strip()
                n = len([l for l in content.split("\n") if l.strip()]) if content else 0
                shutil.copy2(lbl_path, dst_lbl / f"{safe_stem}.txt")
            else:
                _write_label(dst_lbl / f"{safe_stem}.txt", [])

            stats[f"{dst_split}_img"] += 1
            stats[f"{dst_split}_obj"] += n

    return stats


# ================================================================== #
#  Main
# ================================================================== #

def prepare_single() -> Path:
    print(f"Source : {VOC_SOURCE}")
    print(f"Output : {SINGLE_OUTPUT}")
    _ensure_dirs(SINGLE_OUTPUT)
    _clear_dir(SINGLE_OUTPUT)

    stats = ingest_voc(VOC_SOURCE, SINGLE_OUTPUT)
    yaml_path = generate_yaml(SINGLE_OUTPUT, "oil_leak.yaml")

    print(f"  Train: {stats['train_img']} images, {stats['train_obj']} objects")
    print(f"  Val:   {stats['val_img']} images, {stats['val_obj']} objects")
    print(f"  YAML:  {yaml_path}")
    return yaml_path


def prepare_merged() -> Path:
    print(f"Source A (VOC) : {VOC_SOURCE}")
    print(f"Source B (YOLO): {CSDN_SOURCE}")
    print(f"Output         : {MERGED_OUTPUT}")
    _ensure_dirs(MERGED_OUTPUT)
    _clear_dir(MERGED_OUTPUT)

    print("\n  Ingesting VOC source ...")
    voc = ingest_voc(VOC_SOURCE, MERGED_OUTPUT, prefix="orig_")
    print(f"    Train: {voc['train_img']} images, {voc['train_obj']} objects")
    print(f"    Val:   {voc['val_img']} images, {voc['val_obj']} objects")

    print("  Ingesting YOLO source ...")
    csdn = ingest_yolo(CSDN_SOURCE, MERGED_OUTPUT)
    print(f"    Train: {csdn['train_img']} images, {csdn['train_obj']} objects")
    print(f"    Val:   {csdn['val_img']} images, {csdn['val_obj']} objects")

    yaml_path = generate_yaml(MERGED_OUTPUT, "oil_leak_merged.yaml")

    total_tr = voc["train_img"] + csdn["train_img"]
    total_val = voc["val_img"] + csdn["val_img"]
    print(f"\n  Merged train: {total_tr}, val: {total_val}")
    print(f"  YAML: {yaml_path}")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare oil-leak YOLO dataset")
    parser.add_argument(
        "--mode",
        choices=["single", "merged"],
        default="merged",
        help="single = original VOC only; merged = VOC + CSDN",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Preparing oil-leak dataset  [mode={args.mode}]")
    print("=" * 60)

    if args.mode == "single":
        yaml_path = prepare_single()
    else:
        yaml_path = prepare_merged()

    print(f"\n{'='*60}")
    print(f"Done. Train with:  python train.py --data {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
