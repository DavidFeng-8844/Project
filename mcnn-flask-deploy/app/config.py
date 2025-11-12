from __future__ import annotations

import os
from pathlib import Path


class DefaultConfig:
    """Default configuration for the inference service."""

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    CHECKPOINT_DIR: Path = BASE_DIR / "checkpoints"

    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        str(CHECKPOINT_DIR / "maskrcnn_resnet50_fpn_v2.pth"),
    )
    CLASS_NAMES_PATH: str = os.getenv(
        "CLASS_NAMES_PATH",
        str(CHECKPOINT_DIR / "class_names.json"),
    )
    COLOR_MAP_PATH: str | None = os.getenv(
        "COLOR_MAP_PATH",
        str(CHECKPOINT_DIR / "bird_nest-colormap.json")
        if (CHECKPOINT_DIR / "bird_nest-colormap.json").exists()
        else None,
    )
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
    TARGET_SIZE: int = int(os.getenv("TARGET_SIZE", 512))

