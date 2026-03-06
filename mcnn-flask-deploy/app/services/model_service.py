from __future__ import annotations

from functools import lru_cache

from flask import current_app

from .inference import MaskRCNNService, load_class_names


@lru_cache(maxsize=1)
def get_model_service() -> MaskRCNNService:
    config = current_app.config
    class_names = load_class_names(config["CLASS_NAMES_PATH"])
    service = MaskRCNNService(
        model_path=config["MODEL_PATH"],
        class_names=class_names,
        target_size=config["TARGET_SIZE"],
        confidence=config["CONFIDENCE_THRESHOLD"],
        color_map_path=config.get("COLOR_MAP_PATH"),
    )
    return service

