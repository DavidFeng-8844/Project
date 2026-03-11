from __future__ import annotations

import os
from pathlib import Path


def _default_db_uri() -> str:
    base = Path(__file__).resolve().parent.parent.parent.parent
    return f"sqlite:///{base / 'mcnn_saas.db'}"


def _default_checkpoint_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "checkpoints"


def _engine_options(uri: str) -> dict:
    """Connection pool for PostgreSQL; SQLite uses default (no pool)."""
    if uri.startswith("postgresql"):
        return {
            "pool_size": int(os.getenv("SQLALCHEMY_POOL_SIZE", "10")),
            "max_overflow": int(os.getenv("SQLALCHEMY_MAX_OVERFLOW", "20")),
            "pool_pre_ping": True,
        }
    return {}


class DefaultConfig:
    """Configuration via environment variables for multi-env and multi-worker deployment."""

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    PROJECT_ROOT: Path = BASE_DIR.parent.parent
    CHECKPOINT_DIR: Path = Path(os.getenv("CHECKPOINT_DIR", str(_default_checkpoint_dir())))
    DB_PATH: Path = PROJECT_ROOT / "mcnn_saas.db"

    SQLALCHEMY_DATABASE_URI: str = os.getenv("DATABASE_URL", _default_db_uri())
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_ENGINE_OPTIONS: dict = _engine_options(
        os.getenv("DATABASE_URL", _default_db_uri())
    )

    # Production must set SECRET_KEY; same value required on all workers for cookie session.
    SECRET_KEY: str = os.getenv("SECRET_KEY", "mvp-dev-secret-key")

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
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    TARGET_SIZE: int = int(os.getenv("TARGET_SIZE", "512"))

