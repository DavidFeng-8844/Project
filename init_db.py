from __future__ import annotations

import sys
from pathlib import Path

from werkzeug.security import generate_password_hash

PROJECT_ROOT = Path(__file__).resolve().parent
APP_ROOT = PROJECT_ROOT / "mcnn-flask-deploy"

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app import create_app  # noqa: E402
from app.config import DefaultConfig  # noqa: E402
from app.models import AIModel, User, db  # noqa: E402


def seed_users() -> None:
    if not User.query.filter_by(username="admin").first():
        db.session.add(
            User(
                username="admin",
                password_hash=generate_password_hash("admin123"),
                role="Admin",
            )
        )

    if not User.query.filter_by(username="user").first():
        db.session.add(
            User(
                username="user",
                password_hash=generate_password_hash("user123"),
                role="User",
            )
        )


def seed_models() -> None:
    models_to_seed = [
        {
            "name": "Bird Nest Detection",
            "type": "Object Detection",
            "status": "Active",
            "weights_path": "checkpoints/maskrcnn_resnet50_fpn_v2.pth",
        },
        {
            "name": "Oil Leak Detection",
            "type": "Infrared",
            "status": "Inactive",
            "weights_path": "checkpoints/oil_leak_placeholder.pth",
        },
    ]

    for model_data in models_to_seed:
        if not AIModel.query.filter_by(name=model_data["name"]).first():
            db.session.add(AIModel(**model_data))


def main() -> None:
    app = create_app(DefaultConfig)
    with app.app_context():
        db.create_all()
        seed_users()
        seed_models()
        db.session.commit()
        print(f"Database initialized at: {DefaultConfig.DB_PATH}")


if __name__ == "__main__":
    main()
