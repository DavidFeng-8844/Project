from flask import Flask

from .config import DefaultConfig
from .models import db
from .routes import auth_bp, inference_bp, registry_bp, system_bp


def create_app(config_object: type[DefaultConfig] | None = None) -> Flask:
    """Application factory for the bird-nest detection service."""
    app = Flask(__name__)

    config_cls = config_object or DefaultConfig
    app.config.from_object(config_cls)

    db.init_app(app)
    app.register_blueprint(system_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(registry_bp)
    app.register_blueprint(inference_bp)

    return app

