from flask import Flask

from .config import DefaultConfig
from .routes import api_bp


def create_app(config_object: type[DefaultConfig] | None = None) -> Flask:
    """Application factory for the bird-nest detection service."""
    app = Flask(__name__)

    config_cls = config_object or DefaultConfig
    app.config.from_object(config_cls)

    app.register_blueprint(api_bp)

    return app

