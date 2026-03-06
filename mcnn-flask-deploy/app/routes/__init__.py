from .auth import auth_bp
from .inference import inference_bp
from .registry import registry_bp
from .system import system_bp

__all__ = ["auth_bp", "registry_bp", "inference_bp", "system_bp"]
