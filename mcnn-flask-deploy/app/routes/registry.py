from __future__ import annotations

from flask import Blueprint, jsonify, request, session

from app.models import AIModel, db

registry_bp = Blueprint("registry", __name__, url_prefix="/registry")


def _is_admin() -> bool:
    return session.get("role") == "Admin"


@registry_bp.get("/models")
def list_models():
    models = AIModel.query.order_by(AIModel.id.asc()).all()
    return jsonify(
        {
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "type": model.type,
                    "status": model.status,
                    "weights_path": model.weights_path,
                }
                for model in models
            ]
        }
    )


@registry_bp.patch("/models/<int:model_id>/status")
def update_model_status(model_id: int):
    if not session.get("user_id"):
        return jsonify({"error": "authentication required"}), 401
    if not _is_admin():
        return jsonify({"error": "admin permission required"}), 403

    payload = request.get_json(silent=True) or {}
    status = str(payload.get("status", "")).strip().capitalize()
    if status not in {"Active", "Inactive"}:
        return jsonify({"error": "status must be Active or Inactive"}), 400

    model = AIModel.query.get(model_id)
    if not model:
        return jsonify({"error": "model not found"}), 404

    model.status = status
    db.session.commit()

    return jsonify(
        {
            "message": "model status updated",
            "model": {
                "id": model.id,
                "name": model.name,
                "type": model.type,
                "status": model.status,
                "weights_path": model.weights_path,
            },
        }
    )
