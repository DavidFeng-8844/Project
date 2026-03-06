from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, session
from PIL import Image

from app.models import AIModel, InferenceTask, db
from app.services.model_service import get_model_service

inference_bp = Blueprint("inference", __name__, url_prefix="/inference")
BIRD_NEST_MODEL_NAME = "Bird Nest Detection"


def _resolve_weights_path(model: AIModel) -> Path:
    weights = Path(model.weights_path)
    if weights.is_absolute():
        return weights
    checkpoint_dir = Path(current_app.config["CHECKPOINT_DIR"])
    project_root = checkpoint_dir.parent
    return (project_root / weights).resolve()


@inference_bp.get("/history")
def get_inference_history():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "authentication required"}), 401

    tasks = (
        InferenceTask.query.filter_by(user_id=user_id)
        .order_by(InferenceTask.created_at.desc(), InferenceTask.id.desc())
        .all()
    )

    return jsonify(
        {
            "history": [
                {
                    "id": task.id,
                    "created_at": task.created_at.isoformat(),
                    "model_name": task.model.name if task.model else None,
                    "result_summary": task.result_summary,
                }
                for task in tasks
            ]
        }
    )


@inference_bp.post("/tasks")
def create_inference_task():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "authentication required"}), 401

    model_id = request.form.get("model_id", type=int)
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400

    model = AIModel.query.get(model_id)
    if not model:
        return jsonify({"error": "model not found"}), 404
    if model.status != "Active":
        return jsonify({"error": "selected model is inactive"}), 400
    if model.name != BIRD_NEST_MODEL_NAME:
        return jsonify({"error": "currently only Bird Nest Detection is supported"}), 400

    weights_path = _resolve_weights_path(model)
    if not weights_path.exists():
        return jsonify({"error": f"weights file not found: {weights_path}"}), 500

    image_file = request.files["image"]
    try:
        pil_image = Image.open(image_file.stream).convert("RGB")
    except Exception as exc:  # pragma: no cover
        current_app.logger.exception("Failed to parse uploaded image")
        return jsonify({"error": f"invalid image: {exc}"}), 400

    runner = get_model_service()

    confidence = (
        request.form.get("confidence", type=float)
        or request.args.get("confidence", type=float)
    )
    threshold = confidence if confidence is not None else runner.confidence
    predictions, annotated = runner.predict(pil_image, confidence=threshold)

    fallback_threshold = 0.2
    fallback_attempted = False
    fallback_applied = False
    used_threshold = threshold

    if len(predictions) == 0 and threshold > fallback_threshold:
        fallback_attempted = True
        retry_predictions, retry_annotated = runner.predict(
            pil_image, confidence=fallback_threshold
        )
        if len(retry_predictions) > 0:
            predictions = retry_predictions
            annotated = retry_annotated
            used_threshold = fallback_threshold
            fallback_applied = True

    annotated_b64 = runner.encode_image_to_base64(annotated)

    summary = f"{len(predictions)} detections by {model.name}"
    task = InferenceTask(
        user_id=user_id,
        model_id=model.id,
        result_summary=summary,
    )
    db.session.add(task)
    db.session.commit()

    scores = [pred.get("score", 0.0) for pred in predictions]
    labels = [pred.get("label", "unknown") for pred in predictions]
    label_distribution: dict[str, int] = {}
    for label in labels:
        label_distribution[label] = label_distribution.get(label, 0) + 1

    return jsonify(
        {
            "task": {
                "id": task.id,
                "user_id": task.user_id,
                "model_id": task.model_id,
                "result_summary": task.result_summary,
                "created_at": task.created_at.isoformat(),
            },
            "model": {"id": model.id, "name": model.name, "type": model.type},
            "predictions": predictions,
            "annotated_image": annotated_b64,
            "confidence_threshold": used_threshold,
            "summary": {
                "detections": len(predictions),
                "max_score": max(scores) if scores else 0.0,
                "avg_score": (sum(scores) / len(scores)) if scores else 0.0,
                "labels": label_distribution,
            },
            "weights_path": str(weights_path),
            "fallback": {
                "attempted": fallback_attempted,
                "applied": fallback_applied,
                "original_threshold": threshold,
                "fallback_threshold": fallback_threshold,
            },
        }
    )
