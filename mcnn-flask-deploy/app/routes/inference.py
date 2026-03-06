from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, session
from PIL import Image

from app.models import AIModel, InferenceReport, InferenceTask, db
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

    reports = (
        InferenceReport.query.filter_by(user_id=user_id)
        .order_by(InferenceReport.created_at.desc(), InferenceReport.id.desc())
        .all()
    )

    return jsonify(
        {
            "history": [
                {
                    "id": report.id,
                    "created_at": report.created_at.isoformat(),
                    "model_name": report.model_name,
                    "result_summary": report.summary,
                }
                for report in reports
            ]
        }
    )


@inference_bp.get("/history/<int:report_id>")
def get_inference_report_detail(report_id: int):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "authentication required"}), 401

    report = InferenceReport.query.filter_by(id=report_id, user_id=user_id).first()
    if not report:
        return jsonify({"error": "report not found"}), 404

    try:
        detail = json.loads(report.detail_json)
    except Exception:  # pragma: no cover
        detail = {}

    return jsonify(
        {
            "id": report.id,
            "created_at": report.created_at.isoformat(),
            "model_name": report.model_name,
            "summary": report.summary,
            "detail": detail,
        }
    )


def _predict_with_fallback(pil_image: Image.Image, threshold: float):
    runner = get_model_service()
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
    scores = [pred.get("score", 0.0) for pred in predictions]
    labels = [pred.get("label", "unknown") for pred in predictions]
    label_distribution: dict[str, int] = {}
    for label in labels:
        label_distribution[label] = label_distribution.get(label, 0) + 1

    return {
        "predictions": predictions,
        "annotated_image": annotated_b64,
        "confidence_threshold": used_threshold,
        "summary": {
            "detections": len(predictions),
            "max_score": max(scores) if scores else 0.0,
            "avg_score": (sum(scores) / len(scores)) if scores else 0.0,
            "labels": label_distribution,
        },
        "fallback": {
            "attempted": fallback_attempted,
            "applied": fallback_applied,
            "original_threshold": threshold,
            "fallback_threshold": fallback_threshold,
        },
    }


@inference_bp.post("/batch")
def create_batch_inference_report():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "authentication required"}), 401

    selected_model = request.form.get("selected_model", "bird_nest_real")
    confidence = (
        request.form.get("confidence", type=float)
        or request.args.get("confidence", type=float)
        or 0.5
    )
    images = request.files.getlist("images")
    if not images:
        return jsonify({"error": "at least one image is required"}), 400

    report_items: list[dict] = []
    total_detections = 0

    if selected_model == "oil_leak_mock":
        model_name = "Oil Leak Detection (Mock)"
        for idx, image_file in enumerate(images):
            score = 0.82 if idx % 2 == 0 else 0.67
            detections = 1 if idx % 2 == 0 else 0
            predictions = (
                [{"label": "oil_leak", "score": score, "bbox": [120, 90, 320, 280]}]
                if detections > 0
                else []
            )
            total_detections += detections
            report_items.append(
                {
                    "filename": image_file.filename or f"image_{idx + 1}",
                    "predictions": predictions,
                    "confidence_threshold": confidence,
                    "summary": {
                        "detections": detections,
                        "max_score": score if detections else 0.0,
                        "avg_score": score if detections else 0.0,
                        "labels": {"oil_leak": detections} if detections else {},
                    },
                    "mock": True,
                }
            )
    else:
        model = AIModel.query.get(1)
        if not model:
            return jsonify({"error": "bird nest model not found"}), 404
        if model.status != "Active":
            return jsonify({"error": "selected model is inactive"}), 400
        if model.name != BIRD_NEST_MODEL_NAME:
            return jsonify({"error": "unexpected model configuration"}), 500
        weights_path = _resolve_weights_path(model)
        if not weights_path.exists():
            return jsonify({"error": f"weights file not found: {weights_path}"}), 500

        model_name = model.name
        for idx, image_file in enumerate(images):
            try:
                pil_image = Image.open(image_file.stream).convert("RGB")
            except Exception as exc:  # pragma: no cover
                current_app.logger.exception("Failed to parse uploaded image")
                report_items.append(
                    {
                        "filename": image_file.filename or f"image_{idx + 1}",
                        "error": f"invalid image: {exc}",
                    }
                )
                continue

            item_result = _predict_with_fallback(pil_image, confidence)
            item_result["filename"] = image_file.filename or f"image_{idx + 1}"
            report_items.append(item_result)
            total_detections += item_result["summary"]["detections"]

            # Keep compatibility with existing single-task records.
            task_summary = (
                f"{item_result['summary']['detections']} detections by {model.name}"
            )
            db.session.add(
                InferenceTask(user_id=user_id, model_id=model.id, result_summary=task_summary)
            )

    summary = f"{len(images)} images, total {total_detections} detections"
    report = InferenceReport(
        user_id=user_id,
        model_name=model_name,
        summary=summary,
        detail_json=json.dumps(
            {
                "selected_model": selected_model,
                "confidence": confidence,
                "total_images": len(images),
                "total_detections": total_detections,
                "items": [
                    {
                        "filename": item.get("filename"),
                        "detections": item.get("summary", {}).get("detections", 0),
                        "max_score": item.get("summary", {}).get("max_score", 0.0),
                        "avg_score": item.get("summary", {}).get("avg_score", 0.0),
                        "labels": item.get("summary", {}).get("labels", {}),
                    }
                    for item in report_items
                ],
            }
        ),
    )
    db.session.add(report)
    db.session.commit()

    return jsonify(
        {
            "report": {
                "id": report.id,
                "created_at": report.created_at.isoformat(),
                "model_name": report.model_name,
                "summary": report.summary,
            },
            "items": report_items,
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

    confidence = (
        request.form.get("confidence", type=float)
        or request.args.get("confidence", type=float)
    )
    threshold = confidence if confidence is not None else 0.5
    result = _predict_with_fallback(pil_image, threshold)
    predictions = result["predictions"]
    annotated_b64 = result["annotated_image"]

    summary = f"{len(predictions)} detections by {model.name}"
    task = InferenceTask(
        user_id=user_id,
        model_id=model.id,
        result_summary=summary,
    )
    db.session.add(task)
    db.session.commit()

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
            "confidence_threshold": result["confidence_threshold"],
            "summary": result["summary"],
            "weights_path": str(weights_path),
            "fallback": result["fallback"],
        }
    )
