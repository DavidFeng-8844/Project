from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import tempfile
from uuid import uuid4

from flask import Blueprint, current_app, jsonify, request, session
from PIL import Image
import concurrent.futures
import io

from app.models import AIModel, InferenceReport, InferenceTask, db
from app.services.model_service import (
    get_model_service,
    get_yolo_service,
    run_infrared_hybrid_detection,
)

inference_bp = Blueprint("inference", __name__, url_prefix="/inference")

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
BIRD_NEST_MODEL_NAME = "Bird Nest Detection"
OIL_LEAK_MODEL_NAME = "Oil Leak Detection"
OIL_LEAK_DEFAULT_CONFIDENCE = 0.35
BIRD_NEST_DEFAULT_CONFIDENCE = 0.5


def _resolve_weights_path(model: AIModel) -> Path:
    weights = Path(model.weights_path)
    if weights.is_absolute():
        resolved = weights
    else:
        checkpoint_dir = Path(current_app.config["CHECKPOINT_DIR"])
        project_root = checkpoint_dir.parent
        resolved = (project_root / weights).resolve()

    # Robust fallback for Oil Leak model when DB still stores legacy placeholder path.
    # This keeps service available even before DB migration/update is applied.
    if not resolved.exists():
        model_name = (model.name or "").strip().lower()
        model_type = (model.type or "").strip().lower()
        is_oil_leak = ("oil" in model_name and "leak" in model_name) or (
            "oil" in model_type and "leak" in model_type
        )
        if is_oil_leak:
            fallback = Path(current_app.config["CHECKPOINT_DIR"]) / "oil_leak_yolov8.pt"
            if fallback.exists():
                return fallback.resolve()

    return resolved


def _is_bird_nest_model(model: AIModel) -> bool:
    name = (model.name or "").strip().lower()
    return "bird" in name and "nest" in name


def _is_oil_leak_model(model: AIModel) -> bool:
    name = (model.name or "").strip().lower()
    model_type = (model.type or "").strip().lower()
    return ("oil" in name and "leak" in name) or (
        "oil" in model_type and "leak" in model_type
    )


def _is_infrared_model(model: AIModel) -> bool:
    name = (model.name or "").strip().lower()
    return "infrared" in name


def _default_confidence_for_model(model: AIModel) -> float:
    if _is_oil_leak_model(model):
        return OIL_LEAK_DEFAULT_CONFIDENCE
    return BIRD_NEST_DEFAULT_CONFIDENCE


def _save_annotated_image(annotated_image: Image.Image, model_name: str) -> str:
    static_root = Path(current_app.static_folder)
    output_dir = static_root / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_token = model_name.lower().replace(" ", "_")
    filename = f"{model_token}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.jpg"
    save_path = output_dir / filename
    annotated_image.convert("RGB").save(save_path, format="JPEG", quality=90, optimize=True)
    return f"/static/inference_results/{filename}"


def _build_report_detail_payload(
    model: AIModel,
    items: list[dict],
    confidence: float,
    temperature_threshold: float | None = None,
    selected_model: str | None = None,
    weights_path: str | None = None,
) -> tuple[str, dict]:
    model_name_lc = (model.name or "").lower()
    if _is_infrared_model(model):
        model_family = "infrared"
    elif _is_oil_leak_model(model):
        model_family = "oil_leak"
    elif _is_bird_nest_model(model):
        model_family = "bird_nest"
    else:
        model_family = model_name_lc.replace(" ", "_")

    normalized_items: list[dict] = []
    total_detections = 0
    abnormal_images = 0
    error_images = 0
    label_totals: dict[str, int] = {}
    max_score_overall = 0.0
    infrared_max_read_overall = 0.0
    infrared_max_hotspot_overall = 0.0
    infrared_hotspot_temps_all: list[float] = []

    for item in items:
        filename = item.get("filename")
        if item.get("error"):
            error_images += 1
            normalized_items.append(
                {"filename": filename, "status": "error", "error": item.get("error")}
            )
            continue

        summary = item.get("summary", {}) or {}
        detections = int(summary.get("detections", 0) or 0)
        max_score = float(summary.get("max_score", 0.0) or 0.0)
        avg_score = float(summary.get("avg_score", 0.0) or 0.0)
        labels = summary.get("labels", {}) or {}
        predictions = item.get("predictions", []) or []
        hotspot_temps = [
            float(p.get("temperature"))
            for p in predictions
            if str(p.get("label", "")).lower() == "hotspot" and p.get("temperature") is not None
        ]

        total_detections += detections
        if detections > 0:
            abnormal_images += 1
        max_score_overall = max(max_score_overall, max_score)

        for k, v in labels.items():
            label_totals[k] = label_totals.get(k, 0) + int(v)

        if model_family == "infrared":
            max_read = float(summary.get("max_temperature_read", 0.0) or 0.0)
            max_hotspot = float(summary.get("max_hotspot_temperature", 0.0) or 0.0)
            infrared_max_read_overall = max(infrared_max_read_overall, max_read)
            infrared_max_hotspot_overall = max(infrared_max_hotspot_overall, max_hotspot)
            infrared_hotspot_temps_all.extend(hotspot_temps)

        normalized_items.append(
            {
                "filename": filename,
                "status": "ok",
                "detections": detections,
                "max_score": max_score,
                "avg_score": avg_score,
                "labels": labels,
                "hotspot_temperatures": hotspot_temps,
                "max_temperature_read": float(summary.get("max_temperature_read", 0.0) or 0.0),
                "max_hotspot_temperature": float(summary.get("max_hotspot_temperature", 0.0) or 0.0),
            }
        )

    total_images = len(items)
    avg_detections_per_image = (total_detections / total_images) if total_images > 0 else 0.0
    abnormal_ratio = (abnormal_images / total_images) if total_images > 0 else 0.0
    detail_payload = {
        "selected_model": selected_model,
        "model_id": model.id,
        "model_name": model.name,
        "model_type": model.type,
        "model_family": model_family,
        "weights_path": weights_path,
        "confidence": confidence,
        "temperature_threshold": temperature_threshold,
        "total_images": total_images,
        "processed_images": total_images - error_images,
        "error_images": error_images,
        "abnormal_images": abnormal_images,
        "abnormal_ratio": abnormal_ratio,
        "total_detections": total_detections,
        "avg_detections_per_image": avg_detections_per_image,
        "max_score_overall": max_score_overall,
        "label_totals": label_totals,
        "infrared": {
            "max_temperature_read_overall": infrared_max_read_overall,
            "max_hotspot_temperature_overall": infrared_max_hotspot_overall,
            "avg_hotspot_temperature": (
                sum(infrared_hotspot_temps_all) / len(infrared_hotspot_temps_all)
                if infrared_hotspot_temps_all
                else 0.0
            ),
        },
        "items": normalized_items,
    }
    summary_text = (
        f"{total_images} images, {abnormal_images} abnormal, total {total_detections} detections"
    )
    return summary_text, detail_payload


def _get_runner_for_model(model: AIModel):
    weights_path = _resolve_weights_path(model)
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")

    if _is_bird_nest_model(model):
        return get_model_service(
            model_path=str(weights_path),
            confidence=current_app.config["CONFIDENCE_THRESHOLD"],
        )
    if _is_oil_leak_model(model):
        return get_yolo_service(
            model_path=str(weights_path),
            confidence=current_app.config["CONFIDENCE_THRESHOLD"],
        )
    if _is_infrared_model(model):
        raise ValueError("infrared model is handled by hybrid detection route")
    raise ValueError(f"unsupported model: name={model.name}, type={model.type}")


@inference_bp.get("/history")
def get_inference_history():
    user_id = session.get("user_id")
    tenant_id = session.get("tenant_id")
    if not user_id or not tenant_id:
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
    tenant_id = session.get("tenant_id")
    if not user_id or not tenant_id:
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


def _predict_with_fallback(model: AIModel, pil_image: Image.Image, threshold: float):
    runner = _get_runner_for_model(model)
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
    annotated_url = _save_annotated_image(annotated, model.name)
    scores = [pred.get("score", 0.0) for pred in predictions]
    labels = [pred.get("label", "unknown") for pred in predictions]
    label_distribution: dict[str, int] = {}
    for label in labels:
        label_distribution[label] = label_distribution.get(label, 0) + 1

    return {
        "predictions": predictions,
        "annotated_image": annotated_b64,
        "annotated_image_url": annotated_url,
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
    tenant_id = session.get("tenant_id")
    if not user_id or not tenant_id:
        return jsonify({"error": "authentication required"}), 401

    model_id = request.form.get("model_id", type=int)
    selected_model = request.form.get("selected_model")
    confidence = (
        request.form.get("confidence", type=float)
        or request.args.get("confidence", type=float)
    )
    temperature_threshold = (
        request.form.get("temperature_threshold", type=float)
        or request.args.get("temperature_threshold", type=float)
        or 50.0
    )
    disable_suppression = request.form.get("disable_suppression") == "true" or request.args.get("disable_suppression") == "true"
    images = request.files.getlist("images")
    if not images:
        return jsonify({"error": "at least one image is required"}), 400

    if not model_id and selected_model:
        if selected_model == "bird_nest_real":
            matched = AIModel.query.filter_by(name=BIRD_NEST_MODEL_NAME).first()
            model_id = matched.id if matched else None
        elif selected_model == "oil_leak_real":
            matched = AIModel.query.filter_by(name=OIL_LEAK_MODEL_NAME).first()
            model_id = matched.id if matched else None

    if not model_id:
        return jsonify({"error": "model_id is required"}), 400

    model = AIModel.query.get(model_id)
    if not model:
        return jsonify({"error": "model not found"}), 404
    if model.status != "Active":
        return jsonify({"error": "selected model is inactive"}), 400
    if confidence is None:
        confidence = _default_confidence_for_model(model)

    if _is_infrared_model(model):
        weights_path = Path("hybrid://infrared")
    else:
        try:
            weights_path = _resolve_weights_path(model)
            _get_runner_for_model(model)
        except (FileNotFoundError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    report_items: list[dict] = []
    total_detections = 0
    model_name = model.name
    for idx, image_file in enumerate(images):
        if _is_infrared_model(model):
            suffix = Path(image_file.filename or "image.jpg").suffix or ".jpg"
            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    image_file.save(tmp_file.name)
                    tmp_path = tmp_file.name
                item_result = run_infrared_hybrid_detection(
                    image_path=tmp_path, alarm_temp=float(temperature_threshold), disable_suppression=disable_suppression
                )
            except Exception as exc:  # pragma: no cover
                current_app.logger.exception("Failed to parse/process infrared image")
                report_items.append(
                    {
                        "filename": image_file.filename or f"image_{idx + 1}",
                        "error": f"infrared detection failed: {exc}",
                    }
                )
                continue
            finally:
                if tmp_path:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except Exception:
                        pass
        else:
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

            item_result = _predict_with_fallback(model, pil_image, confidence)
        item_result["filename"] = image_file.filename or f"image_{idx + 1}"
        report_items.append(item_result)
        total_detections += item_result["summary"]["detections"]

        task_summary = (
            f"{item_result['summary']['detections']} detections by {model.name}"
        )
        db.session.add(
            InferenceTask(tenant_id=tenant_id, user_id=user_id, model_id=model.id, result_summary=task_summary)
        )

    summary, detail_payload = _build_report_detail_payload(
        model=model,
        items=report_items,
        confidence=float(confidence),
        temperature_threshold=float(temperature_threshold),
        selected_model=selected_model,
        weights_path=str(weights_path),
    )
    report = InferenceReport(
        tenant_id=tenant_id,
        user_id=user_id,
        model_name=model_name,
        summary=summary,
        detail_json=json.dumps(detail_payload),
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
            "model": {"id": model.id, "name": model.name, "type": model.type},
            "items": report_items,
        }
    )


@inference_bp.post("/reports")
def create_inference_report_from_payload():
    user_id = session.get("user_id")
    tenant_id = session.get("tenant_id")
    if not user_id or not tenant_id:
        return jsonify({"error": "authentication required"}), 401

    payload = request.get_json(silent=True) or {}
    model_id = payload.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400

    model = AIModel.query.get(model_id)
    if not model:
        return jsonify({"error": "model not found"}), 404

    items = payload.get("items")
    if not isinstance(items, list) or len(items) == 0:
        return jsonify({"error": "items is required"}), 400

    confidence = float(payload.get("confidence", 0.5))
    temperature_threshold = payload.get("temperature_threshold")
    temperature_threshold = (
        float(temperature_threshold) if temperature_threshold is not None else None
    )

    summary, detail_payload = _build_report_detail_payload(
        model=model,
        items=items,
        confidence=confidence,
        temperature_threshold=temperature_threshold,
        selected_model=payload.get("selected_model"),
        weights_path=str(_resolve_weights_path(model)) if not _is_infrared_model(model) else "hybrid://infrared",
    )
    report = InferenceReport(
        tenant_id=tenant_id,
        user_id=user_id,
        model_name=model.name,
        summary=summary,
        detail_json=json.dumps(detail_payload),
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
            }
        }
    )


@inference_bp.post("/tasks")
def create_inference_task():
    user_id = session.get("user_id")
    tenant_id = session.get("tenant_id")
    if not user_id or not tenant_id:
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
    
    image_file = request.files["image"]
    image_bytes = image_file.read()
    filename = image_file.filename or "image.jpg"
    
    temperature_threshold = (
        request.form.get("temperature_threshold", type=float)
        or request.args.get("temperature_threshold", type=float)
        or 50.0
    )
    confidence = (
        request.form.get("confidence", type=float)
        or request.args.get("confidence", type=float)
    )
    disable_suppression = request.form.get("disable_suppression") == "true" or request.args.get("disable_suppression") == "true"
    threshold = confidence if confidence is not None else _default_confidence_for_model(model)

    task = InferenceTask(
        tenant_id=tenant_id,
        user_id=user_id,
        model_id=model.id,
        status="PENDING"
    )
    db.session.add(task)
    db.session.commit()

    # Dispatch to background thread
    app = current_app._get_current_object()
    executor.submit(
        _background_inference_worker,
        app, task.id, model.id, image_bytes, filename, threshold, temperature_threshold, disable_suppression
    )

    return jsonify({
        "task": {
            "id": task.id,
            "status": task.status
        }
    }), 202

def _background_inference_worker(app, task_id, model_id, image_bytes, filename, threshold, temperature_threshold, disable_suppression=False):
    with app.app_context():
        try:
            task = InferenceTask.query.get(task_id)
            if not task:
                return
            model = AIModel.query.get(model_id)
            
            if _is_infrared_model(model):
                weights_path = Path("hybrid://infrared")
                suffix = Path(filename).suffix or ".jpg"
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(image_bytes)
                        tmp_path = tmp_file.name
                    result = run_infrared_hybrid_detection(
                        image_path=tmp_path, alarm_temp=float(temperature_threshold), disable_suppression=disable_suppression
                    )
                finally:
                    if tmp_path:
                        try:
                            Path(tmp_path).unlink(missing_ok=True)
                        except Exception:
                            pass
            else:
                weights_path = _resolve_weights_path(model)
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                result = _predict_with_fallback(model, pil_image, threshold)

            result_payload = {
                "predictions": result["predictions"],
                "annotated_image": result["annotated_image"],
                "annotated_image_url": result["annotated_image_url"],
                "confidence_threshold": result["confidence_threshold"],
                "summary": result["summary"],
                "weights_path": str(weights_path),
                "fallback": result["fallback"],
            }
            task.result_summary = json.dumps(result_payload)
            task.status = "SUCCESS"
            db.session.commit()
            
        except Exception as exc:
            app.logger.exception("Background task failed")
            task = InferenceTask.query.get(task_id)
            if task:
                task.status = "FAILED"
                task.result_summary = str(exc)
                db.session.commit()

@inference_bp.get("/tasks/<int:task_id>/status")
def get_task_status(task_id):
    user_id = session.get("user_id")
    tenant_id = session.get("tenant_id")
    if not user_id or not tenant_id:
        return jsonify({"error": "authentication required"}), 401
        
    task = InferenceTask.query.filter_by(id=task_id, tenant_id=tenant_id).first()
    if not task:
        return jsonify({"error": "task not found"}), 404
        
    response_data = {
        "task": {
            "id": task.id,
            "status": task.status
        }
    }
    
    if task.status == "SUCCESS":
        try:
            result_payload = json.loads(task.result_summary)
            response_data.update(result_payload)
        except Exception:
            pass
    elif task.status == "FAILED":
        response_data["error"] = task.result_summary
        
    return jsonify(response_data)
