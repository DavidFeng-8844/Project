from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request
from PIL import Image

from .services.model_service import get_model_service

api_bp = Blueprint("api", __name__)


@api_bp.route("/", methods=["GET", "POST"])
def index():
    predictions: list[dict] | None = None
    annotated_image_b64 = None
    error = None
    service = get_model_service()
    default_conf = service.confidence
    selected_confidence = request.form.get("confidence", type=float)

    if request.method == "POST":
        if "image" not in request.files:
            error = "No image file provided."
        else:
            image_file = request.files["image"]
            try:
                pil_image = Image.open(image_file.stream).convert("RGB")
                predictions, annotated = service.predict(
                    pil_image, confidence=selected_confidence
                )
                annotated_image_b64 = service.encode_image_to_base64(annotated)
            except Exception as exc:  # pylint: disable=broad-except
                current_app.logger.exception("Prediction failed")
                error = f"Prediction failed: {exc}"

    no_detections = predictions is not None and len(predictions) == 0

    return render_template(
        "index.html",
        predictions=predictions,
        annotated_image_b64=annotated_image_b64,
        error=error,
        default_confidence=default_conf,
        selected_confidence=selected_confidence or default_conf,
        no_detections=no_detections,
        effective_confidence=selected_confidence or default_conf,
    )


@api_bp.route("/health", methods=["GET"])
def health():
    service = get_model_service()
    return jsonify(
        {
            "status": "ok",
            "device": str(service.device),
            "classes": service.class_names,
            "confidence_threshold": service.confidence,
        }
    )


@api_bp.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image file provided"}), 400

    image_file = request.files["image"]
    try:
        pil_image = Image.open(image_file.stream).convert("RGB")
    except Exception as exc:  # pragma: no cover
        current_app.logger.exception("Failed to read uploaded image")
        return jsonify({"error": f"invalid image: {exc}"}), 400

    service = get_model_service()
    confidence = (
        request.form.get("confidence", type=float)
        or request.args.get("confidence", type=float)
    )
    predictions, annotated = service.predict(pil_image, confidence=confidence)
    annotated_b64 = service.encode_image_to_base64(annotated)
    threshold = confidence if confidence is not None else service.confidence
    message = None
    if len(predictions) == 0:
        message = (
            "No detections were found above the confidence threshold "
            f"{threshold:.2f}."
        )

    return jsonify(
        {
            "predictions": predictions,
            "annotated_image": annotated_b64,
            "confidence_threshold": threshold,
            "message": message,
        }
    )

