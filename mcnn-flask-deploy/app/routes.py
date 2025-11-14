from __future__ import annotations

import base64
import io
import zipfile
from datetime import datetime

from flask import Blueprint, current_app, jsonify, render_template, request, send_file
from PIL import Image

from .services.model_service import get_model_service

api_bp = Blueprint("api", __name__)


@api_bp.route("/", methods=["GET", "POST"])
def index():
    results: list[dict] | None = None
    error = None
    service = get_model_service()
    default_conf = service.confidence
    selected_confidence = request.form.get("confidence", type=float)

    if request.method == "POST":
        if "images" not in request.files:
            error = "No image files provided."
        else:
            image_files = request.files.getlist("images")
            if not image_files or not any(f.filename for f in image_files):
                error = "No image files selected."
            else:
                results = []
                MAX_IMAGE_SIZE = 4096  # Maximum dimension in pixels
                MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size
                
                for idx, image_file in enumerate(image_files):
                    if not image_file.filename:
                        continue
                    try:
                        # Check file size
                        image_file.seek(0, 2)  # Seek to end
                        file_size = image_file.tell()
                        image_file.seek(0)  # Reset to beginning
                        
                        if file_size > MAX_FILE_SIZE:
                            results.append({
                                "filename": image_file.filename,
                                "index": idx,
                                "error": f"File too large ({file_size / 1024 / 1024:.1f}MB). Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB.",
                            })
                            continue
                        
                        pil_image = Image.open(image_file.stream).convert("RGB")
                        
                        # Check and limit image dimensions
                        original_img_size = pil_image.size
                        if max(original_img_size) > MAX_IMAGE_SIZE:
                            scale = MAX_IMAGE_SIZE / max(original_img_size)
                            new_size = (int(original_img_size[0] * scale), int(original_img_size[1] * scale))
                            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                            current_app.logger.info(f"Resized {image_file.filename} from {original_img_size} to {new_size}")
                        
                        predictions, annotated = service.predict(
                            pil_image, confidence=selected_confidence
                        )
                        annotated_image_b64 = service.encode_image_to_base64(annotated, max_size=2048)
                        results.append({
                            "filename": image_file.filename,
                            "index": idx,
                            "predictions": predictions,
                            "annotated_image_b64": annotated_image_b64,
                        })
                    except MemoryError:
                        current_app.logger.exception(f"Memory error for {image_file.filename}")
                        results.append({
                            "filename": image_file.filename,
                            "index": idx,
                            "error": "Image too large - memory error. Please resize the image and try again.",
                        })
                    except Exception as exc:  # pylint: disable=broad-except
                        current_app.logger.exception(f"Prediction failed for {image_file.filename}")
                        results.append({
                            "filename": image_file.filename,
                            "index": idx,
                            "error": f"Prediction failed: {exc}",
                        })

    # Calculate statistics for visualization
    stats = None
    if results:
        all_predictions = []
        for result in results:
            if not result.get("error") and result.get("predictions"):
                all_predictions.extend(result["predictions"])
        
        if all_predictions:
            scores = [p["score"] for p in all_predictions]
            stats = {
                "total_detections": len(all_predictions),
                "avg_confidence": sum(scores) / len(scores) if scores else 0,
                "max_confidence": max(scores) if scores else 0,
                "min_confidence": min(scores) if scores else 0,
                "images_with_detections": sum(1 for r in results if not r.get("error") and r.get("predictions")),
                "total_images": len(results),
                "confidence_distribution": _calculate_confidence_distribution(scores),
            }
        else:
            stats = {
                "total_detections": 0,
                "avg_confidence": 0,
                "max_confidence": 0,
                "min_confidence": 0,
                "images_with_detections": 0,
                "total_images": len(results),
                "confidence_distribution": [],
            }

    return render_template(
        "index.html",
        results=results,
        stats=stats,
        error=error,
        default_confidence=default_conf,
        selected_confidence=selected_confidence or default_conf,
        effective_confidence=selected_confidence or default_conf,
    )


def _calculate_confidence_distribution(scores: list[float], bins: int = 10) -> list[dict]:
    """Calculate confidence score distribution for histogram."""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    bin_width = (max_score - min_score) / bins if max_score > min_score else 1.0
    
    distribution = [0] * bins
    for score in scores:
        bin_idx = min(int((score - min_score) / bin_width), bins - 1) if bin_width > 0 else 0
        distribution[bin_idx] += 1
    
    # Create bins with labels
    result = []
    for i in range(bins):
        bin_start = min_score + i * bin_width
        bin_end = min_score + (i + 1) * bin_width
        result.append({
            "range": f"{bin_start:.2f}-{bin_end:.2f}",
            "count": distribution[i],
            "start": bin_start,
            "end": bin_end,
        })
    return result


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


@api_bp.route("/download/<int:result_index>", methods=["GET"])
def download_single(result_index: int):
    """Download a single annotated image by result index."""
    # This requires storing results in session or passing via form
    # For simplicity, we'll use a different approach with POST
    return jsonify({"error": "Use POST /download with result data"}), 400


@api_bp.route("/download_all", methods=["POST"])
def download_all():
    """Download all annotated images as a ZIP file."""
    try:
        # Use get_json with force=True to handle large payloads
        results_data = request.get_json(force=True, silent=True)
        if not results_data or "results" not in results_data:
            return jsonify({"error": "No results data provided"}), 400

        MAX_ZIP_SIZE = 100 * 1024 * 1024  # 100MB limit
        current_size = 0
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            filename_counts = {}  # Track filename usage to avoid conflicts
            valid_results = [r for r in results_data["results"] if "annotated_image_b64" in r]
            
            if len(valid_results) > 50:  # Limit number of files
                return jsonify({"error": "Too many files. Maximum 50 files per download."}), 400
            
            for result in valid_results:
                try:
                    # Decode base64 image
                    image_data = base64.b64decode(result["annotated_image_b64"])
                    
                    # Check size limit
                    if current_size + len(image_data) > MAX_ZIP_SIZE:
                        current_app.logger.warning(f"ZIP size limit reached, stopping at {len(valid_results)} files")
                        break
                    
                    original_filename = result.get("filename", f"image_{result.get('index', 0)}")
                    
                    # Extract base name and ensure proper extension
                    if "." in original_filename:
                        base_name = ".".join(original_filename.split(".")[:-1])
                        ext = original_filename.split(".")[-1].lower()
                        if ext not in ("png", "jpg", "jpeg"):
                            ext = "png"
                    else:
                        base_name = original_filename
                        ext = "png"
                    
                    # Handle filename conflicts
                    filename = f"{base_name}.{ext}"
                    if filename in filename_counts:
                        filename_counts[filename] += 1
                        filename = f"{base_name}_{filename_counts[filename]}.{ext}"
                    else:
                        filename_counts[filename] = 0
                    
                    zip_file.writestr(filename, image_data)
                    current_size += len(image_data)
                except Exception as e:
                    current_app.logger.warning(f"Failed to add {result.get('filename', 'unknown')} to ZIP: {e}")
                    continue

        zip_buffer.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"bird_nest_detections_{timestamp}.zip",
        )
    except MemoryError:
        current_app.logger.exception("Memory error creating ZIP file")
        return jsonify({"error": "ZIP file too large. Please download files individually."}), 500
    except Exception as exc:  # pylint: disable=broad-except
        current_app.logger.exception("Failed to create ZIP file")
        return jsonify({"error": f"Failed to create ZIP: {exc}"}), 500

