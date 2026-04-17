from __future__ import annotations

import base64
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING
from uuid import uuid4

import cv2
import easyocr
import numpy as np
from flask import current_app

if TYPE_CHECKING:
    from .inference import MaskRCNNService, YOLOService


try:
    # Prefer GPU OCR; fallback to CPU when unavailable.
    reader = easyocr.Reader(["en"], gpu=True)
except Exception:  # pragma: no cover
    reader = easyocr.Reader(["en"], gpu=False)


def _normalize_temperature_candidate(token: str) -> float | None:
    """
    Convert OCR token into plausible temperature value.
    Fixes common OCR issue like '41.1' -> '411' by scaling down large numbers.
    """
    cleaned = token.strip()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None

    # Heuristic: temperatures beyond realistic infrared range likely missed decimal points.
    while value > 200.0:
        value /= 10.0

    if value < -50.0 or value > 200.0:
        return None
    return value


def _extract_temperature_range_and_text_mask(
    image_path: str, image_shape: tuple[int, int]
) -> tuple[float, float, np.ndarray]:
    ocr_results = reader.readtext(image_path, detail=1)
    numbers: list[float] = []
    text_mask = np.zeros(image_shape, dtype=np.uint8)
    h, w = image_shape
    numeric_boxes: list[dict[str, Any]] = []

    for item in ocr_results:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        bbox = item[0]
        text_str = str(item[1])

        # Build text region mask so labels are not treated as hotspots.
        try:
            points = np.array([[int(p[0]), int(p[1])] for p in bbox], dtype=np.int32)
            if points.shape[0] >= 3:
                cv2.fillPoly(text_mask, [points], 255)
        except Exception:
            pass

        matches = re.findall(r"\d+(?:\.\d+)?", text_str)
        normalized_vals: list[float] = []
        for match in matches:
            normalized = _normalize_temperature_candidate(match)
            if normalized is not None:
                numbers.append(normalized)
                normalized_vals.append(normalized)

        if normalized_vals:
            try:
                xs = [int(p[0]) for p in bbox]
                ys = [int(p[1]) for p in bbox]
                x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
                y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                numeric_boxes.append(
                    {
                        "x1": x1,
                        "x2": x2,
                        "y1": y1,
                        "y2": y2,
                        "cx": cx,
                        "cy": cy,
                        "values": normalized_vals,
                    }
                )
            except Exception:
                pass

    # Prefer right-side top/bottom legend values as max/min temperature scale.
    right_candidates = [b for b in numeric_boxes if b["cx"] >= w * 0.68]
    if len(right_candidates) >= 2:
        top_box = min(right_candidates, key=lambda b: b["cy"])
        bottom_box = max(right_candidates, key=lambda b: b["cy"])
        top_val = max(top_box["values"]) if top_box["values"] else None
        bottom_val = min(bottom_box["values"]) if bottom_box["values"] else None

        if top_val is not None and bottom_val is not None:
            # Mask the right-side max/min numeric boxes with padding, including white outline boxes.
            for box in (top_box, bottom_box):
                pad_x = max(6, int((box["x2"] - box["x1"]) * 0.35))
                pad_y = max(4, int((box["y2"] - box["y1"]) * 0.45))
                bx1 = max(0, box["x1"] - pad_x)
                bx2 = min(w - 1, box["x2"] + pad_x)
                by1 = max(0, box["y1"] - pad_y)
                by2 = min(h - 1, box["y2"] + pad_y)
                text_mask[by1:by2 + 1, bx1:bx2 + 1] = 255

            temp_high = max(float(top_val), float(bottom_val))
            temp_low = min(float(top_val), float(bottom_val))
            if temp_high <= temp_low:
                temp_low = max(0.0, temp_high - 1.0)

            # Mask out the narrow right-side color indicator bar between max/min labels.
            y_start = max(0, min(top_box["y2"], bottom_box["y2"]) + 1)
            y_end = min(h - 1, max(top_box["y1"], bottom_box["y1"]) - 1)
            if y_end > y_start:
                digit_w = max(2, int(max(top_box["x2"] - top_box["x1"], bottom_box["x2"] - bottom_box["x1"]) * 0.16))
                bar_x_center = int(max(top_box["x2"], bottom_box["x2"]) + digit_w * 0.9)
                bar_x1 = max(0, bar_x_center - digit_w)
                bar_x2 = min(w - 1, bar_x_center + digit_w)
                text_mask[y_start:y_end + 1, bar_x1:bar_x2 + 1] = 255

            # Expand mask slightly to cover anti-aliased white borders and halos.
            dilate_k = max(3, int(round(min(h, w) * 0.006)))
            if dilate_k % 2 == 0:
                dilate_k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
            text_mask = cv2.dilate(text_mask, kernel, iterations=1)

            return temp_high, temp_low, text_mask

    if not numbers:
        # Even without reliable temperatures, still expand mask to remove OCR white borders.
        dilate_k = max(3, int(round(min(h, w) * 0.006)))
        if dilate_k % 2 == 0:
            dilate_k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)
        return 80.0, 20.0, text_mask

    temp_high = max(numbers)
    temp_low = min(numbers)
    if temp_high <= temp_low:
        temp_low = max(0.0, temp_high - 1.0)
    dilate_k = max(3, int(round(min(h, w) * 0.006)))
    if dilate_k % 2 == 0:
        dilate_k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)
    return temp_high, temp_low, text_mask


def _save_annotated_bgr_image(annotated_bgr: np.ndarray, model_token: str) -> str:
    static_root = Path(current_app.static_folder)
    output_dir = static_root / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{model_token}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.jpg"
    save_path = output_dir / filename
    cv2.imwrite(str(save_path), annotated_bgr)
    return f"/static/inference_results/{filename}"


def _encode_bgr_to_base64(image_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError("failed to encode annotated infrared image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _build_border_watermark_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Mask bright white watermark/logo overlays near image borders (position-agnostic).
    This avoids treating white text/logo overlays as thermal hotspots.
    """
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # White-ish overlay: high brightness + low saturation.
    candidate = cv2.inRange(hsv, (0, 0, 205), (180, 70, 255))

    k = max(3, int(round(min(h, w) * 0.01)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_margin_x = int(w * 0.12)
    border_margin_y = int(h * 0.12)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area < int(h * w * 0.0005):
            continue
        if area > int(h * w * 0.18):
            continue

        # Keep only components close to image border.
        touches_border = x <= 1 or y <= 1 or (x + cw) >= (w - 2) or (y + ch) >= (h - 2)
        near_border = (
            x <= border_margin_x
            or y <= border_margin_y
            or (x + cw) >= (w - border_margin_x)
            or (y + ch) >= (h - border_margin_y)
        )
        if not touches_border and not near_border:
            continue

        pad_x = max(4, int(cw * 0.12))
        pad_y = max(4, int(ch * 0.12))
        x1g = max(0, x - pad_x)
        y1g = max(0, y - pad_y)
        x2g = min(w - 1, x + cw + pad_x)
        y2g = min(h - 1, y + ch + pad_y)
        mask[y1g:y2g + 1, x1g:x2g + 1] = 255

    return mask


def run_infrared_hybrid_detection(
    image_path: str | Path, alarm_temp: float = 50.0
) -> dict[str, Any]:
    """
    Infrared hybrid detection based on OCR-extracted temperature scale + pixel mapping.
    Returns the same top-level structure as existing SaaS inference output.
    """
    source_path = str(image_path)
    image_bgr = cv2.imread(source_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"unable to read image: {source_path}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    temp_high, temp_low, text_mask = _extract_temperature_range_and_text_mask(
        source_path, gray.shape
    )
    watermark_mask = _build_border_watermark_mask(image_bgr)
    annotated = image_bgr.copy()
    predictions: list[dict[str, Any]] = []
    hotspot_temperatures: list[float] = []

    pixel_threshold = 255
    if temp_high >= alarm_temp:
        denom = max(temp_high - temp_low, 1e-6)
        pixel_threshold = int((alarm_temp - temp_low) / denom * 255.0)
        pixel_threshold = int(np.clip(pixel_threshold, 0, 255))

        _, binary = cv2.threshold(gray, pixel_threshold, 255, cv2.THRESH_BINARY)
        # Remove OCR text/legend regions to prevent white digits from false positives.
        if text_mask is not None and text_mask.any():
            binary[text_mask > 0] = 0
        if watermark_mask is not None and watermark_mask.any():
            binary[watermark_mask > 0] = 0
        img_h, img_w = gray.shape
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = annotated.copy()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 25 or (w >= img_w * 0.4 and h >= img_h * 0.4):
                continue

            x2 = x + w
            y2 = y + h
            roi = gray[y:y2, x:x2]
            hotspot_pixel = int(np.max(roi)) if roi.size > 0 else pixel_threshold
            hotspot_temp = temp_low + (hotspot_pixel / 255.0) * (temp_high - temp_low)
            hotspot_temp = float(np.clip(hotspot_temp, temp_low, temp_high))
            hotspot_temperatures.append(hotspot_temp)
            predictions.append(
                {
                    "bbox": [float(x), float(y), float(x2), float(y2)],
                    "score": float(hotspot_temp),
                    "label": "hotspot",
                    "temperature": round(float(hotspot_temp), 2),
                    "area_pixels": int(cv2.contourArea(contour)),
                }
            )

            cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 255), 2)
            cv2.fillPoly(overlay, [contour], (0, 0, 255))
            cv2.putText(
                overlay,
                f"{hotspot_temp:.1f} C",
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0, annotated)

    detections = len(predictions)
    labels = {"hotspot": detections} if detections > 0 else {}
    scores = [float(pred["score"]) for pred in predictions]
    max_hotspot_temperature = max(hotspot_temperatures) if hotspot_temperatures else 0.0

    annotated_image_url = _save_annotated_bgr_image(annotated, "infrared_hybrid")
    annotated_image_b64 = _encode_bgr_to_base64(annotated)

    return {
        "predictions": predictions,
        "annotated_image": annotated_image_b64,
        "annotated_image_url": annotated_image_url,
        "confidence_threshold": float(alarm_temp),
        "summary": {
            "detections": detections,
            "max_score": max(scores) if scores else 0.0,
            "avg_score": (sum(scores) / len(scores)) if scores else 0.0,
            "labels": labels,
            "max_temperature_read": float(temp_high),
            "max_hotspot_temperature": float(max_hotspot_temperature),
            "applied_threshold": float(alarm_temp),
        },
        "fallback": {
            "attempted": False,
            "applied": False,
            "original_threshold": float(alarm_temp),
            "fallback_threshold": float(alarm_temp),
        },
    }


@lru_cache(maxsize=4)
def get_model_service(
    model_path: str | None = None,
    confidence: float | None = None,
) -> "MaskRCNNService":
    # Lazy import to avoid heavy torch/ultralytics import during app boot.
    from .inference import MaskRCNNService, load_class_names

    config = current_app.config
    class_names = load_class_names(config["CLASS_NAMES_PATH"])
    service = MaskRCNNService(
        model_path=model_path or config["MODEL_PATH"],
        class_names=class_names,
        target_size=config["TARGET_SIZE"],
        confidence=confidence if confidence is not None else config["CONFIDENCE_THRESHOLD"],
        color_map_path=config.get("COLOR_MAP_PATH"),
    )
    return service


@lru_cache(maxsize=4)
def get_yolo_service(
    model_path: str,
    confidence: float | None = None,
) -> "YOLOService":
    # Lazy import to avoid heavy torch/ultralytics import during app boot.
    from .inference import YOLOService

    config = current_app.config
    return YOLOService(
        model_path=model_path,
        class_name="oil_leak",
        confidence=confidence if confidence is not None else config["CONFIDENCE_THRESHOLD"],
    )

