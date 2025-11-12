from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Sequence

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import nms


class MaskRCNNService:
    """Wrapper around Mask R-CNN for bird-nest segmentation."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: Sequence[str],
        target_size: int = 512,
        confidence: float = 0.5,
        color_map_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_names = list(class_names)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.confidence = confidence
        self.target_size = target_size
        self.color_map = self._load_color_map(color_map_path)
        self.font = self._load_font()

        self.model = self._load_model()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize([self.target_size, self.target_size], antialias=True),
                transforms.ToImage(),
                transforms.ToDtype(self.dtype, scale=True),
            ]
        )

    @staticmethod
    def _load_color_map(color_map_path: str | Path | None):
        if color_map_path and Path(color_map_path).exists():
            with open(color_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("items", [])
            return {item["label"]: tuple(int(c * 255) if isinstance(c, float) else int(c) for c in item["color"][:3]) for item in items}
        # Fallback palette
        return {
            "background": (120, 120, 120),
            "bird_nest": (220, 20, 60),
        }

    @staticmethod
    def _load_font():
        try:
            return ImageFont.truetype("arial.ttf", size=18)
        except Exception:  # pragma: no cover
            return ImageFont.load_default()

    def _load_model(self):
        model = maskrcnn_resnet50_fpn_v2(weights=None)
        num_classes = len(self.class_names)

        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features_box, num_classes=num_classes
        )
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, dim_reduced, num_classes=num_classes
        )

        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(device=self.device, dtype=self.dtype)
        model.eval()
        return model

    def predict(self, image: Image.Image, confidence: float | None = None):
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
        threshold = confidence if confidence is not None else self.confidence
        predictions = self._postprocess(outputs, threshold, original_size=image.size)
        annotated_image = self._annotate(image, predictions)
        return predictions, annotated_image

    def _postprocess(self, outputs, threshold: float, original_size: tuple[int, int]):
        out = outputs[0]
        scores = out["scores"].detach().cpu()
        keep = scores >= threshold

        if keep.sum() == 0:
            return []

        boxes = out["boxes"][keep].detach().cpu()
        labels = out["labels"][keep].detach().cpu()
        scores = scores[keep]

        keep_idx = nms(boxes, scores, iou_threshold=0.5)
        boxes = boxes[keep_idx]
        labels = labels[keep_idx]
        scores = scores[keep_idx]

        orig_width, orig_height = original_size
        scale_x = orig_width / self.target_size
        scale_y = orig_height / self.target_size

        boxes = boxes.clone()
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        predictions = []
        for box, label, score in zip(boxes, labels, scores):
            label_idx = int(label.item())
            label_name = (
                self.class_names[label_idx]
                if label_idx < len(self.class_names)
                else f"class_{label_idx}"
            )
            predictions.append(
                {
                    "label": label_name,
                    "score": round(float(score.item()), 4),
                    "bbox": [round(float(coord), 2) for coord in box.tolist()],
                }
            )
        return predictions

    def _annotate(self, image: Image.Image, predictions: list[dict]):
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated, mode="RGBA")
        for pred in predictions:
            bbox = pred["bbox"]
            label = pred["label"]
            score = pred["score"]
            color = self.color_map.get(label, (46, 204, 113))
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text = f"{label} ({score:.2f})"
            text_size = draw.textbbox((x1, y1), text, font=self.font)
            draw.rectangle(text_size, fill=(*color, 160))
            draw.text((x1 + 2, y1 + 2), text, fill=(255, 255, 255), font=self.font)
        return annotated

    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_class_names(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


