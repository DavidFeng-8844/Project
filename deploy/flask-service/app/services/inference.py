from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from PIL import Image, ImageFont
from ultralytics import YOLO
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import nms
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.tv_tensors import Mask, BoundingBoxes


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
        predictions, masks, boxes = self._postprocess(outputs, threshold, original_size=image.size)
        annotated_image = self._annotate(image, predictions, masks, boxes)
        return predictions, annotated_image

    def _postprocess(self, outputs, threshold: float, original_size: tuple[int, int]):
        out = outputs[0]
        scores = out["scores"].detach().cpu()
        keep = scores >= threshold

        if keep.sum() == 0:
            return [], None, None

        boxes = out["boxes"][keep].detach().cpu()
        labels = out["labels"][keep].detach().cpu()
        scores = scores[keep]
        masks = out["masks"][keep].detach().cpu()

        keep_idx = nms(boxes, scores, iou_threshold=0.5)
        boxes = boxes[keep_idx]
        labels = labels[keep_idx]
        scores = scores[keep_idx]
        masks = masks[keep_idx]

        orig_width, orig_height = original_size
        scale_x = orig_width / self.target_size
        scale_y = orig_height / self.target_size

        # Scale boxes to original image size
        boxes_scaled = boxes.clone()
        boxes_scaled[:, [0, 2]] *= scale_x
        boxes_scaled[:, [1, 3]] *= scale_y

        # Scale masks to original image size
        if masks.numel() > 0:
            masks_scaled = F.interpolate(
                masks,
                size=(orig_height, orig_width),
                mode='bilinear',
                align_corners=False
            )
            # Apply sigmoid if needed (some torchvision versions output logits)
            if masks_scaled.max().item() > 1.0 or masks_scaled.min().item() < 0.0:
                masks_scaled = masks_scaled.sigmoid()
            # Convert to boolean masks (threshold at 0.5)
            masks_bool = torch.where(masks_scaled.squeeze(1) >= 0.5, True, False)
            masks_tv = Mask(masks_bool)
        else:
            masks_tv = None

        # Create BoundingBoxes tensor
        boxes_tv = BoundingBoxes(
            boxes_scaled,
            format='xyxy',
            canvas_size=(orig_height, orig_width)
        )

        predictions = []
        for box, label, score in zip(boxes_scaled, labels, scores):
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
        return predictions, masks_tv, boxes_tv

    def _annotate(self, image: Image.Image, predictions: list[dict], masks: Mask | None, boxes: BoundingBoxes | None):
        # Limit image size to prevent memory issues (max 2048px on longest side)
        MAX_DISPLAY_SIZE = 2048
        original_size = image.size
        scale_factor = 1.0
        
        if max(original_size) > MAX_DISPLAY_SIZE:
            scale_factor = MAX_DISPLAY_SIZE / max(original_size)
            new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            # Scale boxes and masks if they exist
            if boxes is not None and len(boxes) > 0:
                boxes = BoundingBoxes(
                    boxes.data * scale_factor,
                    format=boxes.format,
                    canvas_size=(new_size[1], new_size[0])
                )
            if masks is not None and len(masks) > 0:
                # Resize masks
                masks_resized = F.interpolate(
                    masks.data.unsqueeze(0).float(),
                    size=(new_size[1], new_size[0]),
                    mode='nearest'
                ).squeeze(0).bool()
                masks = Mask(masks_resized)
        
        # Convert PIL image to tensor
        img_tensor = transforms.PILToTensor()(image)
        
        # Draw masks first (red segmentation regions)
        if masks is not None and len(masks) > 0:
            # Get colors for each prediction
            colors = []
            for pred in predictions:
                label = pred["label"]
                color = self.color_map.get(label, (220, 20, 60))  # Default to red for bird_nest
                colors.append(color)
            
            # Draw segmentation masks with transparency
            img_tensor = draw_segmentation_masks(
                image=img_tensor,
                masks=masks,
                alpha=0.3,
                colors=colors
            )
        
        # Draw bounding boxes and labels
        if boxes is not None and len(boxes) > 0:
            labels_with_scores = [
                f"{pred['label']} ({pred['score']:.2f})"
                for pred in predictions
            ]
            colors = [
                self.color_map.get(pred["label"], (220, 20, 60))
                for pred in predictions
            ]
            
            img_tensor = draw_bounding_boxes(
                image=img_tensor,
                boxes=boxes,
                labels=labels_with_scores,
                colors=colors,
                fill=False,
                width=2
            )
        
        # Convert tensor back to PIL Image
        annotated = transforms.ToPILImage()(img_tensor)
        return annotated

    @staticmethod
    def encode_image_to_base64(image: Image.Image, max_size: int = 2048, quality: int = 85) -> str:
        """Encode image to base64 with size optimization."""
        # Resize if too large
        if max(image.size) > max_size:
            scale = max_size / max(image.size)
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        # Use JPEG for better compression if image doesn't have transparency
        if image.mode in ('RGBA', 'LA', 'P'):
            image.save(buffer, format="PNG", optimize=True)
        else:
            # Convert to RGB if needed and save as JPEG
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_class_names(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class YOLOService:
    """Wrapper around YOLOv8 for oil-leak object detection."""

    def __init__(
        self,
        model_path: str | Path,
        class_name: str = "oil_leak",
        confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_name = class_name
        self.confidence = confidence
        self.model = YOLO(str(self.model_path))

    def predict(self, image: Image.Image, confidence: float | None = None):
        threshold = confidence if confidence is not None else self.confidence
        image_np = np.array(image.convert("RGB"))
        results = self.model.predict(source=image_np, conf=threshold, verbose=False)

        if not results:
            return [], image

        result = results[0]
        predictions: list[dict] = []

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            for box, conf in zip(xyxy, confs):
                predictions.append(
                    {
                        "label": self.class_name,
                        "score": round(float(conf), 4),
                        "bbox": [round(float(coord), 2) for coord in box.tolist()],
                    }
                )

        # Use YOLO built-in plotting for robust visualization.
        plotted = result.plot()
        annotated = Image.fromarray(plotted[..., ::-1])  # BGR -> RGB
        return predictions, annotated

    @staticmethod
    def encode_image_to_base64(image: Image.Image, max_size: int = 2048, quality: int = 85) -> str:
        """Encode image to base64 with size optimization."""
        if max(image.size) > max_size:
            scale = max_size / max(image.size)
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        if image.mode in ("RGBA", "LA", "P"):
            image.save(buffer, format="PNG", optimize=True)
        else:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

