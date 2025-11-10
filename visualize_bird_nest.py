"""
Visualization script for bird nest detection using trained Mask R-CNN model
"""

import json
from pathlib import Path
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
import torchvision.transforms.v2 as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from distinctipy import distinctipy
import pandas as pd

from windows_utils import create_polygon_mask
from cjm_pytorch_utils.core import get_torch_device, tensor_to_pil, move_data_to_device
from cjm_pil_utils.core import resize_img, stack_imgs

# Set device
device = get_torch_device()
dtype = torch.float32
print(f"Using device: {device}")

# Paths
dataset_dir = Path("./Datasets/my_coco_dataset")
coco_json_path = dataset_dir / "dataset.json"
image_dir = dataset_dir

# Load COCO JSON
with open(coco_json_path, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

# Get class names
class_names = ['background'] + [cat['name'] for cat in coco_data['categories']]
print(f"Classes: {class_names}")

# Load color map if available, otherwise generate
color_map_path = None
# Try to find the most recent checkpoint directory
checkpoint_dirs = sorted(Path("./pytorch-mask-r-cnn-bird-nest").glob("*"), reverse=True)
if checkpoint_dirs:
    color_map_path = checkpoint_dirs[0] / "bird_nest-colormap.json"

if color_map_path and color_map_path.exists():
    with open(color_map_path, 'r') as f:
        color_map_data = json.load(f)
    colors = [item['color'] for item in color_map_data['items']]
    # Convert to RGB int tuples
    int_colors = []
    for color in colors:
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            # If colors are floats in [0,1], scale to [0,255]
            if all(isinstance(ch, float) and ch <= 1 for ch in color[:3]):
                int_colors.append(tuple(int(round(ch * 255)) for ch in color[:3]))
            else:
                int_colors.append(tuple(int(ch) for ch in color[:3]))
        else:
            int_colors.append((255, 0, 0))
else:
    # Generate colors as RGB int tuples
    colors = distinctipy.get_colors(len(class_names))
    int_colors = [tuple(int(round(ch * 255)) for ch in color[:3]) for color in colors]

# Load model
def load_model(checkpoint_path, num_classes, device):
    """Load trained model from checkpoint"""
    model = maskrcnn_resnet50_fpn_v2(weights=None)  # Don't load pretrained weights
    
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes=num_classes)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced=dim_reduced, 
                                                        num_classes=num_classes)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

# Find the most recent checkpoint
checkpoint_path = None
if checkpoint_dirs:
    for checkpoint_dir in checkpoint_dirs:
        possible_checkpoints = list(checkpoint_dir.glob("*.pth"))
        if possible_checkpoints:
            checkpoint_path = possible_checkpoints[0]
            break

if checkpoint_path and checkpoint_path.exists():
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, len(class_names), device)
else:
    print("Warning: No checkpoint found. Using pretrained model (will not work well for bird nest detection).")
    model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes=len(class_names))
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced=dim_reduced, 
                                                        num_classes=len(class_names))
    model.to(device)
    model.eval()

# Fixed colors for clarity
GT_COLOR = (0, 255, 0)      # Green for ground truth
PRED_COLOR = (255, 0, 0)    # Red for predictions

# Simple horizontal stack to avoid huge figures
from PIL import ImageDraw

def stack_horiz(imgs):
    if not imgs:
        return None
    widths, heights = zip(*(im.size for im in imgs))
    total_w = sum(widths)
    max_h = max(heights)
    canvas = Image.new('RGB', (total_w, max_h), (0, 0, 0))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.size[0]
    # draw a small legend
    draw = ImageDraw.Draw(canvas)
    legend_y = 10
    draw.rectangle([10, legend_y, 30, legend_y+10], fill=GT_COLOR)
    draw.text((35, legend_y-4), "GT", fill=(255,255,255))
    draw.rectangle([80, legend_y, 100, legend_y+10], fill=PRED_COLOR)
    draw.text((105, legend_y-4), "Pred", fill=(255,255,255))
    return canvas

# Visualization function
def visualize_predictions(image_path, model, class_names, int_colors, device, threshold=0.5, train_sz=512, display_max=1600):
    """
    Visualize predictions on an image (drawn on a downscaled display image to save memory)
    """
    # Load original image
    test_img = Image.open(image_path).convert('RGB')
    
    # Prepare model input (small)
    input_img = resize_img(test_img, target_sz=train_sz, divisor=1)
    input_tensor = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])(input_img)[None].to(device)
    
    # Prepare display image (moderate size)
    display_img = resize_img(test_img, target_sz=display_max, divisor=1)
    
    # Scale from model input space -> display space
    scale_inp_to_disp = min(display_img.size) / min(input_img.size)
    
    # Run model
    with torch.no_grad():
        model_output = model(input_tensor)
    
    # Move output to CPU
    model_output = [move_data_to_device(output, 'cpu') for output in model_output][0]
    
    # Filter by confidence threshold
    scores_mask = model_output['scores'] > threshold
    if scores_mask.sum() == 0:
        return display_img, None
    
    # Boxes scaled to display image
    pred_bboxes = BoundingBoxes(
        model_output['boxes'][scores_mask] * scale_inp_to_disp,
        format='xyxy',
        canvas_size=display_img.size[::-1]
    )
    pred_labels = [class_names[int(label)] for label in model_output['labels'][scores_mask]]
    pred_scores = model_output['scores'][scores_mask]
    
    # Masks scaled to display image
    pred_masks = F.interpolate(
        model_output['masks'][scores_mask],
        size=display_img.size[::-1],
        mode='bilinear',
        align_corners=False
    )
    pred_masks = torch.concat([
        Mask(torch.where(mask >= 0.5, 1, 0), dtype=torch.bool)
        for mask in pred_masks
    ])
    
    # Colors
    pred_colors = [PRED_COLOR for _ in pred_labels]
    
    # Draw
    img_tensor = transforms.PILToTensor()(display_img)
    annotated_tensor = draw_segmentation_masks(
        image=img_tensor,
        masks=pred_masks,
        alpha=0.3,
        colors=pred_colors
    )
    labels_with_scores = [f"{label}\n{prob*100:.1f}%" for label, prob in zip(pred_labels, pred_scores)]
    annotated_tensor = draw_bounding_boxes(
        image=annotated_tensor,
        boxes=pred_bboxes,
        labels=labels_with_scores,
        colors=pred_colors,
        fill=False,
        width=2
    )
    
    annotated_img = tensor_to_pil(annotated_tensor)
    return display_img, annotated_img

# Get ground truth annotations for comparison
def get_ground_truth(image_id, coco_data, image_dir):
    """Get ground truth annotations for an image"""
    img_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)
    if not img_info:
        return None, None
    
    # Get image path
    file_name = Path(img_info['file_name']).name
    image_path = image_dir / file_name
    if not image_path.exists():
        possible_files = list(image_dir.glob(f"{Path(file_name).stem}.*"))
        if possible_files:
            image_path = possible_files[0]
        else:
            return None, None
    
    # Get annotations
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
    return image_path, annotations

# Visualize a specific image
def visualize_with_ground_truth(image_id, coco_data, image_dir, model, class_names, int_colors, device, train_sz=512, display_max=1600):
    """Visualize both predictions and ground truth on a downscaled display image"""
    image_path, annotations = get_ground_truth(image_id, coco_data, image_dir)
    if image_path is None:
        return None
    
    test_img = Image.open(image_path).convert('RGB')
    display_img = resize_img(test_img, target_sz=display_max, divisor=1)
    
    # Build GT on display image
    if annotations:
        gt_mask_imgs = []
        gt_labels = []
        for ann in annotations:
            segmentation = ann.get('segmentation')
            if isinstance(segmentation, list) and len(segmentation) > 0:
                polygon_points = segmentation[0]
                xy_coords = [(polygon_points[i], polygon_points[i+1]) for i in range(0, len(polygon_points), 2)]
                # draw mask on original size then resize to display
                mask_img = create_polygon_mask(test_img.size, xy_coords)
                # resize mask to display size
                mask_img = mask_img.resize(display_img.size, resample=Image.NEAREST)
                gt_mask_imgs.append(mask_img)
                
                category_id = ann['category_id']
                category_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id), 'unknown')
                gt_labels.append(category_name)
        
        if gt_mask_imgs:
            gt_masks = Mask(torch.concat([Mask(transforms.PILToTensor()(m), dtype=torch.bool) for m in gt_mask_imgs]))
            gt_bboxes = BoundingBoxes(
                data=torchvision.ops.masks_to_boxes(gt_masks),
                format='xyxy',
                canvas_size=display_img.size[::-1]
            )
            gt_colors = [GT_COLOR for _ in gt_labels]
            img_tensor = transforms.PILToTensor()(display_img)
            gt_annotated = draw_segmentation_masks(image=img_tensor, masks=gt_masks, alpha=0.3, colors=gt_colors)
            gt_annotated = draw_bounding_boxes(image=gt_annotated, boxes=gt_bboxes, labels=gt_labels, colors=gt_colors, fill=False, width=2)
            gt_img = tensor_to_pil(gt_annotated)
        else:
            gt_img = display_img
    else:
        gt_img = display_img
    
    # Predictions on display image
    _, pred_img = visualize_predictions(image_path, model, class_names, int_colors, device, train_sz=train_sz, display_max=display_max)
    
    if pred_img:
        # Stack downscaled images side by side with legend
        return stack_horiz([gt_img, pred_img])
    else:
        return gt_img

# Main visualization
if __name__ == "__main__":
    all_image_ids = list(set([ann['image_id'] for ann in coco_data['annotations']]))
    print(f"\nFound {len(all_image_ids)} images with annotations")
    print("Visualizing predictions on all images...\n")
    
    output_dir = Path("./visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for image_id in all_image_ids:
        result_img = visualize_with_ground_truth(image_id, coco_data, image_dir, model, class_names, int_colors, device)
        if result_img:
            out_path = output_dir / f"image_{image_id}.png"
            result_img.save(out_path)
            print(f"Saved: {out_path}")

    # Optionally export loss curves if training history exists
    try:
        project_dir = Path('./pytorch-mask-r-cnn-bird-nest')
        run_dirs = sorted([p for p in project_dir.glob('*') if p.is_dir()], reverse=True)
        selected = None
        for rd in run_dirs:
            hp = rd / 'history.jsonl'
            if hp.exists() and hp.stat().st_size > 0:
                selected = rd
                break
        if selected is not None:
            history_path = selected / 'history.jsonl'
            with open(history_path, 'r', encoding='utf-8') as f:
                lines = [json.loads(line) for line in f if line.strip()]
            if lines:
                hist_df = pd.DataFrame(lines)
                plt.figure(figsize=(8,4))
                plt.plot(hist_df['epoch'], hist_df['train_loss'], label='Train Loss', marker='o')
                plt.plot(hist_df['epoch'], hist_df['valid_loss'], label='Valid Loss', marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Loss Curves - {selected.name}')
                plt.grid(True, alpha=0.3)
                plt.legend()
                out_path = output_dir / 'loss_curves.png'
                plt.tight_layout()
                plt.savefig(out_path, dpi=200)
                plt.close()
                print(f"Saved: {out_path}")
        else:
            print("Skip loss curves: no run with non-empty history.jsonl found. Re-run training after logging was added.")
    except Exception as e:
        print(f"Warning: could not export loss curves: {e}")

