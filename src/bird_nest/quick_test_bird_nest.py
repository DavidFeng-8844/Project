"""
Quick test script for bird nest detection
This script tests if the dataset can be loaded correctly
"""

from pathlib import Path
import json
from windows_utils import COCODataset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
from torchvision.tv_tensors import BoundingBoxes, Mask
import torchvision

# Dataset paths
repo_root = Path(__file__).resolve().parent.parent.parent
dataset_dir = repo_root / "data" / "active" / "bird_nest" / "coco"
coco_json_path = dataset_dir / "dataset.json"
image_dir = dataset_dir

# Load COCO JSON to get class names
with open(coco_json_path, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

# Get class names
class_names = ['background'] + [cat['name'] for cat in coco_data['categories']]
print(f"Classes: {class_names}")

# Create class to index mapping
class_to_idx = {c: i for i, c in enumerate(class_names)}

# Create dataset (without transforms for testing)
dataset = COCODataset(coco_json_path, image_dir, class_to_idx, transforms=None)

print(f"\nDataset size: {len(dataset)}")
print(f"Number of images with annotations: {len(dataset._img_ids)}")

# Test loading a sample
if len(dataset) > 0:
    print("\nTesting data loading...")
    sample_image, sample_target = dataset[0]
    
    print(f"Sample image size: {sample_image.size}")
    print(f"Number of objects: {len(sample_target['labels'])}")
    print(f"Labels: {[class_names[int(l)] for l in sample_target['labels']]}")
    print(f"Bounding boxes shape: {sample_target['boxes'].shape}")
    print(f"Masks shape: {sample_target['masks'].shape}")
    
    # Visualize the sample
    from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
    from distinctipy import distinctipy
    
    # Generate colors
    colors = distinctipy.get_colors(len(class_names))
    int_colors = [tuple(int(c*255) for c in color) for color in colors]
    
    # Get colors for this sample
    sample_colors = [int_colors[int(l.item())] for l in sample_target['labels']]
    
    # Convert image to tensor
    img_tensor = transforms.PILToTensor()(sample_image)
    
    # Draw masks
    annotated = draw_segmentation_masks(
        image=img_tensor,
        masks=sample_target['masks'],
        alpha=0.3,
        colors=sample_colors
    )
    
    # Draw bounding boxes
    labels = [class_names[int(l.item())] for l in sample_target['labels']]
    annotated = draw_bounding_boxes(
        image=annotated,
        boxes=sample_target['boxes'],
        labels=labels,
        colors=sample_colors,
        fill=False,
        width=2
    )
    
    # Convert back to PIL and display
    from cjm_pytorch_utils.core import tensor_to_pil
    result_img = tensor_to_pil(annotated)
    
    # Save visualization
    output_path = repo_root / "experiments" / "test_bird_nest_sample.png"
    result_img.save(output_path)
    print(f"\nSample visualization saved to: {output_path}")
    
    # Display
    plt.figure(figsize=(15, 10))
    plt.imshow(result_img)
    plt.axis('off')
    plt.title("Sample Bird Nest Detection (Ground Truth)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("Test completed successfully!")
else:
    print("Error: Dataset is empty. Please check your dataset files.")

