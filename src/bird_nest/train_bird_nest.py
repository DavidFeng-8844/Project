"""
Training script for bird nest detection using Mask R-CNN
Dataset: COCO format bird nest dataset
"""

import datetime
import os
from pathlib import Path
import json
import random
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
import torchvision.transforms.v2 as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from windows_utils import COCODataset, tuple_batch, create_polygon_mask
from cjm_pytorch_utils.core import get_torch_device, set_seed, move_data_to_device
from cjm_pil_utils.core import resize_img
from cjm_torchvision_tfms.core import ResizeMax, PadSquare, CustomRandomIoUCrop

# Wrap IoU crop to safely handle images without boxes (negative samples)
class SafeIoUCrop(torch.nn.Module):
    def __init__(self, inner_crop):
        super().__init__()
        self.inner_crop = inner_crop
    def forward(self, image, target):
        boxes = target.get('boxes', None)
        if boxes is None or (hasattr(boxes, 'shape') and boxes.shape[0] == 0):
            return image, target
        return self.inner_crop(image, target)

# Training loop
def run_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler, epoch_id, is_training):
    model.train() if is_training else model.eval()
    
    epoch_loss = 0
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")
    
    for batch_id, (inputs, targets) in enumerate(dataloader):
        inputs = torch.stack(inputs).to(device, non_blocking=True)
        targets = move_data_to_device(targets, device)
        
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        
        with autocast(device_type=device_type):
            if is_training:
                losses = model(inputs, targets)
            else:
                previous_training_state = model.training
                model.train()
                with torch.no_grad():
                    losses = model(inputs, targets)
                if not previous_training_state:
                    model.eval()
        
        loss = sum([loss for loss in losses.values()])
        
        if is_training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    lr_scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            optimizer.zero_grad()
        
        loss_item = loss.item()
        epoch_loss += loss_item
        
        progress_bar_dict = dict(loss=loss_item, avg_loss=epoch_loss/(batch_id+1))
        if is_training:
            progress_bar_dict.update(lr=lr_scheduler.get_last_lr()[0])
        progress_bar.set_postfix(progress_bar_dict)
        progress_bar.update()
    
    progress_bar.close()
    return epoch_loss / (batch_id + 1)

def train_loop(model, train_dataloader, valid_dataloader, optimizer, lr_scheduler, 
               device, epochs, checkpoint_path, history_path, use_scaler=False):
    scaler = GradScaler('cuda') if device.type == 'cuda' and use_scaler else None
    best_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss = run_epoch(model, train_dataloader, optimizer, lr_scheduler, 
                             device, scaler, epoch, is_training=True)
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_dataloader, None, None, 
                                 device, scaler, epoch, is_training=False)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

            # Save metadata about the training process
            training_metadata = {
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss, 
                'learning_rate': lr_scheduler.get_last_lr()[0],
                'model_architecture': model.name
            }
            with open(Path(checkpoint_path.parent/'training_metadata.json'), 'w') as f:
                json.dump(training_metadata, f)

        # Append epoch metrics to history file for later visualization
        try:
            epoch_record = {
                'epoch': int(epoch),
                'train_loss': float(train_loss),
                'valid_loss': float(valid_loss),
                'learning_rate': float(lr_scheduler.get_last_lr()[0])
            }
            with open(history_path, 'a', encoding='utf-8') as hf:
                hf.write(json.dumps(epoch_record) + "\n")
        except Exception as e:
            print(f"Warning: failed to write history record: {e}")
    
    if device.type != 'cpu':
        torch.cuda.empty_cache()

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    # Set random seed
    seed = 1234
    set_seed(seed)

    # Set device
    raw_device = get_torch_device()
    device = torch.device(raw_device) if not isinstance(raw_device, torch.device) else raw_device
    dtype = torch.float32
    print(f"Using device: {device}")

    # Enable cuDNN autotuner for fixed-size inputs
    try:
        import torch.backends.cudnn as cudnn
        if device.type == 'cuda':
            cudnn.benchmark = True
    except ImportError:
        pass

    # Dataset paths
    dataset_dir = repo_root / "data" / "active" / "bird_nest" / "coco"
    coco_json_path = dataset_dir / "dataset.json"
    image_dir = dataset_dir

    if not coco_json_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {coco_json_path}")

    # Project directory
    project_name = "pytorch-mask-r-cnn-bird-nest"
    project_dir = repo_root / "experiments" / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO JSON to get class names
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # Get class names from categories
    class_names = ['background'] + [cat['name'] for cat in coco_data['categories']]
    print(f"Classes: {class_names}")

    # Create class to index mapping
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # Training parameters
    train_sz = 512
    bs = 4
    epochs = 40
    lr = 5e-4

    # Data augmentation
    iou_crop = CustomRandomIoUCrop(min_scale=0.3, max_scale=1.0,
                                   min_aspect_ratio=0.5, max_aspect_ratio=2.0,
                                   sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                                   trials=50, jitter_factor=0.25)

    resize_max = ResizeMax(max_sz=train_sz)
    pad_square = PadSquare(shift=True, fill=0)

    # Compose transforms
    data_aug_tfms = transforms.Compose([
        SafeIoUCrop(iou_crop),
        transforms.ColorJitter(brightness=(0.875, 1.125), contrast=(0.5, 1.5),
                               saturation=(0.5, 1.5), hue=(-0.05, 0.05)),
        transforms.RandomGrayscale(),
        transforms.RandomEqualize(),
        transforms.RandomPosterize(bits=3, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    resize_pad_tfm = transforms.Compose([
        resize_max,
        pad_square,
        transforms.Resize([train_sz] * 2, antialias=True)
    ])

    final_tfms = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.SanitizeBoundingBoxes(),
    ])

    train_tfms = transforms.Compose([data_aug_tfms, resize_pad_tfm, final_tfms])
    valid_tfms = transforms.Compose([resize_pad_tfm, final_tfms])

    # Split dataset
    all_img_ids = [img['id'] for img in coco_data['images']]  # include negatives
    random.shuffle(all_img_ids)

    train_split = int(len(all_img_ids) * 0.8)
    train_img_ids = all_img_ids[:train_split]
    val_img_ids = all_img_ids[train_split:]

    print(f"Training samples: {len(train_img_ids)}, Validation samples: {len(val_img_ids)}")

    # Create datasets
    train_dataset = COCODataset(coco_json_path, image_dir, class_to_idx, train_tfms, img_ids=train_img_ids)
    val_dataset = COCODataset(coco_json_path, image_dir, class_to_idx, valid_tfms, img_ids=val_img_ids)

    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    # Create data loaders
    import platform

    if platform.system() == 'Windows':
        default_workers = 2
    else:
        import multiprocessing
        default_workers = min(4, max(1, multiprocessing.cpu_count() // 2))

    num_workers = int(os.getenv("NUM_WORKERS", default_workers))
    if platform.system() == 'Windows' and num_workers > 0 and __name__ != "__main__":
        print("Warning: Multiprocessing DataLoader on Windows requires running as a script.")
        print("Set NUM_WORKERS=0 (or rerun from command line if you want >0).")
        num_workers = 0
    num_workers = max(0, num_workers)

    data_loader_params = {
        'batch_size': bs,
        'num_workers': num_workers,
        'persistent_workers': num_workers > 0,
        'pin_memory': device.type == 'cuda',
        'collate_fn': tuple_batch,
        'prefetch_factor': 2 if num_workers > 0 else None,
    }

    train_dataloader = DataLoader(train_dataset, **data_loader_params, shuffle=True)
    val_dataloader = DataLoader(val_dataset, **data_loader_params)

    # Initialize model
    model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes=len(class_names))
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced=dim_reduced,
                                                       num_classes=len(class_names))

    model.to(device=device, dtype=dtype)
    model.device = device
    model.name = 'maskrcnn_resnet50_fpn_v2'

    # Setup training artifacts
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = Path(project_dir / timestamp)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model.name}.pth"
    history_path = checkpoint_dir / "history.jsonl"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                      total_steps=epochs * len(train_dataloader))

    # Save color map
    from distinctipy import distinctipy
    colors = distinctipy.get_colors(len(class_names))
    color_map = {'items': [{'label': label, 'color': list(color)}
                           for label, color in zip(class_names, colors)]}
    with open(checkpoint_dir / "bird_nest-colormap.json", "w") as f:
        json.dump(color_map, f)

    print(f"Starting training...")
    print(f"Checkpoint will be saved to: {checkpoint_path}")

    # Train
    train_loop(model=model,
               train_dataloader=train_dataloader,
               valid_dataloader=val_dataloader,
               optimizer=optimizer,
               lr_scheduler=lr_scheduler,
               device=device,
               epochs=epochs,
               checkpoint_path=checkpoint_path,
               history_path=history_path,
               use_scaler=True)

    print(f"Training completed! Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    import platform
    if platform.system() == "Windows":
        import multiprocessing
        multiprocessing.freeze_support()
    main()