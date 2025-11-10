from torch.utils.data import Dataset
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
import torchvision.transforms.v2  as transforms

from PIL import Image, ImageDraw
import json
from pathlib import Path

class StudentIDDataset(Dataset):
    """
    This class represents a PyTorch Dataset for a collection of images and their annotations.
    The class is designed to load images along with their corresponding segmentation masks, bounding box annotations, and labels.
    """
    def __init__(self, img_keys, annotation_df, img_dict, class_to_idx, transforms=None):
        """
        Constructor for the HagridDataset class.

        Parameters:
        img_keys (list): List of unique identifiers for images.
        annotation_df (DataFrame): DataFrame containing the image annotations.
        img_dict (dict): Dictionary mapping image identifiers to image file paths.
        class_to_idx (dict): Dictionary mapping class labels to indices.
        transforms (callable, optional): Optional transform to be applied on a sample.
        """
        super(Dataset, self).__init__()
        
        self._img_keys = img_keys  # List of image keys
        self._annotation_df = annotation_df  # DataFrame containing annotations
        self._img_dict = img_dict  # Dictionary mapping image keys to image paths
        self._class_to_idx = class_to_idx  # Dictionary mapping class names to class indices
        self._transforms = transforms  # Image transforms to be applied
        
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        int: The number of items in the dataset.
        """
        return len(self._img_keys)
        
    def __getitem__(self, index):
        """
        Fetch an item from the dataset at the specified index.

        Parameters:
        index (int): Index of the item to fetch from the dataset.

        Returns:
        tuple: A tuple containing the image and its associated target (annotations).
        """
        # Retrieve the key for the image at the specified index
        img_key = self._img_keys[index]
        # Get the annotations for this image
        annotation = self._annotation_df.loc[img_key]
        # Load the image and its target (segmentation masks, bounding boxes and labels)
        image, target = self._load_image_and_target(annotation)
        
        # Apply the transformations, if any
        if self._transforms:
            image, target = self._transforms(image, target)
        
        return image, target

    def _load_image_and_target(self, annotation):
        """
        Load an image and its target (bounding boxes and labels).

        Parameters:
        annotation (pandas.Series): The annotations for an image.

        Returns:
        tuple: A tuple containing the image and a dictionary with 'boxes' and 'labels' keys.
        """
        # Retrieve the file path of the image
        filepath = self._img_dict[annotation.name]
        # Open the image file and convert it to RGB
        image = Image.open(filepath).convert('RGB')
        
        # Convert the class labels to indices
        labels = [shape['label'] for shape in annotation['shapes']]
        labels = torch.Tensor([self._class_to_idx[label] for label in labels])
        labels = labels.to(dtype=torch.int64)

        # Convert polygons to mask images
        shape_points = [shape['points'] for shape in annotation['shapes']]
        xy_coords = [[tuple(p) for p in points] for points in shape_points]
        mask_imgs = [create_polygon_mask(image.size, xy) for xy in xy_coords]
        masks = Mask(torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs]))

        # Generate bounding box annotations from segmentation masks
        bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=image.size[::-1])
                
        return image, {'masks': masks,'boxes': bboxes, 'labels': labels}
    


def tuple_batch(batch):
    return tuple(zip(*batch))


def create_polygon_mask(image_size, vertices):
    """
    Create a grayscale image with a white polygonal area on a black background.

    Parameters:
    - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
    - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                        of the polygon. Vertices should be in clockwise or counter-clockwise order.

    Returns:
    - PIL.Image.Image: A PIL Image object containing the polygonal mask.
    """

    # Create a new black image with the given dimensions
    mask_img = Image.new('L', image_size, 0)
    
    # Draw the polygon on the image. The area inside the polygon will be white (255).
    ImageDraw.Draw(mask_img, 'L').polygon(vertices, fill=(255))

    # Return the image with the drawn polygon
    return mask_img


class COCODataset(Dataset):
    """
    PyTorch Dataset for COCO format annotations.
    Designed for bird nest detection on power grid images.
    """
    def __init__(self, coco_json_path, image_dir, class_to_idx, transforms=None, img_ids=None):
        """
        Constructor for the COCODataset class.

        Parameters:
        coco_json_path (str or Path): Path to the COCO format JSON annotation file.
        image_dir (str or Path): Directory containing the images.
        class_to_idx (dict): Dictionary mapping class names to indices.
        transforms (callable, optional): Optional transform to be applied on a sample.
        img_ids (list, optional): List of image IDs to include. If None, includes all images with annotations.
        """
        super(Dataset, self).__init__()
        
        self._image_dir = Path(image_dir)
        self._class_to_idx = class_to_idx
        self._transforms = transforms
        
        # Load COCO JSON file
        with open(coco_json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # Build image_id to image info mapping
        self._images = {img['id']: img for img in coco_data['images']}
        
        # Build image_id to annotations mapping
        self._annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self._annotations:
                self._annotations[img_id] = []
            self._annotations[img_id].append(ann)
        
        # Build category_id to category name mapping
        self._categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Get list of image IDs
        all_img_ids = list(self._images.keys())

        # Filter by img_ids if provided (keeps negatives too)
        if img_ids is not None:
            self._img_ids = [img_id for img_id in all_img_ids if img_id in img_ids]
        else:
            self._img_ids = all_img_ids
        
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        int: The number of items in the dataset.
        """
        return len(self._img_ids)
        
    def __getitem__(self, index):
        """
        Fetch an item from the dataset at the specified index.

        Parameters:
        index (int): Index of the item to fetch from the dataset.

        Returns:
        tuple: A tuple containing the image and its associated target (annotations).
        """
        # Retrieve the image ID at the specified index
        img_id = self._img_ids[index]
        # Get the image info
        img_info = self._images[img_id]
        # Get the annotations for this image (may be empty for negatives)
        annotations = self._annotations.get(img_id, [])
        
        # Load the image and its target (segmentation masks, bounding boxes and labels)
        image, target = self._load_image_and_target(img_info, annotations)
        
        # Apply the transformations, if any
        if self._transforms:
            image, target = self._transforms(image, target)
        
        return image, target

    def _load_image_and_target(self, img_info, annotations):
        """
        Load an image and its target (bounding boxes, masks and labels).

        Parameters:
        img_info (dict): The image information from COCO JSON.
        annotations (list): List of annotation dictionaries for this image.

        Returns:
        tuple: A tuple containing the image and a dictionary with 'boxes', 'masks' and 'labels' keys.
        """
        # Get image file path - try to find the image in the image_dir
        # First try the filename from JSON, then try just the filename
        file_name = Path(img_info['file_name']).name
        filepath = self._image_dir / file_name
        
        # If file doesn't exist, try to find it with different extensions
        if not filepath.exists():
            # Try to find any file with the same base name
            possible_files = list(self._image_dir.glob(f"{filepath.stem}.*"))
            if possible_files:
                filepath = possible_files[0]
            else:
                raise FileNotFoundError(f"Image file not found: {filepath}")
        
        # Open the image file and convert it to RGB
        image = Image.open(filepath).convert('RGB')
        
        # Extract labels, masks, and bounding boxes from annotations
        labels = []
        mask_imgs = []
        bboxes_list = []
        
        for ann in annotations:
            # Get category name and convert to label index
            category_id = ann['category_id']
            category_name = self._categories[category_id]
            label_idx = self._class_to_idx.get(category_name, 0)  # Default to 0 (background) if not found
            labels.append(label_idx)
            
            # Get segmentation polygon
            # COCO segmentation can be a list of polygons (for complex shapes)
            # We'll use the first polygon if multiple exist
            segmentation = ann['segmentation']
            if isinstance(segmentation, list) and len(segmentation) > 0:
                # Flatten the polygon points and convert to (x, y) tuples
                polygon_points = segmentation[0]  # Use first polygon
                xy_coords = [(polygon_points[i], polygon_points[i+1]) 
                            for i in range(0, len(polygon_points), 2)]
                
                # Create mask from polygon
                mask_img = create_polygon_mask(image.size, xy_coords)
                mask_imgs.append(mask_img)
            
            # Get bounding box (COCO format: [x, y, width, height])
            # Convert to [x1, y1, x2, y2] format
            bbox = ann['bbox']
            x, y, w, h = bbox
            bboxes_list.append([x, y, x + w, y + h])
        
        # Convert labels to tensor
        labels = torch.Tensor(labels).to(dtype=torch.int64)
        
        # Convert mask images to tensors and combine
        if mask_imgs:
            masks = Mask(torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) 
                                      for mask_img in mask_imgs]))
        else:
            # If no masks, create empty mask
            masks = Mask(torch.zeros((0, image.size[1], image.size[0]), dtype=torch.bool))
        
        # Convert bounding boxes to BoundingBoxes tensor
        if bboxes_list:
            bboxes = BoundingBoxes(data=torch.Tensor(bboxes_list), format='xyxy', 
                                  canvas_size=image.size[::-1])
        else:
            # If no boxes, create empty boxes
            bboxes = BoundingBoxes(data=torch.zeros((0, 4)), format='xyxy', 
                                  canvas_size=image.size[::-1])
                
        return image, {'masks': masks, 'boxes': bboxes, 'labels': labels}