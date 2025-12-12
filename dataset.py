"""
Dataset classes and utilities for COCO CLIP fine-tuning.
"""

import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLIP_MEAN, CLIP_STD, IMAGE_SIZE


def get_clip_transforms(image_size=IMAGE_SIZE):
    """
    Get standard CLIP image transforms.

    Args:
        image_size: Target image size (default: 224)

    Returns:
        transform: torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])


def denormalize_image(tensor):
    """
    Denormalize image tensor for visualization.

    Args:
        tensor: [3, H, W] normalized image tensor

    Returns:
        denormalized: [3, H, W] denormalized tensor (range [0, 1])
    """
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)


class COCOClipDataset(Dataset):
    """
    COCO dataset for CLIP fine-tuning.

    Loads images and pre-computed text embeddings from cache.
    """

    def __init__(self, split='train', dataset_dir=None, transform=None, return_all_captions=False, return_raw_image=False):
        """
        Args:
            split: 'train' or 'val'
            dataset_dir: Path to COCO dataset directory (default: /content/coco2014)
            transform: Image transforms (default: CLIP transforms)
            return_all_captions: If True, return all captions; if False, return random caption
            return_raw_image: If True, return raw PIL image for visualization (default: False)
        """
        self.split = split
        self.return_all_captions = return_all_captions
        self.return_raw_image = return_raw_image

        # Set paths
        if dataset_dir is None:
            from config import DEFAULT_DATASET_DIR
            dataset_dir = DEFAULT_DATASET_DIR
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / f'{split}2014'
        self.cache_file = self.dataset_dir / f'{split}_text_embeddings.pt'

        # Load cached text embeddings
        if not self.cache_file.exists():
            raise FileNotFoundError(
                f"Text embedding cache not found at {self.cache_file}. "
                f"Please run coco_dataset_prep.ipynb first."
            )
        cache = torch.load(self.cache_file)
        self.cache_data = cache['data']

        # Set transform
        self.transform = transform if transform is not None else get_clip_transforms()

        print(f"Loaded {split} dataset: {len(self.cache_data):,} images")

    def __len__(self):
        return len(self.cache_data)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            dict with keys:
                - image: [3, H, W] transformed image tensor
                - text_embedding(s): [512] or [num_captions, 512] text embeddings
                - caption(s): str or list[str]
                - image_id: int
                - image_path: str (optional)
                - image_raw: PIL.Image (optional, for visualization)
        """
        item = self.cache_data[idx]
        image_id = item['image_id']
        embeddings = item['embeddings']
        captions = item['captions']

        # Load image
        image_filename = f'COCO_{self.split}2014_{image_id:012d}.jpg'
        image_path = self.image_dir / image_filename

        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            # If image fails to load, return zero tensor
            print(f"Warning: Failed to load {image_path}: {e}")
            image = None
            image_tensor = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

        # Prepare return dict
        result = {
            'image': image_tensor,
            'image_id': image_id,
            'image_path': str(image_path)
        }

        # Add raw image for visualization (only if requested)
        if self.return_raw_image and image is not None:
            result['image_raw'] = image

        # Return all captions or random caption
        if self.return_all_captions:
            result['text_embeddings'] = embeddings
            result['captions'] = captions
        else:
            caption_idx = random.randint(0, len(captions) - 1)
            result['text_embedding'] = embeddings[caption_idx]
            result['caption'] = captions[caption_idx]

        return result


class COCOClipDatasetWithCaptions(Dataset):
    """
    COCO dataset that returns raw captions for on-the-fly encoding.

    Used in dataset preparation notebook for encoding captions.
    """

    def __init__(self, split='train', dataset_dir=None, transform=None):
        """
        Args:
            split: 'train' or 'val'
            dataset_dir: Path to COCO dataset directory
            transform: Image transforms
        """
        import json

        self.split = split

        # Set paths
        if dataset_dir is None:
            from config import DEFAULT_DATASET_DIR
            dataset_dir = DEFAULT_DATASET_DIR
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / f'{split}2014'
        self.annotation_file = self.dataset_dir / 'annotations' / f'captions_{split}2014.json'

        # Load annotations
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)

        # Build image_id -> captions mapping
        self.image_to_captions = {}
        for ann in annotations['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            if image_id not in self.image_to_captions:
                self.image_to_captions[image_id] = []
            self.image_to_captions[image_id].append(caption)

        self.image_ids = sorted(list(self.image_to_captions.keys()))

        # Set transform
        self.transform = transform if transform is not None else get_clip_transforms()

        print(f"Loaded {split} dataset: {len(self.image_ids):,} images")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            dict with keys:
                - image_id: int
                - captions: list[str]
                - image: [3, H, W] transformed image tensor (optional)
        """
        image_id = self.image_ids[idx]
        captions = self.image_to_captions[image_id]

        result = {
            'image_id': image_id,
            'captions': captions
        }

        # Optionally load image
        if self.transform is not None:
            image_filename = f'COCO_{self.split}2014_{image_id:012d}.jpg'
            image_path = self.image_dir / image_filename

            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.transform(image)
                result['image'] = image_tensor
                result['image_raw'] = image
            except Exception as e:
                print(f"Warning: Failed to load {image_path}: {e}")

        return result
