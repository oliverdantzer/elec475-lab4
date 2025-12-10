"""
Example Usage Script for COCO CLIP Dataset

This script demonstrates how to use the prepared COCO dataset
for CLIP fine-tuning after running coco_dataset_prep.ipynb.

Author: ELEC 475 Lab 4
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from collections import defaultdict
import random


# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_SIZE = 224

# Define image transforms
image_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
])


class COCOClipDataset(Dataset):
    """
    COCO Dataset for CLIP fine-tuning.

    Returns:
        image: Preprocessed image tensor [3, 224, 224]
        text_embedding: Pre-computed CLIP text embedding [512]
        caption: Original caption text (for reference)
        image_id: COCO image ID
    """

    def __init__(self, split='train', dataset_dir='/content/coco2014',
                 transform=None, return_all_captions=False):
        """
        Args:
            split: 'train' or 'val'
            dataset_dir: Path to COCO dataset directory
            transform: Image transforms to apply
            return_all_captions: If True, return all captions. If False, randomly select one.
        """
        self.split = split
        self.transform = transform or image_transforms
        self.return_all_captions = return_all_captions

        # Set paths
        dataset_dir = Path(dataset_dir)
        self.image_dir = dataset_dir / f'{split}2014'
        self.cache_file = dataset_dir / f'{split}_text_embeddings.pt'

        # Load cached embeddings
        print(f"Loading cached embeddings from {self.cache_file.name}...")
        cache = torch.load(self.cache_file)
        self.cache_data = cache['data']
        self.embedding_dim = cache['embedding_dim']

        # Build index: image_id -> cache index
        self.image_id_to_idx = {
            item['image_id']: idx
            for idx, item in enumerate(self.cache_data)
        }

        print(f"  ✓ Loaded {len(self.cache_data)} images")
        print(f"  ✓ Embedding dimension: {self.embedding_dim}")

    def __len__(self):
        return len(self.cache_data)

    def __getitem__(self, idx):
        # Get cached data
        item = self.cache_data[idx]
        image_id = item['image_id']
        embeddings = item['embeddings']  # [num_captions, 512]
        captions = item['captions']

        # Load image
        image_filename = f'COCO_{self.split}2014_{image_id:012d}.jpg'
        image_path = self.image_dir / image_filename

        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, 224, 224)

        # Select caption(s)
        if self.return_all_captions:
            # Return all captions and embeddings
            return {
                'image': image,
                'text_embeddings': embeddings,
                'captions': captions,
                'image_id': image_id
            }
        else:
            # Randomly select one caption
            caption_idx = random.randint(0, len(captions) - 1)
            return {
                'image': image,
                'text_embedding': embeddings[caption_idx],
                'caption': captions[caption_idx],
                'image_id': image_id
            }


def main():
    """
    Example usage of the COCO CLIP dataset.
    """
    print("="*60)
    print("COCO CLIP Dataset Example Usage")
    print("="*60)

    # Create datasets
    print("\n1. Creating dataset instances...")
    train_dataset = COCOClipDataset(split='train')
    val_dataset = COCOClipDataset(split='val')

    print(f"\nTrain set: {len(train_dataset):,} images")
    print(f"Val set: {len(val_dataset):,} images")

    # Create data loaders
    print("\n2. Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")

    # Example: Iterate through one batch
    print("\n3. Loading one batch...")
    batch = next(iter(train_loader))

    print(f"\nBatch contents:")
    print(f"  Images shape: {batch['image'].shape}")
    print(f"  Text embeddings shape: {batch['text_embedding'].shape}")
    print(f"  Number of captions: {len(batch['caption'])}")
    print(f"  Number of image IDs: {len(batch['image_id'])}")

    print(f"\nExample caption: \"{batch['caption'][0]}\"")
    print(f"Example image ID: {batch['image_id'][0]}")

    # Example: Check data types and ranges
    print("\n4. Data validation...")
    print(f"  Image dtype: {batch['image'].dtype}")
    print(f"  Image value range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
    print(f"  Text embedding dtype: {batch['text_embedding'].dtype}")
    print(f"  Text embedding norm: {torch.norm(batch['text_embedding'][0]):.3f}")

    # Example: Single vs All captions
    print("\n5. Testing return_all_captions mode...")
    dataset_all_captions = COCOClipDataset(split='train', return_all_captions=True)
    sample = dataset_all_captions[0]

    print(f"  Image shape: {sample['image'].shape}")
    print(f"  All text embeddings shape: {sample['text_embeddings'].shape}")
    print(f"  Number of captions: {len(sample['captions'])}")
    print(f"  Captions:")
    for i, cap in enumerate(sample['captions']):
        print(f"    {i+1}. {cap}")

    print("\n" + "="*60)
    print("Example usage complete!")
    print("="*60)

    return train_loader, val_loader


if __name__ == '__main__':
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False

    if not IN_COLAB:
        print("WARNING: This script is designed for Google Colab.")
        print("If running locally, ensure the dataset is at /content/coco2014/")
        print("or modify DATASET_DIR in COCOClipDataset initialization.\n")

    train_loader, val_loader = main()

    print("\nData loaders are ready for training!")
    print("Access them as: train_loader, val_loader")
