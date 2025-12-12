"""
Utility functions for CLIP fine-tuning.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


@torch.no_grad()
def compute_embeddings(model, dataset, batch_size=128, device='cuda'):
    """
    Compute image and text embeddings for entire dataset.

    Args:
        model: CLIPModel instance
        dataset: Dataset instance
        batch_size: Batch size for processing
        device: Device to use

    Returns:
        image_embeddings: [N, 512]
        text_embeddings: [N, 512]
        captions: List[str] of length N
        image_ids: List[int] of length N
    """
    from torch.utils.data import DataLoader

    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_image_embeddings = []
    all_text_embeddings = []
    all_captions = []
    all_image_ids = []

    print("Computing embeddings...")
    for batch in tqdm(loader):
        images = batch['image'].to(device)
        text_embeddings = batch['text_embedding'].to(device)

        # Encode images
        image_embeddings = model.encode_image(images)

        # Store
        all_image_embeddings.append(image_embeddings.cpu())
        all_text_embeddings.append(text_embeddings.cpu())
        all_captions.extend(batch['caption'])
        all_image_ids.extend(batch['image_id'].tolist())

    # Concatenate
    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    text_embeddings = torch.cat(all_text_embeddings, dim=0)

    print(f"✓ Computed embeddings:")
    print(f"  Images: {image_embeddings.shape}")
    print(f"  Texts: {text_embeddings.shape}")

    return image_embeddings, text_embeddings, all_captions, all_image_ids


def compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10]):
    """
    Compute Recall@K for retrieval evaluation.

    Args:
        similarity_matrix: [N, M] similarity scores
        k_values: List of K values to compute

    Returns:
        recall_dict: Dictionary of {K: recall_value}
    """
    N = similarity_matrix.shape[0]

    # Get top-K indices for each query
    top_k_indices = torch.topk(similarity_matrix, k=max(k_values), dim=1, largest=True).indices

    recalls = {}
    for k in k_values:
        # Check if correct index is in top-K
        correct_in_top_k = torch.any(
            top_k_indices[:, :k] == torch.arange(N).unsqueeze(1),
            dim=1
        )
        recall = correct_in_top_k.float().mean().item()
        recalls[k] = recall

    return recalls


def visualize_samples(dataset, num_samples=6, figsize=(15, 10)):
    """
    Visualize random samples from the dataset.

    Args:
        dataset: Dataset instance (must have return_raw_image=True)
        num_samples: Number of samples to display
        figsize: Figure size
    """
    import random
    from dataset import denormalize_image

    # Temporarily enable raw image return if not already enabled
    original_return_raw_image = getattr(dataset, 'return_raw_image', False)
    dataset.return_raw_image = True

    # Select random indices
    indices = random.sample(range(len(dataset)), num_samples)

    # Create subplot grid
    rows = (num_samples + 2) // 3
    cols = min(3, num_samples)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, ax in zip(indices, axes):
        # Get sample
        sample = dataset[idx]
        image = sample['image']
        caption = sample['caption']
        image_id = sample['image_id']

        # Denormalize image for display
        image_display = denormalize_image(image)
        image_display = torch.clamp(image_display, 0, 1)
        image_display = image_display.permute(1, 2, 0).numpy()

        # Display image
        ax.imshow(image_display)
        ax.axis('off')

        # Add caption as title
        wrapped_caption = '\n'.join(
            [caption[i:i+40] for i in range(0, len(caption), 40)]
        )
        ax.set_title(f"ID: {image_id}\n{wrapped_caption}",
                     fontsize=9, pad=10)

    # Hide extra subplots
    for ax in axes[num_samples:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"\nDisplayed {num_samples} random samples")

    # Restore original setting
    dataset.return_raw_image = original_return_raw_image


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_acc, config, filepath):
    """
    Save model checkpoint.

    Args:
        model: CLIPModel instance
        optimizer: Optimizer instance
        scheduler: Scheduler instance
        epoch: Current epoch
        val_loss: Validation loss
        val_acc: Validation accuracy
        config: Training configuration dict
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'config': config,
        'flags': {
            'use_batch_norm': config.get('use_batch_norm', False),
            'use_attention_pooling': config.get('use_attention_pooling', False)
        }
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint
        model: CLIPModel instance
        optimizer: Optimizer instance (optional)
        scheduler: Scheduler instance (optional)
        device: Device to load to

    Returns:
        checkpoint: Checkpoint dict
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


# ========== Dataset Preparation Functions ==========

def encode_and_cache_captions(split, dataset_dir, tokenizer, text_encoder, device='cuda', batch_size=64):
    """
    Encode all captions using CLIP text encoder and cache to disk.

    Args:
        split: 'train' or 'val'
        dataset_dir: Path to COCO dataset directory
        tokenizer: CLIP tokenizer
        text_encoder: CLIP text encoder model
        device: Device to use
        batch_size: Batch size for encoding

    Returns:
        cache_file: Path to saved cache file
    """
    dataset_dir = Path(dataset_dir)
    caption_file = dataset_dir / 'annotations' / f'captions_{split}2014.json'

    print(f"Loading captions from {caption_file.name}...")
    with open(caption_file, 'r') as f:
        coco_data = json.load(f)

    # Organize captions by image_id
    print("Organizing captions by image_id...")
    image_to_captions = defaultdict(list)
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        image_to_captions[image_id].append(caption)

    print(f"Found {len(image_to_captions)} unique images with captions")

    # Flatten all captions for batch processing with index tracking
    print(f"\nPreparing captions for batch encoding...")
    image_ids = list(image_to_captions.keys())
    all_captions = []
    image_id_to_indices = {}  # Maps image_id to list of caption indices
    current_idx = 0

    for img_id in image_ids:
        captions = image_to_captions[img_id]
        num_captions = len(captions)

        # Track indices for this image
        image_id_to_indices[img_id] = list(range(current_idx, current_idx + num_captions))
        current_idx += num_captions

        all_captions.extend(captions)

    print(f"Total captions to encode: {len(all_captions):,}")

    # Encode all captions in batches
    print(f"\nEncoding captions in batches of {batch_size}...")
    text_encoder.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(all_captions), batch_size), desc=f"Encoding {split} captions"):
            batch_captions = all_captions[i:i+batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_captions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors='pt'
            )

            # Encode
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = text_encoder(**inputs)
            embeddings = outputs.pooler_output.cpu()

            all_embeddings.append(embeddings)

            if i % (batch_size * 100) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Organize embeddings by image_id (fast O(n) lookup)
    print("\nOrganizing embeddings by image_id...")
    cache_data = []
    for img_id in image_ids:
        # Get caption indices for this image (O(1) lookup)
        caption_indices = image_id_to_indices[img_id]

        # Get embeddings and captions for this image
        embeddings = all_embeddings[caption_indices]
        captions = image_to_captions[img_id]

        # Store
        cache_data.append({
            'image_id': img_id,
            'embeddings': embeddings,
            'captions': captions
        })

    # Save cache
    cache_file = dataset_dir / f'{split}_text_embeddings.pt'
    print(f"\nSaving cache to {cache_file}...")

    torch.save({
        'data': cache_data,
        'model_name': 'openai/clip-vit-base-patch32',
        'embedding_dim': text_encoder.config.hidden_size
    }, cache_file)

    cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
    total_captions = sum(len(item['captions']) for item in cache_data)

    print(f"\n{'='*60}")
    print(f"Cache created successfully!")
    print(f"{'='*60}")
    print(f"  File: {cache_file.name}")
    print(f"  Size: {cache_size_mb:.2f} MB")
    print(f"  Images: {len(cache_data):,}")
    print(f"  Total captions: {total_captions:,}")
    print(f"  Avg captions/image: {total_captions/len(cache_data):.2f}")
    print(f"{'='*60}\n")

    return cache_file


# ========== Training Functions ==========

def setup_optimizer_and_scheduler(model, config, total_steps):
    """
    Setup optimizer and learning rate scheduler.

    Args:
        model: CLIPModel instance
        config: Training configuration dict
        total_steps: Total number of training steps

    Returns:
        optimizer: Optimizer instance
        scheduler: LR scheduler instance
    """
    # Optimizer - only trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Cosine annealing with warmup
    def lr_lambda(current_step):
        warmup_steps = config.get('warmup_steps', 500)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, config, device='cuda'):
    """
    Train for one epoch.

    Args:
        model: CLIPModel instance
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: LR scheduler
        epoch: Current epoch number
        config: Training configuration dict
        device: Device to use

    Returns:
        avg_loss: Average training loss
        avg_acc: Average training accuracy
        history: Training history for this epoch
    """
    model.train()
    model.text_encoder.eval()  # Keep text encoder in eval mode

    epoch_loss = 0
    epoch_acc = 0
    history = {'loss': [], 'acc': [], 'lr': []}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['image'].to(device)
        text_embeddings = batch['text_embedding'].to(device)

        # Forward pass
        image_embeddings = model.encode_image(images)

        # Compute loss
        loss, accuracy = criterion(image_embeddings, text_embeddings)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 1.0))

        optimizer.step()
        scheduler.step()

        # Update metrics
        epoch_loss += loss.item()
        epoch_acc += accuracy.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{accuracy.item():.3f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })

        # Log
        history['loss'].append(loss.item())
        history['acc'].append(accuracy.item())
        history['lr'].append(scheduler.get_last_lr()[0])

    avg_loss = epoch_loss / len(train_loader)
    avg_acc = epoch_acc / len(train_loader)

    return avg_loss, avg_acc, history


@torch.no_grad()
def validate(model, val_loader, criterion, device='cuda'):
    """
    Validate the model.

    Args:
        model: CLIPModel instance
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use

    Returns:
        avg_loss: Average validation loss
        avg_acc: Average validation accuracy
    """
    model.eval()

    total_loss = 0
    total_acc = 0

    for batch in tqdm(val_loader, desc="Validating"):
        images = batch['image'].to(device)
        text_embeddings = batch['text_embedding'].to(device)

        # Forward pass
        image_embeddings = model.encode_image(images)

        # Compute loss
        loss, accuracy = criterion(image_embeddings, text_embeddings)

        total_loss += loss.item()
        total_acc += accuracy.item()

    avg_loss = total_loss / len(val_loader)
    avg_acc = total_acc / len(val_loader)

    return avg_loss, avg_acc


# ========== Evaluation Functions ==========

@torch.no_grad()
def retrieve_images_for_text(query_text, model, tokenizer, dataset, image_embeddings, top_k=5, device='cuda'):
    """
    Retrieve top-K images for a text query.

    Args:
        query_text: String text query
        model: CLIP model
        tokenizer: CLIP tokenizer
        dataset: Dataset to retrieve images from
        image_embeddings: Pre-computed image embeddings [N, 512]
        top_k: Number of images to retrieve
        device: Device to use

    Returns:
        top_images: List of PIL images
        top_scores: List of similarity scores
        top_captions: List of captions
        top_indices: List of dataset indices
    """
    model.eval()

    # Encode query text
    inputs = tokenizer(
        [query_text],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    query_embedding = model.encode_text(inputs['input_ids'], inputs['attention_mask'])
    query_embedding = query_embedding.cpu()

    # Compute similarities
    similarities = torch.matmul(query_embedding, image_embeddings.T).squeeze(0)

    # Get top-K
    top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(similarities)), largest=True)

    # Retrieve images (temporarily enable raw image return)
    original_return_raw_image = getattr(dataset, 'return_raw_image', False)
    dataset.return_raw_image = True

    top_images = []
    top_captions = []

    for idx in top_indices:
        sample = dataset[idx.item()]
        if 'image_raw' in sample and sample['image_raw'] is not None:
            top_images.append(sample['image_raw'])
        top_captions.append(sample['caption'])

    # Restore original setting
    dataset.return_raw_image = original_return_raw_image

    return top_images, top_scores.tolist(), top_captions, top_indices.tolist()


@torch.no_grad()
def zero_shot_classify(image, class_labels, model, tokenizer, use_templates=True, device='cuda'):
    """
    Classify an image using zero-shot CLIP.

    Args:
        image: PIL Image or tensor [3, 224, 224]
        class_labels: List of class names (e.g., ['a person', 'an animal'])
        model: CLIP model
        tokenizer: CLIP tokenizer
        use_templates: If True, use prompt templates
        device: Device to use

    Returns:
        probs: Probability distribution over classes
        predicted_class: Index of predicted class
        class_scores: Raw similarity scores
    """
    from PIL import Image
    from torchvision import transforms
    from config import IMAGE_SIZE, CLIP_MEAN, CLIP_STD

    model.eval()

    # Prepare image
    if isinstance(image, Image.Image):
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
    else:
        image_tensor = image.unsqueeze(0).to(device)

    # Encode image
    image_embedding = model.encode_image(image_tensor)

    # Prepare text prompts
    if use_templates:
        templates = [
            'a photo of {}',
            'a picture of {}',
            'an image of {}',
        ]
        texts = []
        for label in class_labels:
            for template in templates:
                texts.append(template.format(label))
    else:
        texts = class_labels

    # Encode texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    text_embeddings = model.encode_text(inputs['input_ids'], inputs['attention_mask'])

    # Compute similarities
    if use_templates:
        text_embeddings = text_embeddings.view(len(class_labels), len(templates), -1)
        text_embeddings = text_embeddings.mean(dim=1)

    similarities = torch.matmul(image_embedding, text_embeddings.T).squeeze(0)

    # Convert to probabilities
    probs = F.softmax(similarities * 100, dim=0)
    predicted_class = torch.argmax(similarities).item()

    return probs.cpu(), predicted_class, similarities.cpu()


def plot_training_curves(history, save_path=None, flags_suffix=''):
    """
    Plot training curves.

    Args:
        history: Training history dict
        save_path: Path to save plot (optional)
        flags_suffix: Suffix for experimental flags
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    main_title = "Training Curves"
    if flags_suffix:
        main_title += f" (Modifications: {flags_suffix.replace('_', ' ').strip()})"
    fig.suptitle(main_title, fontsize=16, fontweight='bold')

    # Training loss
    if 'train_loss_steps' in history:
        axes[0, 0].plot(history['train_loss_steps'], history['train_loss'], alpha=0.6)
        axes[0, 0].set_xlabel('Steps')
    else:
        axes[0, 0].plot(history['train_loss'], alpha=0.6)
        axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Training accuracy
    if 'train_acc_steps' in history:
        axes[0, 1].plot(history['train_acc_steps'], history['train_acc'], alpha=0.6)
        axes[0, 1].set_xlabel('Steps')
    else:
        axes[0, 1].plot(history['train_acc'], alpha=0.6)
        axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].grid(True, alpha=0.3)

    # Validation loss
    axes[1, 0].plot(history['val_loss'], 'o-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Validation Loss')
    axes[1, 0].grid(True, alpha=0.3)

    # Validation accuracy
    axes[1, 1].plot(history['val_acc'], 'o-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
