# CLIP Fine-tuning on MS COCO 2014

Complete implementation for fine-tuning a CLIP-style vision-language model on MS COCO 2014 for ELEC 475 Lab 4.

## Quick Start

1. **Open `main.ipynb`** - Single notebook containing the entire pipeline
2. **Configure hyperparameters** - Modify Section 2 (Configuration) to set flags and hyperparameters
3. **Run all cells** - Complete workflow from dataset prep → training → evaluation

## Project Structure

```
project/
├── main.ipynb          # Complete pipeline (dataset prep, training, evaluation)
├── config.py           # Configuration constants
├── models.py           # Model architectures (CLIPModel, InfoNCELoss, etc.)
├── dataset.py          # Dataset classes (COCOClipDataset)
└── utils.py            # Training, evaluation, and visualization functions
```

All complex code is in Python modules. The notebook contains only high-level workflow.

## Features

- **Single Notebook Workflow**: Everything in one place - from dataset prep to evaluation
- **Modular Code**: Complex logic in importable Python modules
- **Experimental Flags**: Easy A/B testing with `use_batch_norm` and `use_attention_pooling`
- **Automatic Organization**: Outputs organized by flags (e.g., `best_model_bn.pt`, `training_curves_attn.png`)
- **Text Embedding Caching**: Pre-encodes captions to save time and memory during training

## Model Architecture

| Component | Details | Status |
|-----------|---------|--------|
| **Text Encoder** | CLIP ViT-B/32 (pretrained) | Frozen |
| **Image Encoder** | ResNet50 (ImageNet pretrained) | Trainable |
| **Projection Head** | 2048→1024→512 MLP with GELU | Trainable |
| **Loss** | InfoNCE (temperature=0.07) | - |

**Training Strategy**: Only image encoder + projection head are trained (~26M parameters)

## Experimental Flags

Set in Section 2 of `main.ipynb`:

```python
CONFIG['use_batch_norm'] = False        # Add BatchNorm to projection head
CONFIG['use_attention_pooling'] = False  # Use attention pooling instead of avgpool
```

**Hypotheses**:
- `use_batch_norm`: Stabilizes training, may improve convergence
- `use_attention_pooling`: Focuses on discriminative regions, may improve retrieval accuracy

**Output Organization**: Flags automatically suffix all outputs (`_bn`, `_attn`, `_bn_attn`)

## Configuration

Key hyperparameters (configurable in notebook Section 2):

```python
batch_size: 128
num_epochs: 10
learning_rate: 1e-4
temperature: 0.07
warmup_steps: 500
```

## Dataset

- **Source**: [COCO 2014 on Kaggle](https://www.kaggle.com/datasets/jeffaudi/coco-2014-dataset-for-yolov3)
- **Train**: ~82,783 images
- **Val**: ~40,504 images
- **Preprocessing**: 224×224, CLIP normalization

**Kaggle Setup**: Upload your `kaggle.json` to Colab before running Section 3.

## Evaluation Metrics

The notebook automatically computes:

1. **Recall@K**: Image↔Text retrieval (K=1,5,10)
2. **Text Queries**: Retrieve top-5 images for text queries (e.g., "sport", "a cat")
3. **Zero-Shot Classification**: Classify images using text prompts

## Requirements

Auto-installed by notebook:
- PyTorch ≥2.0.0
- transformers ≥4.30.0
- torchvision, pillow, matplotlib, tqdm

## Usage Example

```python
# Simple training script using modules
from config import DEFAULT_CONFIG
from models import CLIPModel, InfoNCELoss
from dataset import COCOClipDataset
from utils import train_epoch, validate

# Load data
train_dataset = COCOClipDataset(split='train')

# Create model with flags
model = CLIPModel(text_encoder, use_batch_norm=True)

# Train
train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch=0, config=DEFAULT_CONFIG)
```

## Memory Requirements

- **GPU**: ~15GB (Colab free tier sufficient)
- **Disk**: ~15GB for dataset + embeddings
- **RAM**: 12GB recommended

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Radford et al., 2021
- [COCO Dataset](https://cocodataset.org/)
- [HuggingFace CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
