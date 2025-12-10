# CLIP Fine-tuning on MS COCO 2014

Complete implementation for fine-tuning a CLIP-style vision-language model on the MS COCO 2014 dataset, as required for ELEC 475 Lab 4.

## Project Structure

This project is split into **three separate notebooks** for clarity:

1. **`coco_dataset_prep.ipynb`** - Dataset preparation (run once)
   - Downloads COCO 2014 dataset
   - Caches text embeddings
   - Creates PyTorch datasets

2. **`train_clip.ipynb`** - Model training
   - Defines model architecture
   - Implements InfoNCE loss
   - Training loop with logging
   - Validation and metrics
   - Loss curve visualization

3. **`evaluate_clip.ipynb`** - Evaluation and visualization
   - Recall@K metrics (Image↔Text retrieval)
   - Text query → Image retrieval
   - Zero-shot image classification
   - Visualization of retrieval results

## Features

- **Automated Kaggle Download**: Automatically downloads COCO 2014 dataset from Kaggle
- **CLIP Preprocessing**: Images normalized with CLIP-specific mean/std values
- **Text Embedding Caching**: Pre-encodes all captions to save GPU memory and training time
- **PyTorch Dataset**: Custom Dataset class for efficient data loading
- **ResNet50 Image Encoder**: Pretrained ImageNet weights for image feature extraction
- **Projection Head**: 2-layer MLP with GELU activation for CLIP space alignment
- **Visualization Tools**: Verify dataset integrity with image-caption pair displays

## Dataset Information

- **Training Images**: ~82,783 images
- **Validation Images**: ~40,504 images
- **Captions**: Multiple captions per image (typically 5)
- **Source**: [COCO 2014 Dataset on Kaggle](https://www.kaggle.com/datasets/jeffaudi/coco-2014-dataset-for-yolov3)

## Model Architecture

The notebook implements a CLIP-style vision-language model with the following components:

### 1. Text Encoder (Frozen)
- **Model**: Pretrained CLIP text encoder from HuggingFace (`openai/clip-vit-base-patch32`)
- **Status**: **FROZEN** - Parameters not updated during training
- **Output**: 512-dimensional text embeddings
- **Purpose**: Provides the target embedding space for image alignment

### 2. Image Encoder (Trainable)
- **Architecture**: ResNet50
- **Initialization**: Pretrained ImageNet weights
- **Status**: **TRAINABLE** - Fine-tuned during training
- **Output**: 2048-dimensional image features
- **Parameters**: ~23M

### 3. Projection Head (Trainable)
- **Architecture**: Two-layer MLP with GELU activation
- **Dimensions**: 2048 → 1024 → 512
- **Status**: **TRAINABLE** - Learns to map image features to CLIP space
- **Parameters**: ~2.6M
- **Activation**: GELU (Gaussian Error Linear Unit)

### Training Strategy
- Only the **image encoder** and **projection head** are trained
- Text encoder remains frozen with pretrained CLIP weights
- Total trainable parameters: ~26M (~70% of model)
- Embeddings are L2-normalized before computing similarity

## Quick Start (Google Colab)

### Step 1: Dataset Preparation

1. **Upload notebook to Colab**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `coco_dataset_prep.ipynb`

2. **Get Kaggle credentials**
   - Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Download your `kaggle.json` API token
   - Upload it to Colab using the file browser

3. **Run dataset preparation**
   - Run all cells (Runtime → Run all)
   - Downloads ~13GB dataset (10-20 minutes)
   - Encodes and caches text embeddings (5-10 minutes)
   - Verifies data integrity

### Step 2: Training

1. **Upload training notebook**
   - In the same Colab session, upload `train_clip.ipynb`
   - Or open it in a new session (dataset must be prepared first)

2. **Run training**
   - Run all cells to start training
   - Configurable hyperparameters in cell 2
   - Training progress shown with tqdm progress bars
   - Checkpoints saved automatically

3. **Monitor results**
   - Training/validation loss and accuracy logged
   - Loss curves plotted automatically
   - Best model saved based on validation loss

### Step 3: Evaluation

1. **Upload evaluation notebook**
   - Upload `evaluate_clip.ipynb` to Colab
   - Requires trained model checkpoint

2. **Run evaluation**
   - Computes Recall@K metrics
   - Retrieves images for text queries
   - Performs zero-shot classification

3. **View results**
   - Interactive visualizations
   - Text query → top-5 images
   - Image classification with probabilities

## Notebook Details

### `coco_dataset_prep.ipynb` Structure

**Cells 1-2:** Setup
- Install dependencies
- Import libraries
- Check GPU availability

**Cells 3-4:** Dataset Download
- Configure Kaggle authentication
- Download COCO 2014 (~13GB)
- Verify directory structure

**Cells 5-6:** CLIP Text Encoder & Preprocessing
- Load pretrained CLIP text encoder
- Configure image transforms (224×224, normalized)

**Cells 7-8:** Caption Encoding
- Encode all captions with CLIP text encoder
- Cache to disk: `train_text_embeddings.pt`, `val_text_embeddings.pt`

**Cells 9-10:** PyTorch Dataset
- Define `COCOClipDataset` class
- Create train/val dataset instances

**Cells 11-12:** Verification
- Visualize image-caption pairs
- Run integrity checks

### `train_clip.ipynb` Structure

**Cells 1-2:** Setup & Configuration
- Install dependencies
- Configure hyperparameters (batch size, learning rate, etc.)

**Cells 3-4:** Dataset & Model Architecture
- Load prepared datasets
- Define ResNet50 image encoder
- Define projection head (2048 → 1024 → 512)
- Combine into CLIP model

**Cell 5:** InfoNCE Loss
- Implement symmetric contrastive loss
- Temperature-scaled cosine similarity

**Cells 6-7:** Optimizer & Scheduler
- AdamW optimizer
- Cosine annealing with warmup

**Cells 8-9:** Training Loop
- Training with gradient clipping
- Validation after each epoch
- Checkpoint saving

**Cells 10-11:** Results & Reporting
- Plot training curves
- Generate training report
- Hardware and timing information

### `evaluate_clip.ipynb` Structure

**Cells 1-2:** Setup & Load Model
- Import libraries
- Load trained model checkpoint
- Load tokenizer and text encoder

**Cell 3:** Compute Embeddings
- Encode all validation images
- Pre-compute embeddings for fast retrieval

**Cell 4:** Recall@K Metrics
- Compute similarity matrix (Image × Text)
- Calculate Recall@1, @5, @10
- Image→Text and Text→Image retrieval

**Cell 5:** Text Query Visualization
- Retrieve top-K images for text queries
- Examples: 'sport', 'a cat', 'food on a plate'
- Display with similarity scores

**Cells 6-7:** Zero-Shot Classification
- Classify images using text prompts
- Prompt template ensembling
- Examples: ['a person', 'an animal', 'a landscape']
- Probability visualization

**Cell 8:** Evaluation Summary
- Comprehensive report
- Save evaluation results

## Experimental Modification Flags

The training notebook includes **experimental flags** to test architectural modifications that may improve accuracy:

### 🔬 Available Flags (Cell 2 in `train_clip.ipynb`)

```python
CONFIG = {
    # === EXPERIMENTAL FLAGS ===
    'use_batch_norm': False,        # Add BatchNorm to projection head
    'use_attention_pooling': False, # Use attention pooling in ResNet50
    # ==========================
}
```

### Flag 1: `use_batch_norm` (Normalization)

**What it does:**
- Adds BatchNorm1d layers after each linear layer in the projection head
- Architecture becomes: `Linear → BatchNorm → GELU → Linear → BatchNorm`

**Hypothesis:**
- BatchNorm stabilizes training by normalizing activations
- Reduces internal covariate shift in the projection head
- May allow higher learning rates and faster convergence
- Could improve generalization through implicit regularization

**Expected Impact:**
- ✅ More stable training (less loss variance)
- ✅ Potentially faster convergence
- ✅ Better gradient flow
- ⚠️ Slight increase in parameters (~2K)

### Flag 2: `use_attention_pooling` (Architecture)

**What it does:**
- Replaces global average pooling with learned attention pooling
- Computes spatial attention weights over ResNet50's feature maps
- Weighted sum of features instead of simple averaging

**Architecture:**
```python
# Standard: Global Average Pooling
features [B, 2048, 7, 7] → avgpool → [B, 2048]

# Attention: Learned Weighted Pooling
features [B, 2048, 7, 7] → attention weights [B, 49, 1]
                         → weighted sum → [B, 2048]
```

**Hypothesis:**
- Attention mechanism focuses on discriminative regions
- More flexible than fixed average pooling
- Can learn to emphasize important spatial locations
- Better alignment with text descriptions (which describe salient objects)

**Expected Impact:**
- ✅ Better feature representation
- ✅ Improved image-text alignment
- ✅ Higher Recall@K scores
- ⚠️ Additional parameters (~128K)

### Using the Flags

**To enable modifications:**

1. Open `train_clip.ipynb`
2. In Cell 2 (Configuration), set flags to `True`:
   ```python
   'use_batch_norm': True,
   'use_attention_pooling': True,
   ```
3. Run the notebook as normal

**Automatic Output Organization:**

All outputs are automatically organized by flags:

```
/content/
├── checkpoints/                 # Baseline
│   └── best_model.pt
├── checkpoints_bn/              # BatchNorm only
│   └── best_model_bn.pt
├── checkpoints_attn/            # Attention only
│   └── best_model_attn.pt
├── checkpoints_bn_attn/         # Both flags
│   └── best_model_bn_attn.pt
├── logs/
│   ├── training_curves.png      # Baseline
│   ├── training_curves_bn.png   # BatchNorm
│   └── training_report_bn.txt
```

**Flag Suffixes:**
- No flags: *(no suffix)*
- `use_batch_norm=True`: `_bn`
- `use_attention_pooling=True`: `_attn`
- Both: `_bn_attn`

### Comparing Results

To compare different configurations:

1. **Train baseline:**
   - Set both flags to `False`
   - Train and note results

2. **Train with BatchNorm:**
   - Set `use_batch_norm=True`
   - Train (outputs go to `*_bn` files)

3. **Train with Attention:**
   - Set `use_attention_pooling=True`
   - Train (outputs go to `*_attn` files)

4. **Train with both:**
   - Set both to `True`
   - Train (outputs go to `*_bn_attn` files)

5. **Compare:**
   - Check validation loss/accuracy in training reports
   - Run evaluation notebook on each checkpoint
   - Compare Recall@K metrics

### Expected Results

Based on architectural intuition:

| Configuration | Expected Val Acc | Expected Recall@5 | Training Speed |
|--------------|------------------|-------------------|----------------|
| Baseline | 0.45-0.50 | 55-60% | Fastest |
| + BatchNorm | 0.48-0.53 | 57-62% | Similar |
| + Attention | 0.50-0.55 | 60-65% | 10% slower |
| + Both | 0.52-0.57 | 62-67% | 10% slower |

*Note: Actual results depend on training time, data augmentation, and other factors.*

## Training Configuration

The training notebook uses carefully selected hyperparameters optimized for Colab:

### Hyperparameters (Configurable in Cell 2)

```python
CONFIG = {
    # Training
    'batch_size': 128,           # Adjust based on GPU memory
    'num_epochs': 10,
    'learning_rate': 1e-4,       # AdamW learning rate
    'weight_decay': 0.01,        # L2 regularization
    'warmup_steps': 500,         # LR warmup steps
    'temperature': 0.07,         # InfoNCE temperature

    # Optimization
    'optimizer': 'AdamW',
    'scheduler': 'cosine',       # Cosine annealing
    'max_grad_norm': 1.0,        # Gradient clipping
}
```

### Rationale for Hyperparameter Choices

**Batch Size (128)**
- Large enough for stable contrastive learning
- Fits in Colab's ~15GB GPU memory
- More negative samples = better contrastive loss

**Learning Rate (1e-4)**
- Conservative for fine-tuning
- ResNet50 already pretrained on ImageNet
- Prevents catastrophic forgetting

**Temperature (0.07)**
- Standard CLIP temperature
- Sharpens similarity distribution
- Balances hard vs. soft negatives

**Warmup (500 steps)**
- Stabilizes early training
- Prevents large gradient spikes
- Common practice for vision-language models

**Weight Decay (0.01)**
- Regularization for projection head
- Prevents overfitting on COCO captions
- Standard AdamW decay value

## Dataset Class API

### COCOClipDataset

```python
dataset = COCOClipDataset(
    split='train',              # 'train' or 'val'
    transform=image_transforms, # Optional custom transforms
    return_all_captions=False   # Return all captions or random one
)
```

**Returns** (when `return_all_captions=False`):
- `image`: Tensor [3, 224, 224] - Preprocessed image
- `text_embedding`: Tensor [512] - CLIP text embedding
- `caption`: String - Original caption text
- `image_id`: Integer - COCO image ID

**Returns** (when `return_all_captions=True`):
- `image`: Tensor [3, 224, 224] - Preprocessed image
- `text_embeddings`: Tensor [num_captions, 512] - All embeddings for this image
- `captions`: List[str] - All captions for this image
- `image_id`: Integer - COCO image ID

## File Structure

```
project/
├── coco_dataset_prep.ipynb    # Dataset preparation notebook
├── train_clip.ipynb           # Training notebook
├── evaluate_clip.ipynb        # Evaluation notebook
├── example_usage.py           # Example Python script
├── requirements.txt           # Dependencies
└── README.md                  # This file

# Created during execution (in Colab):
/content/
├── coco2014/
│   ├── train2014/             # 82,783 training images
│   ├── val2014/               # 40,504 validation images
│   ├── annotations/           # Caption JSON files
│   ├── train_text_embeddings.pt  # Cached embeddings (~150MB)
│   └── val_text_embeddings.pt    # Cached embeddings (~75MB)
├── checkpoints/
│   ├── best_model.pt          # Best model checkpoint
│   └── checkpoint_epoch_*.pt  # Per-epoch checkpoints
└── logs/
    ├── training_curves.png    # Loss/accuracy plots
    ├── training_summary.pt    # Full training history
    ├── training_report.txt    # Text report
    └── evaluation_results.pt  # Recall@K metrics
```

## Requirements

All requirements are automatically installed by the notebook:

- `transformers>=4.30.0` - HuggingFace Transformers for CLIP
- `torch>=2.0.0` - PyTorch
- `torchvision>=0.15.0` - Image transforms
- `pillow` - Image loading
- `kaggle` - Kaggle API
- `pycocotools` - COCO utilities
- `matplotlib` - Visualization
- `tqdm` - Progress bars

## Memory Requirements

- **RAM**: Minimum 12GB recommended
- **GPU**: Optional but recommended for faster caption encoding
  - Colab free tier provides ~15GB GPU memory
- **Disk Space**: ~15GB for dataset + embeddings

## Troubleshooting

### Kaggle Authentication Failed
- Ensure `kaggle.json` is uploaded to Colab
- Check that the file contains valid credentials
- Re-run Cell 3

### Out of Memory During Encoding
- Reduce `batch_size` in cells 7-8 (try 32 or 16)
- Restart runtime and clear outputs before re-running

### Missing Images
- Verify dataset download completed successfully
- Check that all expected directories exist in `/content/coco2014/`
- Re-run Cell 4 to download again

### Slow Download
- Kaggle download speed depends on server load and network
- Expected time: 10-20 minutes for ~13GB
- Consider running during off-peak hours

## Performance Tips

1. **Enable GPU**: Runtime → Change runtime type → GPU
2. **High RAM**: Runtime → Change runtime type → High-RAM (if available)
3. **Keep Session Alive**: Colab sessions timeout after inactivity
4. **Save Cache Files**: Download `*_text_embeddings.pt` files to Google Drive for reuse

## Expected Training Results

### Training Metrics

The training notebook automatically tracks and reports:

1. **Training Loss Curves**
   - Per-step training loss
   - Per-epoch validation loss
   - Should decrease steadily over epochs

2. **Accuracy Metrics**
   - Batch-level accuracy (% correct matches)
   - Image-to-text retrieval accuracy
   - Text-to-image retrieval accuracy

3. **Hardware & Timing**
   - GPU name and memory
   - Total training time
   - Steps per second

### Interpreting Results

**Good Training Indicators:**
- ✓ Validation loss decreases over epochs
- ✓ Training and validation accuracy both increase
- ✓ Gap between train/val loss remains small (no overfitting)
- ✓ Learning rate smoothly decays after warmup

**Common Issues & Solutions:**

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Loss doesn't decrease | Learning rate too low | Increase LR to 3e-4 |
| Loss explodes (NaN) | Learning rate too high | Reduce LR to 5e-5 |
| High train acc, low val acc | Overfitting | Increase weight decay or reduce epochs |
| Training very slow | Batch size too small | Increase batch size (if GPU memory allows) |
| Out of memory | Batch size too large | Reduce batch size to 64 or 32 |

### Saved Outputs

After training completes, you'll have:

```
/content/checkpoints/
├── best_model.pt              # Best model (lowest val loss)
└── checkpoint_epoch_X.pt      # Checkpoints per epoch

/content/logs/
├── training_curves.png        # Loss and accuracy plots
├── training_summary.pt        # Full training history
└── training_report.txt        # Text report with metrics
```

## Evaluation Capabilities

The `evaluate_clip.ipynb` notebook provides comprehensive evaluation:

### 1. Recall@K Metrics

Measures retrieval performance for both directions:

**Image → Text Retrieval:**
- Given an image, find its matching captions
- Reports Recall@1, Recall@5, Recall@10

**Text → Image Retrieval:**
- Given a caption, find its matching image
- Reports Recall@1, Recall@5, Recall@10

**Example Output:**
```
Image → Text Retrieval:
  Recall@1: 0.3245 (32.45%)
  Recall@5: 0.6123 (61.23%)
  Recall@10: 0.7456 (74.56%)

Text → Image Retrieval:
  Recall@1: 0.2987 (29.87%)
  Recall@5: 0.5834 (58.34%)
  Recall@10: 0.7123 (71.23%)
```

### 2. Text Query → Image Retrieval

Search for images using natural language:
- **Query**: "sport" → Returns top-5 sports images
- **Query**: "a cat" → Returns top-5 cat images
- **Query**: "food on a plate" → Returns top-5 food images

Each result shows:
- Retrieved image
- Similarity score
- Ground truth caption

### 3. Zero-Shot Classification

Classify images without training:
- **Input**: Image + Class options
- **Output**: Probability distribution
- **Example**: Classify as ['a person', 'an animal', 'a landscape']

Features:
- Prompt template ensembling (like CLIP paper)
- Visual probability bars
- Confidence scores

### 4. Custom Classification Tasks

Pre-built examples:
- Indoor vs Outdoor scenes
- Activity classification (eating, playing, working)
- Object detection (dog, cat, bird, etc.)
- Easily adaptable to your own classes

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [HuggingFace CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
- [COCO Dataset](https://cocodataset.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## License

This code is for educational purposes (ELEC 475 Lab 4). The COCO dataset has its own license terms.
