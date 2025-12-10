# COCO 2014 Dataset Preparation for CLIP Fine-tuning

This project prepares the MS COCO 2014 dataset for fine-tuning CLIP models, as required for ELEC 475 Lab 4.

## Features

- **Automated Kaggle Download**: Automatically downloads COCO 2014 dataset from Kaggle
- **CLIP Preprocessing**: Images normalized with CLIP-specific mean/std values
- **Text Embedding Caching**: Pre-encodes all captions to save GPU memory and training time
- **PyTorch Dataset**: Custom Dataset class for efficient data loading
- **Visualization Tools**: Verify dataset integrity with image-caption pair displays

## Dataset Information

- **Training Images**: ~82,783 images
- **Validation Images**: ~40,504 images
- **Captions**: Multiple captions per image (typically 5)
- **Source**: [COCO 2014 Dataset on Kaggle](https://www.kaggle.com/datasets/jeffaudi/coco-2014-dataset-for-yolov3)

## Quick Start (Google Colab)

### 1. Upload to Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `coco_dataset_prep.ipynb`

### 2. Get Kaggle API Credentials

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
2. Scroll to **API** section
3. Click **Create New API Token**
4. Download `kaggle.json` file
5. In Colab, use the file browser (left sidebar) to upload `kaggle.json`

### 3. Run the Notebook

Run all cells in order (Runtime → Run all). The notebook will:

1. Install dependencies
2. Setup Kaggle authentication
3. Download COCO 2014 dataset (~13GB, takes 10-20 minutes)
4. Load CLIP text encoder
5. Encode and cache all captions (~5-10 minutes)
6. Create PyTorch Dataset instances
7. Display verification visualizations

## Notebook Structure

### Cell 1-2: Setup
- Install required packages
- Import libraries
- Check GPU availability

### Cell 3-4: Dataset Download
- Configure Kaggle authentication
- Download and extract COCO 2014 dataset
- Verify directory structure

### Cell 5-6: CLIP Model & Preprocessing
- Load CLIP text encoder from HuggingFace
- Configure image transforms with CLIP normalization:
  - **Size**: 224×224
  - **Mean**: [0.48145466, 0.4578275, 0.40821073]
  - **Std**: [0.26862954, 0.26130258, 0.27577711]

### Cell 7-8: Caption Encoding
- Encode all captions using CLIP text encoder
- Cache embeddings to `.pt` files:
  - `train_text_embeddings.pt` (~100-200MB)
  - `val_text_embeddings.pt` (~50-100MB)

### Cell 9-10: PyTorch Dataset
- Define `COCOClipDataset` class
- Create train/val dataset instances
- Features:
  - On-the-fly image loading (memory efficient)
  - Pre-computed text embeddings
  - Random caption selection per epoch

### Cell 11-12: Verification
- Visualize random image-caption pairs
- Run integrity checks
- Display dataset statistics

## Usage After Preparation

Once the dataset is prepared, you can use it for training:

```python
from torch.utils.data import DataLoader

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

# Iterate through batches
for batch in train_loader:
    images = batch['image']           # [batch_size, 3, 224, 224]
    text_embeddings = batch['text_embedding']  # [batch_size, 512]
    captions = batch['caption']       # List of strings
    image_ids = batch['image_id']    # List of integers

    # Your training code here
    pass
```

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
├── coco_dataset_prep.ipynb    # Main notebook
├── README.md                  # This file
└── /content/coco2014/         # Created by notebook (in Colab)
    ├── train2014/             # Training images
    ├── val2014/               # Validation images
    ├── annotations/           # Caption JSON files
    ├── train_text_embeddings.pt  # Cached training embeddings
    └── val_text_embeddings.pt    # Cached validation embeddings
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

## Next Steps

After completing dataset preparation:

1. **Fine-tune CLIP**: Implement training loop with contrastive loss
2. **Evaluate**: Test model on validation set
3. **Experiment**: Try different architectures or training strategies

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [HuggingFace CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
- [COCO Dataset](https://cocodataset.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## License

This code is for educational purposes (ELEC 475 Lab 4). The COCO dataset has its own license terms.
