"""
Configuration constants for CLIP fine-tuning on MS COCO 2014.
"""

from pathlib import Path

# Model configuration
MODEL_NAME = 'openai/clip-vit-base-patch32'
IMAGE_SIZE = 224
EMBEDDING_DIM = 512
RESNET_OUTPUT_DIM = 2048
PROJECTION_HIDDEN_DIM = 1024

# CLIP normalization statistics
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Default paths (can be overridden)
DEFAULT_DATASET_DIR = Path('/content/coco2014')
DEFAULT_CHECKPOINT_DIR = Path('/content/checkpoints')
DEFAULT_LOG_DIR = Path('/content/logs')

# Training defaults
DEFAULT_CONFIG = {
    # Training
    'batch_size': 128,
    'num_epochs': 10,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'temperature': 0.07,

    # Optimization
    'optimizer': 'AdamW',
    'scheduler': 'cosine',
    'max_grad_norm': 1.0,

    # Experimental flags
    'use_batch_norm': False,
    'use_attention_pooling': False,

    # Paths
    'dataset_dir': DEFAULT_DATASET_DIR,
    'checkpoint_dir': DEFAULT_CHECKPOINT_DIR,
    'log_dir': DEFAULT_LOG_DIR,
}
