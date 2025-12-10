"""
Model architectures for CLIP fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AttentionPooling(nn.Module):
    """Attention-based pooling for better feature aggregation."""

    def __init__(self, input_dim=2048):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 8),
            nn.Tanh(),
            nn.Linear(input_dim // 8, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] feature maps

        Returns:
            pooled: [B, C] pooled features
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]

        # Compute attention weights
        attn_weights = self.attention(x_flat)  # [B, H*W, 1]
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum
        pooled = torch.sum(x_flat * attn_weights, dim=1)  # [B, C]
        return pooled


class ResNet50ImageEncoder(nn.Module):
    """ResNet50 image encoder with optional attention pooling."""

    def __init__(self, pretrained=True, use_attention_pooling=False):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)

        if use_attention_pooling:
            # Remove final avgpool and fc, keep conv layers
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            self.pooling = AttentionPooling(input_dim=2048)
            self.use_attention = True
        else:
            # Remove only final fc layer
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            self.pooling = None
            self.use_attention = False

        self.output_dim = 2048

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input images

        Returns:
            features: [B, 2048] image features
        """
        features = self.features(x)

        if self.use_attention:
            pooled = self.pooling(features)
        else:
            pooled = features.view(features.size(0), -1)

        return pooled


class ProjectionHead(nn.Module):
    """2-layer MLP projection head with optional BatchNorm."""

    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=512, use_batch_norm=False):
        super().__init__()

        if use_batch_norm:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        """
        Args:
            x: [B, input_dim] input features

        Returns:
            projected: [B, output_dim] projected features
        """
        return self.projection(x)


class CLIPModel(nn.Module):
    """Combined CLIP model with image encoder, text encoder, and projection head."""

    def __init__(self, text_encoder, freeze_text_encoder=True,
                 use_batch_norm=False, use_attention_pooling=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.use_attention_pooling = use_attention_pooling

        # Text encoder (frozen)
        self.text_encoder = text_encoder
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder.eval()

        # Image encoder (trainable)
        self.image_encoder = ResNet50ImageEncoder(
            pretrained=True,
            use_attention_pooling=use_attention_pooling
        )

        # Projection head (trainable)
        self.projection_head = ProjectionHead(
            input_dim=2048,
            hidden_dim=1024,
            output_dim=512,
            use_batch_norm=use_batch_norm
        )

    def encode_image(self, images):
        """
        Encode images to normalized embeddings.

        Args:
            images: [B, 3, H, W] input images

        Returns:
            embeddings: [B, 512] normalized image embeddings
        """
        features = self.image_encoder(images)
        embeddings = self.projection_head(features)
        return F.normalize(embeddings, p=2, dim=1)

    def encode_text(self, input_ids, attention_mask):
        """
        Encode text to normalized embeddings.

        Args:
            input_ids: [B, seq_len] tokenized text
            attention_mask: [B, seq_len] attention mask

        Returns:
            embeddings: [B, 512] normalized text embeddings
        """
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.pooler_output
            return F.normalize(embeddings, p=2, dim=1)

    def forward(self, images, input_ids, attention_mask):
        """
        Forward pass through both encoders.

        Args:
            images: [B, 3, H, W] input images
            input_ids: [B, seq_len] tokenized text
            attention_mask: [B, seq_len] attention mask

        Returns:
            image_embeddings: [B, 512] normalized image embeddings
            text_embeddings: [B, 512] normalized text embeddings
        """
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        return image_embeddings, text_embeddings


class InfoNCELoss(nn.Module):
    """Symmetric contrastive loss for CLIP training."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        """
        Compute symmetric InfoNCE loss.

        Args:
            image_embeddings: [B, D] normalized image features
            text_embeddings: [B, D] normalized text features

        Returns:
            loss: Scalar contrastive loss
            accuracy: Batch accuracy (for logging)
        """
        batch_size = image_embeddings.shape[0]

        # Compute similarity matrix [B, B]
        logits = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=logits.device)

        # Image-to-text loss
        loss_i2t = F.cross_entropy(logits, labels)

        # Text-to-image loss
        loss_t2i = F.cross_entropy(logits.T, labels)

        # Symmetric loss
        loss = (loss_i2t + loss_t2i) / 2

        # Compute accuracy for logging
        with torch.no_grad():
            pred_i2t = torch.argmax(logits, dim=1)
            pred_t2i = torch.argmax(logits.T, dim=1)
            accuracy = ((pred_i2t == labels).float().mean() +
                       (pred_t2i == labels).float().mean()) / 2

        return loss, accuracy
