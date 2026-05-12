"""
model_multitask.py - Dual-Head UNet for Segmentation + Classification
Brain Tumor Segmentation - BRISC 2025 Dataset

Architecture:
    Shared Encoder: EfficientNet-B4 (ImageNet pretrained)
    Head 1 - Segmentation: UNet decoder + SCSE attention -> (B, 4, 256, 256) logits
    Head 2 - Classification: GAP + FC -> (B, 4) logits
        Classification label = dominant tumor class in the image
        (0=background, 1=glioma, 2=meningioma, 3=pituitary)

    The idea: segmentation draws the tumor boundary (where),
    classification identifies the tumor type (what).
    Shared encoder learns richer features because both tasks demand
    different information from the same feature maps.

Usage:
    from model_multitask import get_multitask_model, predict_mask, derive_cls_label

    model = get_multitask_model(device)         # encoder frozen
    seg_logits, cls_logits = model(images)      # forward pass
    cls_targets = derive_cls_label(masks)       # extract classification labels
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# #############################################################################
# CONFIGURATION
# #############################################################################
NUM_CLASSES       = 4
ENCODER_NAME      = "efficientnet-b4"
ENCODER_WEIGHTS   = "imagenet"
IN_CHANNELS       = 3
IMG_SIZE          = 256
ENCODER_OUT_CH    = 1792   # efficientnet-b4 last stage output channels

CLASS_NAMES = {0: "background", 1: "glioma", 2: "meningioma", 3: "pituitary"}


# #############################################################################
# DUAL-HEAD UNET MODEL
# #############################################################################
class DualHeadUNet(nn.Module):
    """UNet with two output heads: segmentation + classification.

    The segmentation decoder uses SCSE attention on skip connections
    to suppress background noise (same as our single-head model).

    The classification head takes the deepest encoder features,
    applies global average pooling, and outputs a 4-class prediction.
    """

    def __init__(self, device):
        super().__init__()

        # Build the base UNet (we'll use its encoder + seg decoder)
        self.unet = smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
            activation=None,
            decoder_attention_type="scse",
        )

        # Convenience accessors (match model.py interface)
        self.encoder = self.unet.encoder
        self.decoder = self.unet.decoder
        self.segmentation_head = self.unet.segmentation_head

        # Classification head: GAP -> Dropout -> FC
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # (B, 1792, H, W) -> (B, 1792, 1, 1)
            nn.Flatten(),                   # (B, 1792)
            nn.Dropout(0.3),
            nn.Linear(ENCODER_OUT_CH, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES),    # (B, 4) raw logits
        )

        # Move to device
        self.to(device)

        # Freeze encoder by default
        self.freeze_encoder()

    def forward(self, x):
        """Forward pass through both heads.

        Args:
            x: (B, 3, H, W) input tensor

        Returns:
            seg_logits: (B, 4, H, W) segmentation logits
            cls_logits: (B, 4) classification logits
        """
        # Encoder forward (list of feature maps at different scales)
        features = self.encoder(x)

        # Segmentation head: decoder + final conv
        decoder_out = self.decoder(*features)
        seg_logits = self.segmentation_head(decoder_out)

        # Classification head: use deepest encoder features (last element)
        deepest = features[-1]  # (B, 1792, 8, 8) for 256x256 input
        cls_logits = self.cls_head(deepest)

        return seg_logits, cls_logits

    def freeze_encoder(self):
        """Freeze encoder parameters (epochs 1-5)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning (after epoch 5)."""
        for param in self.encoder.parameters():
            param.requires_grad = True


# #############################################################################
# FACTORY FUNCTION
# #############################################################################
def get_multitask_model(device):
    """Create a DualHeadUNet with frozen encoder.

    Args:
        device: torch.device

    Returns:
        DualHeadUNet on the specified device, encoder frozen
    """
    return DualHeadUNet(device)


# #############################################################################
# INFERENCE HELPERS
# #############################################################################
def predict_mask(model, image_tensor, device):
    """Run segmentation inference.

    Args:
        model: DualHeadUNet
        image_tensor: (B, 3, H, W) float32
        device: torch.device

    Returns:
        preds: (B, H, W) int64, values in {0, 1, 2, 3}
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        seg_logits, _ = model(image_tensor)
        probs = torch.softmax(seg_logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds


def get_class_masks(pred_mask):
    """Extract per-class binary masks from argmax labels.

    Args:
        pred_mask: (B, H, W) int64

    Returns:
        tuple of 4 binary masks, each (B, H, W) float32
    """
    return (
        (pred_mask == 0).float(),
        (pred_mask == 1).float(),
        (pred_mask == 2).float(),
        (pred_mask == 3).float(),
    )


# #############################################################################
# CLASSIFICATION LABEL EXTRACTION
# #############################################################################
def derive_cls_label(mask_tensor):
    """Derive per-image classification label from segmentation mask.

    For each image in the batch:
        - If any tumor pixels exist, label = most frequent tumor class
        - If only background (class 0), label = 0

    Args:
        mask_tensor: (B, H, W) int64, values in {0, 1, 2, 3}

    Returns:
        labels: (B,) int64 tensor
    """
    B = mask_tensor.shape[0]
    labels = torch.zeros(B, dtype=torch.long, device=mask_tensor.device)

    for i in range(B):
        m = mask_tensor[i]
        # Count pixels per tumor class (1, 2, 3)
        tumor_counts = torch.zeros(NUM_CLASSES, dtype=torch.long,
                                   device=mask_tensor.device)
        for c in range(1, NUM_CLASSES):
            tumor_counts[c] = (m == c).sum()

        total_tumor = tumor_counts[1:].sum()
        if total_tumor > 0:
            # Most frequent tumor class
            labels[i] = torch.argmax(tumor_counts[1:]).item() + 1
        else:
            labels[i] = 0  # no tumor

    return labels


# #############################################################################
# SANITY CHECK
# #############################################################################
if __name__ == "__main__":
    print("=" * 60)
    print("  model_multitask.py - DualHeadUNet Sanity Check")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = get_multitask_model(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"\nTotal params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {frozen:,}")

    print("\nForward pass...")
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(device)
    seg_logits, cls_logits = model(dummy)
    print(f"Seg logits: {seg_logits.shape}  (expected: (2, 4, 256, 256))")
    print(f"Cls logits: {cls_logits.shape}  (expected: (2, 4))")
    assert seg_logits.shape == (2, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
    assert cls_logits.shape == (2, NUM_CLASSES)

    print("\nPredict mask...")
    preds = predict_mask(model, dummy, device)
    print(f"Preds: {preds.shape}, unique: {preds.unique().tolist()}")
    assert preds.shape == (2, IMG_SIZE, IMG_SIZE)

    print("\nDerive cls labels...")
    fake_mask = torch.zeros(2, IMG_SIZE, IMG_SIZE, dtype=torch.long).to(device)
    fake_mask[0, 50:100, 50:100] = 1  # glioma region
    fake_mask[1, 30:80, 30:80] = 3    # pituitary region
    labels = derive_cls_label(fake_mask)
    print(f"Labels: {labels.tolist()}  (expected: [1, 3])")
    assert labels.tolist() == [1, 3]

    print("\nUnfreeze test...")
    model.unfreeze_encoder()
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable after unfreeze: {trainable_after:,}")
    assert trainable_after == total

    print("\n" + "=" * 60)
    print("  All checks passed!")
    print("=" * 60)
