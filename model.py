"""
model.py - UNet + EfficientNet-B4 Segmentation Model
Brain Tumor Segmentation - BRISC 2025 Dataset

Architecture:
    - Encoder: EfficientNet-B4 (ImageNet pretrained)
    - Decoder: UNet
    - Input:   (B, 3, 256, 256) float32 - [gray, clahe, sobel_edges]
    - Output:  (B, 4, 256, 256) raw logits (NO softmax)

Classes:
    0 = No Tumor (background)
    1 = Glioma
    2 = Meningioma
    3 = Pituitary Tumor

Usage:
    from model import get_model, predict_mask, freeze_encoder, unfreeze_encoder
    from model import get_class_masks

    model = get_model(device)           # frozen encoder by default
    unfreeze_encoder(model)             # call after epoch 5
    preds = predict_mask(model, imgs, device)   # (B, H, W) int64
    masks = get_class_masks(preds)              # 4 binary masks
"""

import torch
import segmentation_models_pytorch as smp


# #############################################################################
# CONFIGURATION
# #############################################################################
NUM_CLASSES    = 4
ENCODER_NAME   = "efficientnet-b4"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS    = 3       # matches 3-channel preprocessing from dataset.py
IMG_SIZE       = 256

CLASS_NAMES = {0: "background", 1: "glioma", 2: "meningioma", 3: "pituitary"}


# #############################################################################
# MODEL FACTORY
# #############################################################################
def get_model(device):
    """
    Instantiate UNet with EfficientNet-B4 encoder and move to device.
    Encoder is frozen by default (for the first 5 epochs).

    Args:
        device: torch.device

    Returns:
        model on the specified device with encoder frozen
    """
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        activation=None,   # raw logits - NO softmax
    )

    model = model.to(device)

    # Freeze encoder by default (unfreeze after epoch 5)
    freeze_encoder(model)

    return model


# #############################################################################
# FREEZE / UNFREEZE ENCODER
# #############################################################################
def freeze_encoder(model):
    """
    Freeze all encoder parameters so only the decoder trains.
    Call this before training starts (epochs 1-5).
    """
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model):
    """
    Unfreeze all encoder parameters for end-to-end fine-tuning.
    Call this after epoch 5.
    """
    for param in model.encoder.parameters():
        param.requires_grad = True


# #############################################################################
# INFERENCE
# #############################################################################
def predict_mask(model, image_tensor, device):
    """
    Run inference and return predicted class labels.

    IMPORTANT: Softmax is applied ONLY here at inference time,
    NEVER during training. The loss function receives raw logits.

    Args:
        model: trained UNet model
        image_tensor: (B, 3, H, W) float32 tensor
        device: torch.device

    Returns:
        preds: (B, H, W) int64 tensor, values in {0, 1, 2, 3}
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)            # (B, 4, H, W)
        probs  = torch.softmax(logits, dim=1)   # (B, 4, H, W)
        preds  = torch.argmax(probs, dim=1)     # (B, H, W)
    return preds  # int64, values in {0, 1, 2, 3}


# #############################################################################
# PER-CLASS BINARY MASKS
# #############################################################################
def get_class_masks(pred_mask):
    """
    Extract per-class binary masks from an argmax label map.
    Used during per-class Dice evaluation.

    Args:
        pred_mask: (B, H, W) int64 tensor from predict_mask()

    Returns:
        tuple of 4 binary masks (each B, H, W, float32):
            (no_tumor_mask, glioma_mask, meningioma_mask, pituitary_mask)
    """
    no_tumor_mask   = (pred_mask == 0).float()
    glioma_mask     = (pred_mask == 1).float()
    meningioma_mask = (pred_mask == 2).float()
    pituitary_mask  = (pred_mask == 3).float()

    return no_tumor_mask, glioma_mask, meningioma_mask, pituitary_mask


# #############################################################################
# SANITY CHECK
# #############################################################################
if __name__ == "__main__":
    print("=" * 60)
    print("  model.py - UNet + EfficientNet-B4 Sanity Check")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Build model (encoder frozen by default)
    model = get_model(device)

    # Parameter counts
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"\nTotal params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {frozen:,}")

    # Forward pass with dummy input
    print("\nRunning forward pass...")
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(device)
    logits = model(dummy)
    assert logits.shape == (2, NUM_CLASSES, IMG_SIZE, IMG_SIZE), \
        f"Logit shape mismatch: expected (2,4,256,256), got {logits.shape}"
    print(f"Logits shape: {logits.shape}  (correct)")
    print(f"Logits dtype: {logits.dtype}")
    print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

    # Inference (softmax + argmax)
    print("\nRunning inference (softmax -> argmax)...")
    preds = predict_mask(model, dummy, device)
    assert preds.shape == (2, IMG_SIZE, IMG_SIZE), \
        f"Pred shape mismatch: expected (2,256,256), got {preds.shape}"
    unique_vals = preds.unique().cpu().numpy().tolist()
    assert set(unique_vals).issubset({0, 1, 2, 3}), \
        f"Unexpected predicted classes: {unique_vals}"
    print(f"Preds shape: {preds.shape}  (correct)")
    print(f"Preds dtype: {preds.dtype}")
    print(f"Unique predicted classes: {unique_vals}")

    # Per-class binary masks
    print("\nExtracting per-class masks...")
    masks = get_class_masks(preds)
    mask_names = ["no_tumor", "glioma", "meningioma", "pituitary"]
    for name, m in zip(mask_names, masks):
        assert m.shape == (2, IMG_SIZE, IMG_SIZE), \
            f"{name}_mask shape mismatch: {m.shape}"
        assert m.dtype == torch.float32, \
            f"{name}_mask dtype mismatch: {m.dtype}"
        print(f"  {name}_mask: shape={m.shape}, dtype={m.dtype}, "
              f"sum={m.sum().item():.0f}")

    # Test unfreeze
    print("\nTesting unfreeze_encoder()...")
    unfreeze_encoder(model)
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable after unfreeze: {trainable_after:,}")
    assert trainable_after == total, "Not all params are trainable after unfreeze!"
    print("All parameters now trainable  (correct)")

    print("\n" + "=" * 60)
    print("  model.py -- All checks passed!")
    print("=" * 60)
