import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from sklearn.metrics import classification_report
import random

from src.config import (
    SEED, DEVICE,
    DATA_ROOT, METADATA_PATH, OUT_DIR,
    BATCH_SIZE, NUM_FRAMES, NUM_WORKERS,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LABEL_SMOOTHING,
    WARMUP_EPOCHS, MIXUP_ALPHA,
    EMBED_DIM, BACKBONE_NAME, PRETRAINED_BACKBONE, FREEZE_BACKBONE,
    PARTIAL_UNFREEZE, BACKBONE_LR_SCALE, DROP_PATH_RATE,
)
from src.data.split import stratified_video_split
from src.data.dataset import CustomDataset
from src.models.network import EmotionClassifier
from src.utils.evaluation import evaluate_model

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")   # TF32 on Ampere GPUs


# ── Augmentation ───────────────────────────────────────────────────────────────

def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),   # Modern > ColorJitter
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── Mixup ──────────────────────────────────────────────────────────────────────

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Returns mixed inputs, pairs of targets, and lambda."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}  |  Backbone: {BACKBONE_NAME}")

    if not Path(DATA_ROOT).exists():
        print(f"ERROR: DATA_ROOT not found: {DATA_ROOT}")
        return
    if not Path(METADATA_PATH).exists():
        print(f"ERROR: METADATA_PATH not found: {METADATA_PATH}")
        return

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        full_metadata = json.load(f)

    all_moods  = sorted({item["Internal Mood"] for item in full_metadata})
    label_map  = {label: idx for idx, label in enumerate(all_moods)}
    num_classes = len(all_moods)
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    print(f"Classes ({num_classes}): {all_moods}")

    train_meta, val_meta = stratified_video_split(full_metadata, val_ratio=0.2, seed=SEED)
    print(f"Train: {len(train_meta)} items  |  Val: {len(val_meta)} items")

    _pin        = DEVICE.type == "cuda"
    _persistent = NUM_WORKERS > 0

    train_dataset = CustomDataset(
        root_dir=DATA_ROOT, metadata_source=train_meta,
        num_frames=NUM_FRAMES, transform=build_transforms(train=True),
        mode="train", label_map=label_map,
    )
    val_dataset = CustomDataset(
        root_dir=DATA_ROOT, metadata_source=val_meta,
        num_frames=NUM_FRAMES, transform=build_transforms(train=False),
        mode="val", label_map=label_map,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=_pin, persistent_workers=_persistent,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=_pin, persistent_workers=_persistent,
    )

    # Class-weighted loss
    counts = np.bincount([s["label"] for s in train_dataset.samples], minlength=num_classes)
    counts = np.where(counts == 0, 1, counts)
    class_weights = torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float).to(DEVICE)
    print(f"Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EmotionClassifier(
        num_classes=num_classes,
        embed_dim=EMBED_DIM,
        num_frames=NUM_FRAMES,
        pretrained_backbone=PRETRAINED_BACKBONE,
        freeze_backbone=FREEZE_BACKBONE,
        partial_unfreeze=PARTIAL_UNFREEZE,
        backbone_name=BACKBONE_NAME,
        drop_path_rate=DROP_PATH_RATE,
    ).to(DEVICE)

    # ── Parameter groups: backbone gets 10× lower LR ──────────────────────────
    backbone_trainable = [
        p for p in model.backbone.spatial_cnn.backbone.parameters() if p.requires_grad
    ]
    other_trainable = [
        p for p in model.parameters()
        if p.requires_grad and not any(p is bp for bp in backbone_trainable)
    ]

    param_groups = [{"params": other_trainable, "lr": LEARNING_RATE}]
    if backbone_trainable:
        param_groups.insert(0, {"params": backbone_trainable, "lr": LEARNING_RATE * BACKBONE_LR_SCALE})
        print(f"Partial unfreeze: {sum(p.numel() for p in backbone_trainable)/1e6:.2f}M backbone params at LR×{BACKBONE_LR_SCALE}")

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_trainable/1e6:.2f}M")

    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    # Warmup 3 epochs → cosine annealing for the rest
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, NUM_EPOCHS - WARMUP_EPOCHS), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    scaler    = GradScaler(device=DEVICE.type)

    # torch.compile: ~20-30% speedup on GPU (PyTorch 2.0+)
    if DEVICE.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1 = -1.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS} [Train]")

        for inputs, labels in pbar:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # Mixup augmentation
            if MIXUP_ALPHA > 0:
                inputs, y_a, y_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
            else:
                y_a, y_b, lam = labels, labels, 1.0

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type, enabled=_pin):
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], max_norm=5.0
            )
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[-1]['lr']:.2e}")

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader.dataset)
        val_stats = evaluate_model(model, val_loader, DEVICE, criterion=criterion)
        current_lr = scheduler.get_last_lr()[-1]

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {val_stats['loss']:.4f} | "
            f"Val acc: {val_stats['acc']:.2f}% | "
            f"Val F1: {val_stats['f1']:.2f}% | "
            f"LR: {current_lr:.2e}"
        )

        if val_stats["f1"] > best_val_f1:
            best_val_f1 = val_stats["f1"]
            ckpt_path = OUT_DIR / "best_model.pth"
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_f1":               best_val_f1,
                "label_map":            label_map,
                "backbone_name":        BACKBONE_NAME,
                "embed_dim":            EMBED_DIM,
            }, ckpt_path)
            print(f"  ✓ Best model saved (F1 {best_val_f1:.2f}%) → {ckpt_path}")
            print(classification_report(val_stats["labels"], val_stats["preds"], target_names=label_names))

    print(f"\nTraining complete. Best val F1: {best_val_f1:.2f}%")


if __name__ == "__main__":
    main()
