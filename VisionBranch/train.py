import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import classification_report
from pathlib import Path

from src.config import *
from src.data.split import stratified_video_split
from src.data.dataset import CustomDataset
from src.models.network import EmotionClassifier
from src.utils.evaluation import evaluate_model, plot_confusion_matrix

import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def main():
    print("Device:", DEVICE)
    
    # sanity checks
    if not Path(DATA_ROOT).exists():
        print(f"ERROR: DATA_ROOT path ({DATA_ROOT}) does not exist. Check src/config.py.")
        return
    elif not Path(METADATA_PATH).exists():
        print(f"ERROR: METADATA_PATH ({METADATA_PATH}) does not exist. Check src/config.py.")
        return

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        full_metadata = json.load(f)

    all_moods = sorted({item["Internal Mood"] for item in full_metadata})
    label_map_global = {label: idx for idx, label in enumerate(all_moods)}
    NUM_CLASSES = len(all_moods)
    print(f"Detected {NUM_CLASSES} classes: {all_moods}")

    train_meta, val_meta = stratified_video_split(full_metadata, val_ratio=0.2, seed=SEED)
    print(f"Train items: {len(train_meta)}, Val items: {len(val_meta)}")

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    train_dataset = CustomDataset(root_dir=DATA_ROOT, metadata_source=train_meta, num_frames=NUM_FRAMES, transform=data_transform, mode='train', label_map=label_map_global)
    val_dataset = CustomDataset(root_dir=DATA_ROOT, metadata_source=val_meta, num_frames=NUM_FRAMES, transform=data_transform, mode='val', label_map=label_map_global)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    train_label_indices = [s['label'] for s in train_dataset.samples]
    counts = np.bincount(train_label_indices, minlength=NUM_CLASSES)
    counts = np.where(counts == 0, 1, counts)  # avoid zero
    class_weights = counts.sum() / (NUM_CLASSES * counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print("Class weights:", class_weights.tolist())

    model = EmotionClassifier(num_classes=NUM_CLASSES, embed_dim=EMBED_DIM, num_frames=NUM_FRAMES, pretrained_backbone=PRETRAINED_BACKBONE, freeze_backbone=FREEZE_BACKBONE)
    model = model.to(DEVICE)

    optim_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.AdamW(optim_params, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scaler = GradScaler()

    best_val_f1 = -1.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for inputs, labels in pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast(enabled=(DEVICE.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(optim_params, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader.dataset)

        val_stats = evaluate_model(model, val_loader, DEVICE, criterion=criterion)
        print(f"Epoch {epoch+1} | Train loss: {avg_train_loss:.4f} | Val loss: {val_stats['loss']:.4f} | Val acc: {val_stats['acc']:.2f}% | Val F1: {val_stats['f1']:.2f}%")

        if val_stats['f1'] > best_val_f1:
            best_val_f1 = val_stats['f1']
            ckpt_path = os.path.join(OUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path} (F1 {best_val_f1:.2f}%)")
            
            label_names = [k for k, _ in sorted(label_map_global.items(), key=lambda kv: kv[1])]

            print(classification_report(val_stats['labels'], val_stats['preds'], target_names=label_names))

    print("Training finished. Best val F1:", best_val_f1)
    print("Best checkpoint saved to:", os.path.join(OUT_DIR, "best_model.pth"))

if __name__ == "__main__":
    main()