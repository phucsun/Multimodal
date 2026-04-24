import torch
from torch.amp import autocast
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, dataloader, device, criterion=None):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    _use_amp = device.type == "cuda"

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=_use_amp):
                outputs = model(inputs)
                if criterion is not None:
                    running_loss += criterion(outputs, labels).item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    avg_loss = running_loss / len(dataloader.dataset) if criterion is not None and n > 0 else None
    acc = accuracy_score(all_labels, all_preds) * 100 if n > 0 else 0.0
    f1  = f1_score(all_labels, all_preds, average="weighted") * 100 if n > 0 else 0.0

    model.train()
    return {"loss": avg_loss, "acc": acc, "f1": f1, "preds": all_preds, "labels": all_labels}


def plot_confusion_matrix(labels, preds, label_names, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()
