import random
from collections import defaultdict
from typing import List, Dict, Tuple


def group_by_video_key(metadata: List[dict], char_zfill: int = 2) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for item in metadata:
        vid  = str(item.get("Video_ID"))
        char = str(item.get("Character_ID", "")).zfill(char_zfill) if char_zfill else str(item.get("Character_ID", ""))
        groups[f"{vid}_{char}"].append(item)
    return groups


def stratified_video_split(
    metadata: List[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
    char_zfill: int = 2,
) -> Tuple[List[dict], List[dict]]:
    """Split metadata into train/val by video, stratified on majority emotion label."""
    random.seed(seed)
    groups = group_by_video_key(metadata, char_zfill=char_zfill)

    # Bucket video keys by majority label for stratification
    keys_by_label: Dict[str, List[str]] = defaultdict(list)
    for key, items in groups.items():
        labels = [it["Internal Mood"] for it in items]
        if not labels:
            continue
        majority = max(set(labels), key=labels.count)
        keys_by_label[majority].append(key)

    train_keys, val_keys = [], []
    for keys in keys_by_label.values():
        random.shuffle(keys)
        n_val = max(1, int(len(keys) * val_ratio)) if len(keys) > 1 else 0
        val_keys   += keys[:n_val]
        train_keys += keys[n_val:]

    # Fallback: if val is empty, move one train video over
    if not val_keys and train_keys:
        val_keys.append(train_keys.pop(0))

    train_meta = [item for k in train_keys for item in groups[k]]
    val_meta   = [item for k in val_keys   for item in groups[k]]
    return train_meta, val_meta
