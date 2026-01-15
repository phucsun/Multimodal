import random
from collections import defaultdict
from typing import List, Dict, Tuple

def group_by_video_key(metadata: List[dict]) -> Dict[str, List[dict]]:
    groups = defaultdict(list)
    for it in metadata:
        vid = str(it.get("Video_ID"))
        char = str(it.get("Character_ID", "")).zfill(2)
        key = f"{vid}_{char}"
        groups[key].append(it)
    return groups

def stratified_video_split(metadata: List[dict], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """
    Split metadata into train and val lists, splitting by video_key and stratifying by majority label.
    """
    random.seed(seed)
    groups = group_by_video_key(metadata)
    video_keys_by_label = defaultdict(list)
    
    # Determine majority label for stratification
    for key, items in groups.items():
        labels = [it["Internal Mood"] for it in items]
        if not labels: 
            continue
        maj = max(set(labels), key=labels.count)
        video_keys_by_label[maj].append(key)

    train_keys, val_keys = [], []
    for label, keys in video_keys_by_label.items():
        random.shuffle(keys)
        n_val = max(1, int(len(keys) * val_ratio)) if len(keys) > 1 else 0
        val = keys[:n_val]
        train = keys[n_val:]
        train_keys += train
        val_keys += val

    train_meta, val_meta = [], []
    for k in train_keys:
        train_meta.extend(groups[k])
    for k in val_keys:
        val_meta.extend(groups[k])

    # Fallback if val is empty
    if len(val_meta) == 0 and len(train_keys) > 0:
        first_key = train_keys.pop(0)
        val_meta.extend(groups[first_key])
        train_meta = [it for k in train_keys for it in groups[k]]

    return train_meta, val_meta