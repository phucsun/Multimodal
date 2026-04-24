import re
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

PATTERN_VIDEO_CHAR_FRAME = re.compile(r"(?P<video>[^_]+)_(?P<char>[^_]+)_(?P<frame>\d+)$")
PATTERN_LAST_UNDERSCORE  = re.compile(r"(?P<video_key>.+)_(?P<frame>\d+)$")

_FALLBACK_TRANSFORM = transforms.ToTensor()   # Module-level singleton — avoids re-creating per call


class CustomDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        metadata_source,
        num_frames: int = 16,
        transform: Optional[transforms.Compose] = None,
        mode: str = "train",
        video_zfill: int = 0,
        char_zfill: int = 2,
        cache_index_path: Optional[str] = None,
        label_map: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.num_frames = int(num_frames)
        self.transform = transform
        assert mode in ("train", "val", "test"), f"Unknown mode: {mode}"
        self.mode = mode
        self.video_zfill = int(video_zfill)
        self.char_zfill  = int(char_zfill)
        self.cache_index_path = Path(cache_index_path) if cache_index_path is not None else None

        if isinstance(metadata_source, (str, Path)):
            with open(metadata_source, "r", encoding="utf-8") as f:
                self.metadata_list = json.load(f)
        else:
            self.metadata_list = list(metadata_source)

        if label_map is not None:
            self.label_map = label_map
        else:
            moods = sorted({item["Internal Mood"] for item in self.metadata_list})
            self.label_map = {label: idx for idx, label in enumerate(moods)}

        if self.cache_index_path and self.cache_index_path.exists():
            self.image_pool = self._load_index_cache(self.cache_index_path)
        else:
            self.image_pool = self._index_images()
            if self.cache_index_path:
                self._save_index_cache(self.cache_index_path, self.image_pool)

        self.samples: List[dict] = []
        for item in self.metadata_list:
            self._process_item(item)

    # ── Indexing ───────────────────────────────────────────────────────────────

    def _process_item(self, item: dict) -> None:
        vid_raw  = str(item.get("Video_ID"))
        char_raw = str(item.get("Character_ID", ""))
        vid_key  = self._normalize_id(vid_raw, self.video_zfill)
        char_key = self._normalize_id(char_raw, self.char_zfill)
        key = f"{vid_key}_{char_key}"

        try:
            onset  = int(item["Onset_Index"])
            offset = int(item["Offset_Index"])
        except (KeyError, ValueError, TypeError):
            return

        label_str = item.get("Internal Mood")
        if key not in self.image_pool:
            # Fallback: try raw IDs without zero-padding
            alt_key = f"{vid_raw}_{char_raw.zfill(self.char_zfill) if self.char_zfill else char_raw}"
            if alt_key in self.image_pool:
                key = alt_key
            else:
                return

        frames_map = self.image_pool[key]
        segment_paths = [frames_map[i] for i in range(onset, offset + 1) if i in frames_map]
        if not segment_paths:
            return

        label_idx = self.label_map.get(label_str)
        if label_idx is None:
            return

        self.samples.append({"paths": segment_paths, "label": int(label_idx), "video_id": key})

    @staticmethod
    def _normalize_id(id_str: str, zpad: int) -> str:
        s = str(id_str).strip()
        if s.isdigit() and zpad > 0:
            return s.zfill(zpad)
        return s

    def _index_images(self) -> Dict[str, Dict[int, str]]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir}")

        image_pool: Dict[str, Dict[int, str]] = {}
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        for filepath in sorted(self.root_dir.iterdir()):
            if not filepath.is_file() or filepath.suffix.lower() not in valid_exts:
                continue
            name = filepath.stem

            m = PATTERN_VIDEO_CHAR_FRAME.search(name)
            if m:
                vid_key   = self._normalize_id(m.group("video"), self.video_zfill)
                char_key  = self._normalize_id(m.group("char"),  self.char_zfill)
                frame_idx = int(m.group("frame"))
                video_key = f"{vid_key}_{char_key}"
            else:
                m2 = PATTERN_LAST_UNDERSCORE.search(name)
                if not m2:
                    continue
                video_key_raw = m2.group("video_key")
                frame_idx     = int(m2.group("frame"))
                parts = video_key_raw.split("_")
                if len(parts) >= 2 and parts[-1].isdigit():
                    vid_key   = self._normalize_id("_".join(parts[:-1]), self.video_zfill)
                    char_key  = self._normalize_id(parts[-1], self.char_zfill)
                    video_key = f"{vid_key}_{char_key}"
                else:
                    video_key = video_key_raw

            image_pool.setdefault(video_key, {})[frame_idx] = str(filepath)

        return image_pool

    def _save_index_cache(self, cache_path: Path, image_pool: Dict[str, Dict[int, str]]) -> None:
        try:
            serializable = {k: {str(fi): p for fi, p in v.items()} for k, v in image_pool.items()}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f)
        except Exception as e:
            print(f"Warning: failed to save index cache: {e}")

    def _load_index_cache(self, cache_path: Path) -> Dict[str, Dict[int, str]]:
        with open(cache_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        return {k: {int(fi): p for fi, p in v.items()} for k, v in loaded.items()}

    # ── Sampling ───────────────────────────────────────────────────────────────

    def _sample_indices(self, num_available: int) -> np.ndarray:
        n = self.num_frames
        if num_available == 0:
            return np.zeros(n, dtype=int)

        if num_available >= n:
            segments = np.array_split(np.arange(num_available), n)
            if self.mode == "train":
                indices = [int(np.random.choice(seg)) if len(seg) else 0 for seg in segments]
            else:
                indices = [int(seg[len(seg) // 2]) if len(seg) else 0 for seg in segments]
            return np.array(indices, dtype=int)

        # Fewer frames than needed — linear spread + pad last frame
        idxs = np.linspace(0, num_available - 1, num=num_available, dtype=int)
        pad  = np.full(n - num_available, idxs[-1], dtype=int)
        return np.concatenate([idxs, pad])

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample      = self.samples[idx]
        frame_paths: List[str] = sample["paths"]
        label       = int(sample["label"])
        n           = len(frame_paths)

        images = []
        for i in self._sample_indices(n):
            i = int(np.clip(i, 0, n - 1))
            try:
                img = Image.open(frame_paths[i]).convert("RGB")
                img = self.transform(img) if self.transform is not None else _FALLBACK_TRANSFORM(img)
            except Exception:
                img = torch.zeros(3, 224, 224)
            images.append(img)

        return torch.stack(images), torch.tensor(label, dtype=torch.long)
