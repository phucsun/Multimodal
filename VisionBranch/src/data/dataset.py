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
PATTERN_LAST_UNDERSCORE = re.compile(r"(?P<video_key>.+)_(?P<frame>\d+)$")

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
        assert mode in ("train", "val", "test")
        self.mode = mode
        self.video_zfill = int(video_zfill)
        self.char_zfill = int(char_zfill)
        self.cache_index_path = Path(cache_index_path) if cache_index_path is not None else None

        # load metadata
        if isinstance(metadata_source, (str, Path)):
            with open(metadata_source, "r", encoding="utf-8") as f:
                self.metadata_list = json.load(f)
        else:
            self.metadata_list = metadata_source

        # label map
        if label_map is not None:
            self.label_map = label_map
        else:
            moods = sorted({item["Internal Mood"] for item in self.metadata_list})
            self.label_map = {label: idx for idx, label in enumerate(moods)}

        # index images
        if self.cache_index_path and self.cache_index_path.exists():
            self.image_pool = self._load_index_cache(self.cache_index_path)
        else:
            self.image_pool = self._index_images()
            if self.cache_index_path:
                self._save_index_cache(self.cache_index_path, self.image_pool)

        # build samples list
        self.samples = []
        for item in self.metadata_list:
            self._process_item(item)

    def _process_item(self, item):
        vid_raw = str(item.get("Video_ID"))
        char_raw = str(item.get("Character_ID", ""))
        vid_key = self._normalize_id(vid_raw, self.video_zfill)
        char_key = self._normalize_id(char_raw, self.char_zfill)
        key = f"{vid_key}_{char_key}"

        try:
            onset = int(item.get("Onset_Index"))
            offset = int(item.get("Offset_Index"))
        except Exception:
            return

        label_str = item.get("Internal Mood")
        if key not in self.image_pool:
            # Fallback logic
            alt_vid = vid_raw
            alt_char = char_raw.zfill(self.char_zfill) if self.char_zfill else char_raw
            alt_key = f"{alt_vid}_{alt_char}"
            if alt_key in self.image_pool:
                key = alt_key
            else:
                return

        frames_map = self.image_pool[key]
        segment_paths = []
        for f_idx in range(onset, offset + 1):
            if f_idx in frames_map:
                segment_paths.append(frames_map[f_idx])

        if len(segment_paths) == 0:
            return

        label_idx = self.label_map.get(label_str, None)
        if label_idx is None:
            return

        self.samples.append({"paths": segment_paths, "label": int(label_idx), "video_id": key})

    def _normalize_id(self, id_str: str, zpad: int) -> str:
        s = str(id_str).strip()
        if s.isdigit() and zpad and int(zpad) > 0:
            return s.zfill(zpad)
        return s

    def _index_images(self) -> Dict[str, Dict[int, str]]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir}")
        image_pool: Dict[str, Dict[int, str]] = {}
        all_files = sorted([p for p in self.root_dir.iterdir() if p.is_file()])
        
        for filepath in all_files:
            if filepath.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            name_no_ext = filepath.stem
            
            # Logic parsing filename regex
            m = PATTERN_VIDEO_CHAR_FRAME.search(name_no_ext)
            if m:
                vid_raw = m.group("video")
                char_raw = m.group("char")
                frame_idx = int(m.group("frame"))
                vid_key = self._normalize_id(vid_raw, self.video_zfill)
                char_key = self._normalize_id(char_raw, self.char_zfill)
                video_key = f"{vid_key}_{char_key}"
            else:
                m2 = PATTERN_LAST_UNDERSCORE.search(name_no_ext)
                if not m2:
                    continue
                video_key_raw = m2.group("video_key")
                frame_idx = int(m2.group("frame"))
                parts = video_key_raw.split("_")
                if len(parts) >= 2 and parts[-1].isdigit():
                    vid_raw = "_".join(parts[:-1])
                    char_raw = parts[-1]
                    vid_key = self._normalize_id(vid_raw, self.video_zfill)
                    char_key = self._normalize_id(char_raw, self.char_zfill)
                    video_key = f"{vid_key}_{char_key}"
                else:
                    video_key = video_key_raw

            if video_key not in image_pool:
                image_pool[video_key] = {}
            image_pool[video_key][frame_idx] = str(filepath)
        return image_pool

    def _save_index_cache(self, cache_path: Path, image_pool: Dict[str, Dict[int, str]]):
        try:
            serializable = {k: {str(fi): p for fi, p in v.items()} for k, v in image_pool.items()}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f)
        except Exception as e:
            print("Warning: failed to save index cache:", e)

    def _load_index_cache(self, cache_path: Path) -> Dict[str, Dict[int, str]]:
        with open(cache_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        return {k: {int(fi): p for fi, p in v.items()} for k, v in loaded.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, num_frames_available: int) -> np.ndarray:
        n_req = self.num_frames
        if num_frames_available >= n_req:
            segments = np.array_split(np.arange(num_frames_available), n_req)
            indices = []
            for seg in segments:
                if len(seg) == 0:
                    indices.append(0)
                else:
                    if self.mode == "train":
                        indices.append(int(np.random.choice(seg)))
                    else:
                        indices.append(int(seg[len(seg) // 2]))
            return np.array(indices, dtype=int)
        else:
            if num_frames_available == 0:
                return np.zeros(n_req, dtype=int)
            idxs = np.linspace(0, num_frames_available - 1, num=num_frames_available, dtype=int)
            if num_frames_available < n_req:
                pad_count = n_req - num_frames_available
                pad = np.full(pad_count, idxs[-1], dtype=int)
                idxs = np.concatenate([idxs, pad])
            return idxs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        frame_paths: List[str] = sample["paths"]
        label = int(sample["label"])

        indices = self._sample_indices(len(frame_paths))
        images = []
        for i in indices:
            i = int(i)
            # Boundary checks
            i = max(0, min(i, len(frame_paths) - 1))
            path = frame_paths[i]
            try:
                img = Image.open(path).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                else:
                    img = transforms.ToTensor()(img)
                images.append(img)
            except Exception as e:
                images.append(torch.zeros(3, 224, 224))
        
        video_tensor = torch.stack(images)  # (T, C, H, W)
        return video_tensor, torch.tensor(label, dtype=torch.long)