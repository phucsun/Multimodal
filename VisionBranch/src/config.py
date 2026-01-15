import os
import torch
from pathlib import Path

SEED = 42


DATA_ROOT = "/Users/phuc/Phúc/HMI/Multimodal/CheoFamo/Frames"
METADATA_PATH = "/Users/phuc/Phúc/HMI/Multimodal/CheoFamo/Metadata.json"

VISION_BRANCH_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = VISION_BRANCH_DIR / "outputs"

OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4
NUM_FRAMES = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
EMBED_DIM = 512
PRETRAINED_BACKBONE = True
FREEZE_BACKBONE = True
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")