import torch
from pathlib import Path

SEED = 42

DATA_ROOT     = "/Users/phuc/Phúc/HMI/Multimodal/CheoFamo/Frames"
METADATA_PATH = "/Users/phuc/Phúc/HMI/Multimodal/CheoFamo/Metadata.json"

VISION_BRANCH_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = VISION_BRANCH_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 4
NUM_FRAMES  = 16
NUM_WORKERS = 4

# ── Backbone ──────────────────────────────────────────────────────────────────
# Options: "resnet18" | "efficientnet_b0" | "convnext_tiny"
BACKBONE_NAME       = "efficientnet_b0"
PRETRAINED_BACKBONE = True
FREEZE_BACKBONE     = True
PARTIAL_UNFREEZE    = True    # Fine-tune last block of backbone at BACKBONE_LR_SCALE * LR
BACKBONE_LR_SCALE   = 0.1    # 10× lower LR for fine-tuned backbone layers

# ── Model ─────────────────────────────────────────────────────────────────────
EMBED_DIM      = 512
DROP_PATH_RATE = 0.1          # Stochastic depth — linearly increases per layer

# ── Training ──────────────────────────────────────────────────────────────────
NUM_EPOCHS      = 20
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-2
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS   = 3           # Linear LR warmup before cosine annealing
MIXUP_ALPHA     = 0.4         # Beta distribution alpha for Mixup (0 = disabled)

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
