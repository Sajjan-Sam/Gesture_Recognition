# ==============================================================================
#  day_night_training.py
#  Training script: GestFormerFusion + ConvNeXtTinyGRU
#  Dataset: Day_Night_Dataset  (rgb + thermal, day + night)
#  NOTE: Single-stream — NO fusion. Each model takes ONE image [B,3,H,W].
# ==============================================================================
#
#  DATASET STRUCTURE:
#    Day_Night_Dataset/
#      rgb/
#        day/   train/ val/ test/  → each split has 7 class folders
#        night/ train/ val/ test/
#      thermal/
#        day/   train/ val/ test/
#        night/ train/ val/ test/
#
#  MODALITIES:
#    - rgb     : Day_Night_Dataset/rgb/
#    - thermal : Day_Night_Dataset/thermal/
#
#  DAY_NIGHT CONDITIONS:
#    - day
#    - night
#
#  MODELS:
#    1. GestFormerFusion  – EfficientNet-B0 spatial encoder
#       + Transformer (CM-Diff / CFC inspired)
#       Input: single stream [B, 3, H, W]
#
#    2. ConvNeXtTinyGRU   – ConvNeXt-Tiny + 1-layer GRU
#       Input: single stream [B, 3, H, W]
#
#  TRAINING PERMUTATIONS  (train_modality × train_day_night):
#    1.  rgb   / day
#    2.  rgb   / night
#    3.  thermal / day
#    4.  thermal / night
#
#  TESTING (for every trained model):
#    All 4 combinations of (modality × day_night):
#      rgb/day   rgb/night   thermal/day   thermal/night
#
#  TOTAL RUNS: 2 models × 4 train_configs = 8 training runs
#  TEST ROWS PER RUN: 4  →  32 rows total in results CSV
#
#  CHECKPOINT FLOW PER RUN:
#    START
#      ↓
#    Checkpoint exists? → YES: resume from saved epoch
#                       → NO : start fresh
#      ↓
#    Train epoch N  →  evaluate val  →  save if best  →  patience check
#      ↓
#    Early stop OR all epochs done
#      ↓
#    Load BEST model weights  (once, outside loop)
#      ↓
#    Test on all 4 (modality × day_night) test sets
#      ↓
#    Save results to dn_result_final/
# ==============================================================================

import os
import gc
import math
import time
import json
import random
import datetime
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.models as tvm
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# ==============================================================================
#  DEVICE & REPRODUCIBILITY
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True
torch.backends.cudnn.deterministic    = False


def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_seed(42)

# ==============================================================================
#  PATHS & CONSTANTS
# ==============================================================================

DATASET_ROOT = Path("Day_Night_Dataset")   # root of the new dataset
CKPT_DIR     = Path("dn_checkpoints_final")      # checkpoint directory
RESULT_DIR   = Path("dn_result_final")     # result CSV / plot directory

CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Two modalities present in the dataset
MODALITIES  = ["rgb", "thermal"]
# Two conditions present in the dataset
CONDITIONS  = ["day", "night"]
# The three dataset splits
SPLITS      = ["train", "val", "test"]
# Seven gesture classes
CLASSES     = ["doctor", "emergency", "fire", "help",
                "robbery", "sit_down", "stand_up"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ==============================================================================
#  HYPER-PARAMETERS
# ==============================================================================

IMG_SIZE     = 224
BATCH_SIZE   = 16
NUM_EPOCHS   = 60
LR           = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 10          # early-stopping patience
NUM_WORKERS  = 4

# ==============================================================================
#  TRANSFORMS
# ==============================================================================

train_tf = T.Compose([
    T.Resize(int(IMG_SIZE * 1.14)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tf = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

thermal_tf_train = T.Compose([
    T.Resize(int(IMG_SIZE * 1.14)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(0.5),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize([0.456, 0.456, 0.456], [0.224, 0.224, 0.224]),
])

thermal_tf_eval = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize([0.456, 0.456, 0.456], [0.224, 0.224, 0.224]),
])


def get_transform(modality: str, is_train: bool) -> T.Compose:
    if modality == "thermal":
        return thermal_tf_train if is_train else thermal_tf_eval
    return train_tf if is_train else eval_tf


# ==============================================================================
#  DATASET
#  Loads ONE modality + ONE condition + ONE split.
#  Returns (image_tensor, label_idx).
#  Images are collected recursively from
#    DATASET_ROOT/<modality>/<condition>/<split>/<class>/<pair_folder>/frames
# ==============================================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class DayNightDataset(Dataset):
    """
    Single-stream dataset for one (modality, condition, split) combination.

    folder_path: DATASET_ROOT / modality / condition / split
    """

    def __init__(self, modality: str, condition: str, split: str,
                 transform: T.Compose = None):
        self.transform = transform
        self.samples   = []   # list of (img_path, class_idx)

        base = DATASET_ROOT / modality / condition / split
        if not base.exists():
            print(f"  [WARN] Dataset path not found: {base}")
            return

        for cls in CLASSES:
            cls_dir = base / cls
            if not cls_dir.exists():
                continue
            cls_idx = CLASS_TO_IDX[cls]

            # Walk recursively: class_dir / pair_folder / frames
            for root, dirs, files in os.walk(cls_dir):
                for fname in sorted(files):
                    if Path(fname).suffix.lower() in IMAGE_EXTS:
                        self.samples.append(
                            (Path(root) / fname, cls_idx)
                        )

        print(f"  [{modality}/{condition}/{split}]  {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(str(img_path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def make_loader(modality: str, condition: str, split: str,
                is_train: bool = False) -> DataLoader:
    tf  = get_transform(modality, is_train)
    ds  = DayNightDataset(modality, condition, split, transform=tf)
    if len(ds) == 0:
        return None
    return DataLoader(
        ds,
        batch_size  = BATCH_SIZE,
        shuffle     = is_train,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
        drop_last   = is_train and len(ds) > BATCH_SIZE,
    )


# ==============================================================================
#  MODEL 1: GestFormerFusion  (CM-Diff / CFC inspired)
#  Single-stream — NO fusion.
#  Architecture:
#    image [B,3,H,W]
#      → EfficientNet-B0 spatial encoder → feat [B, 1280]
#      → proj + LayerNorm → token [B, 1, td]
#      → [CLS] prepended → TemporalPE
#      → TransformerEncoder (norm_first, GELU)
#      → CLS token → MLP head → logits [B, num_classes]
# ==============================================================================

class SpatialEncoder(nn.Module):
    """EfficientNet-B0 feature extractor (no classifier head)."""
    def __init__(self):
        super().__init__()
        base          = tvm.efficientnet_b0(
            weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim  = 1280

    def forward(self, x):               # x: [B, 3, H, W]
        return self.pool(self.features(x)).flatten(1)   # [B, 1280]


class TemporalPE(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):               # x: [B, T, d]
        return self.drop(x + self.pe[:, :x.size(1)])


class GestFormerFusion(nn.Module):
    """
    Single-stream GestFormer (CM-Diff / CFC inspired).
    Input : image tensor [B, 3, H, W]
    Output: logits       [B, num_classes]
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder  = SpatialEncoder()
        bd            = self.encoder.out_dim     # 1280

        td = 256; nh = 8; nl = 4; fd = 512

        self.proj       = nn.Sequential(nn.Linear(bd, td), nn.LayerNorm(td))
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, td))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pe         = TemporalPE(td, max_len=16)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=td, nhead=nh, dim_feedforward=fd,
            dropout=0.1, activation='gelu',
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nl)

        self.head = nn.Sequential(
            nn.LayerNorm(td), nn.Dropout(0.1), nn.Linear(td, num_classes))
        nn.init.trunc_normal_(self.head[-1].weight, std=0.02)
        nn.init.zeros_(self.head[-1].bias)

        n = sum(p.numel() for p in self.parameters())
        print(f"  GestFormerFusion (single-stream): {n:,} params  |  "
              f"{num_classes} classes")

    def forward(self, x: torch.Tensor):     # x: [B, 3, H, W]
        B   = x.size(0)
        f   = self.encoder(x)               # [B, 1280]
        tok = self.proj(f).unsqueeze(1)     # [B, 1, td]

        cls    = self.cls_token.expand(B, -1, -1)          # [B, 1, td]
        tokens = self.pe(torch.cat([cls, tok], dim=1))     # [B, 2, td]
        out    = self.transformer(tokens)
        return self.head(out[:, 0])         # CLS → logits


# ==============================================================================
#  MODEL 2: ConvNeXtTinyGRU  (ConvNeXt-Tiny + 1-layer GRU)
#  Single-stream — NO fusion.
#  Architecture:
#    image [B,3,H,W]
#      → ConvNeXt-Tiny (bottom 60% frozen) → pool → feat [B, 768]
#      → unsqueeze(1) → [B, 1, 768]  (seq_len = 1)
#      → 1-layer GRU  → hidden [B, 256]
#      → LayerNorm + Dropout + Linear → logits [B, num_classes]
# ==============================================================================

class ConvNeXtTinyEncoder(nn.Module):
    """ConvNeXt-Tiny without classifier head. Bottom 60% params frozen."""
    def __init__(self):
        super().__init__()
        base          = tvm.convnext_tiny(
            weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim  = 768

        # freeze bottom 60 % of parameters by count
        params   = list(self.parameters())
        n_freeze = int(len(params) * 0.6)
        for i, p in enumerate(params):
            p.requires_grad = i >= n_freeze

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  ConvNeXtEncoder: {total:,} params  "
              f"| {trainable:,} trainable ({100*trainable/total:.0f}%)")

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)   # [B, 768]


class ConvNeXtTinyGRU(nn.Module):
    """
    Single-stream ConvNeXt-Tiny + 1-layer GRU.
    Input : image tensor [B, 3, H, W]
    Output: logits       [B, num_classes]
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder  = ConvNeXtTinyEncoder()
        feat_dim      = self.encoder.out_dim    # 768
        gru_hidden    = 256

        self.gru  = nn.GRU(feat_dim, gru_hidden,
                            num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden),
            nn.Dropout(0.3),
            nn.Linear(gru_hidden, num_classes))

        n = sum(p.numel() for p in self.parameters())
        print(f"  ConvNeXtTinyGRU (single-stream): {n:,} params  |  "
              f"{num_classes} classes")

    def forward(self, x: torch.Tensor):         # x: [B, 3, H, W]
        f      = self.encoder(x)                # [B, 768]
        _, h   = self.gru(f.unsqueeze(1))       # h: [1, B, 256]
        return self.head(h.squeeze(0))          # [B, num_classes]


# ==============================================================================
#  MODEL FACTORY
# ==============================================================================

def build_model(model_name: str) -> nn.Module:
    if model_name == "GestFormerFusion":
        return GestFormerFusion(NUM_CLASSES)
    elif model_name == "ConvNeXtTinyGRU":
        return ConvNeXtTinyGRU(NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ==============================================================================
#  EPOCH TRAIN / EVAL  (single-stream)
# ==============================================================================

def run_epoch(model, loader, optimizer, criterion,
              is_train: bool, scaler=None):
    """
    Single-stream epoch.
    Returns (avg_loss, accuracy, macro_f1, all_labels, all_preds).
    """
    model.train() if is_train else model.eval()
    total_loss, all_labels, all_preds = 0.0, [], []

    use_amp  = (scaler is not None) and torch.cuda.is_available()
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()

    with grad_ctx:
        for imgs, labs in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labs = labs.to(DEVICE, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                    device_type='cuda' if torch.cuda.is_available() else 'cpu',
                    dtype=torch.float16 if use_amp else torch.float32,
                    enabled=use_amp):
                logits = model(imgs)            # single-stream forward
                loss   = criterion(logits, labs)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            all_labels.extend(labs.cpu().tolist())
            all_preds.extend(logits.detach().argmax(1).cpu().tolist())
            total_loss += loss.item()

    avg_loss = total_loss / max(len(loader), 1)
    acc      = accuracy_score(all_labels, all_preds) * 100
    mf1      = f1_score(all_labels, all_preds,
                         average='macro', zero_division=0) * 100
    return avg_loss, acc, mf1, all_labels, all_preds


# ==============================================================================
#  CHECKPOINT HELPERS
# ==============================================================================

def save_ckpt(path, model, optimizer, epoch, best_val):
    torch.save({
        "epoch"          : epoch,
        "model_state"    : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val"       : best_val,
    }, path)


def load_ckpt(path, model, optimizer):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"] + 1, ckpt["best_val"]   # resume AFTER saved epoch


def load_best_weights(path, model):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])


# ==============================================================================
#  VISUALISATION HELPERS
# ==============================================================================

def plot_history(history: list, tag: str, out_dir: Path):
    """Plot and save training/val loss + accuracy curves."""
    df  = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(df["epoch"], df["train_acc"],  label="train")
    axes[1].plot(df["epoch"], df["val_acc"],    label="val")
    axes[1].set_title("Accuracy (%)"); axes[1].legend()
    plt.suptitle(tag, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"{tag}_curves.png", dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_confusion(labels, preds, tag: str, out_dir: Path):
    """Save confusion matrix as PNG."""
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(tag, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"{tag}_cm.png", dpi=120, bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
#  SINGLE TRAINING RUN
# ==============================================================================

def run_training(model_name: str,
                 train_modality: str,
                 train_condition: str) -> list:
    """
    Train one model on (train_modality, train_condition) data — single stream.
    After training, test on ALL 4 combinations of (modality × condition).
    Returns a list of result dicts for CSV logging.
    """
    run_tag   = f"{model_name}__train_{train_modality}_{train_condition}"
    ckpt_path = CKPT_DIR / f"{run_tag}.pth"

    print(f"\n{'='*72}")
    print(f"  RUN: {run_tag}")
    print(f"{'='*72}")

    # ── Loaders ────────────────────────────────────────────────────────────────
    print("\n  Building data loaders ...")
    train_loader = make_loader(train_modality, train_condition, "train",
                               is_train=True)
    val_loader   = make_loader(train_modality, train_condition, "val",
                               is_train=False)

    if train_loader is None:
        print(f"  [SKIP] No training data for "
              f"{train_modality}/{train_condition}")
        return []

    # ── Model, optimiser, criterion ────────────────────────────────────────────
    print(f"\n  Building model: {model_name} ...")
    set_seed(42)
    model = build_model(model_name).to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    if scaler:
        print("  AMP (FP16) active")

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01
    )

    # ── Resume or start fresh ──────────────────────────────────────────────────
    start_epoch  = 0
    best_val_acc = 0.0

    if ckpt_path.exists():
        print(f"  [RESUME] Checkpoint found: {ckpt_path}")
        start_epoch, best_val_acc = load_ckpt(ckpt_path, model, optimizer)
        for _ in range(start_epoch):
            scheduler.step()
        print(f"  [RESUME] Resuming from epoch {start_epoch}  "
              f"(best_val_acc = {best_val_acc:.2f}%)")
    else:
        print(f"  [FRESH ] No checkpoint — starting from epoch 0.")

    # ── Training loop ──────────────────────────────────────────────────────────
    history          = []
    patience_counter = 0

    if start_epoch >= NUM_EPOCHS:
        print(f"  [SKIP TRAINING] Already completed {NUM_EPOCHS} epochs.")
    else:
        print(f"\n  >> START TRAINING  "
              f"(epochs {start_epoch + 1} → {NUM_EPOCHS}  |  "
              f"patience={PATIENCE})\n")

        for epoch in range(start_epoch, NUM_EPOCHS):

            # ── Train ──────────────────────────────────────────────────────────
            t_loss, t_acc, t_f1, _, _ = run_epoch(
                model, train_loader, optimizer, criterion,
                is_train=True, scaler=scaler
            )

            # ── Validate ───────────────────────────────────────────────────────
            v_loss, v_acc, v_f1 = 0.0, 0.0, 0.0
            if val_loader is not None:
                v_loss, v_acc, v_f1, _, _ = run_epoch(
                    model, val_loader, None, criterion,
                    is_train=False
                )

            scheduler.step()

            history.append({
                "epoch"     : epoch + 1,
                "train_loss": round(t_loss, 4),
                "train_acc" : round(t_acc,  2),
                "train_f1"  : round(t_f1,   2),
                "val_loss"  : round(v_loss,  4),
                "val_acc"   : round(v_acc,   2),
                "val_f1"    : round(v_f1,    2),
            })

            print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
                  f"Loss={t_loss:.4f} | Train={t_acc:.2f}% | "
                  f"Val={v_acc:.2f}% | Best={best_val_acc:.2f}% | "
                  f"Patience={patience_counter}/{PATIENCE}")

            # ── Save best ──────────────────────────────────────────────────────
            if v_acc > best_val_acc:
                best_val_acc     = v_acc
                patience_counter = 0
                save_ckpt(ckpt_path, model, optimizer, epoch, best_val_acc)
                print(f"  ✔  Best checkpoint saved  "
                      f"(val_acc={best_val_acc:.2f}%)")
            else:
                patience_counter += 1

            # ── Early stop ─────────────────────────────────────────────────────
            if patience_counter >= PATIENCE:
                print(f"\n  ⏹  Early stopping  "
                      f"(no improvement for {PATIENCE} epochs)")
                break

        print(f"\n  << TRAINING COMPLETE  "
              f"(best_val_acc = {best_val_acc:.2f}%)")

    # ── Save training history & curves ────────────────────────────────────────
    if history:
        pd.DataFrame(history).to_csv(
            RESULT_DIR / f"{run_tag}_history.csv", index=False
        )
        plot_history(history, run_tag, RESULT_DIR)

    # ── Load BEST weights (once, outside training loop) ───────────────────────
    if ckpt_path.exists():
        print(f"\n  >> Loading BEST weights from: {ckpt_path}")
        load_best_weights(ckpt_path, model)
        print(f"  ✔  Best model loaded  (best_val_acc = {best_val_acc:.2f}%)")
    else:
        print("  [WARN] No checkpoint — using current (last-epoch) weights.")

    # ── Testing: all 4 (modality × condition) test combinations ───────────────
    print(f"\n  >> Testing on all modality × condition combinations ...")
    all_results = []

    for test_modality in MODALITIES:
        for test_condition in CONDITIONS:

            test_tag    = f"test_{test_modality}_{test_condition}"
            test_loader = make_loader(test_modality, test_condition, "test",
                                      is_train=False)

            if test_loader is None:
                print(f"  [SKIP TEST] {test_modality}/{test_condition} "
                      f"— no data")
                continue

            _, t_acc, t_mf1, labels, preds = run_epoch(
                model, test_loader, None, criterion, is_train=False
            )

            wf1 = f1_score(labels, preds,
                            average='weighted', zero_division=0) * 100

            # Save confusion matrix
            save_confusion(
                labels, preds,
                f"{run_tag}__{test_tag}",
                RESULT_DIR
            )

            # Save per-class report
            report = classification_report(
                labels, preds,
                target_names=CLASSES, zero_division=0
            )
            report_path = RESULT_DIR / f"{run_tag}__{test_tag}_report.txt"
            with open(report_path, "w") as f:
                f.write(f"Run   : {run_tag}\n")
                f.write(f"Test  : {test_modality}/{test_condition}\n\n")
                f.write(report)

            print(f"  [TEST {test_modality}/{test_condition}]  "
                  f"Acc={t_acc:.2f}%  MacroF1={t_mf1:.2f}%  "
                  f"WeightedF1={wf1:.2f}%")

            all_results.append({
                "Model"           : model_name,
                "Train_Modality"  : train_modality,
                "Train_Condition" : train_condition,
                "Test_Modality"   : test_modality,
                "Test_Condition"  : test_condition,
                "Accuracy"        : round(t_acc,  2),
                "Macro_F1"        : round(t_mf1,  2),
                "Weighted_F1"     : round(wf1,    2),
                "Best_Val_Acc"    : round(best_val_acc, 2),
            })

    # ── Free VRAM ──────────────────────────────────────────────────────────────
    try:
        model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return all_results


# ==============================================================================
#  MAIN  — ITERATE ALL TRAINING PERMUTATIONS
# ==============================================================================
#
#  Training permutations (4 total per model):
#    train_modality × train_condition
#
#  MODEL NAMES:
#    GestFormerFusion | ConvNeXtTinyGRU
#
#  TRAIN CONFIGS:
#    (rgb, day) | (rgb, night) | (thermal, day) | (thermal, night)
#
#  TEST CONFIGS (for every trained model):
#    (rgb, day) | (rgb, night) | (thermal, day) | (thermal, night)
#
#  TOTAL: 2 models × 4 train_configs = 8 runs
#         8 runs   × 4 test_configs  = 32 result rows
# ==============================================================================

if __name__ == "__main__":

    GRAND_START = time.time()

    MODEL_NAMES   = ["GestFormerFusion", "ConvNeXtTinyGRU"]
    TRAIN_CONFIGS = list(itertools.product(MODALITIES, CONDITIONS))
    # → [(rgb,day), (rgb,night), (thermal,day), (thermal,night)]

    all_rows   = []
    total_runs = len(MODEL_NAMES) * len(TRAIN_CONFIGS)
    run_idx    = 0

    for model_name in MODEL_NAMES:
        for train_modality, train_condition in TRAIN_CONFIGS:

            run_idx += 1
            print(f"\n\n{'#'*72}")
            print(f"  RUN {run_idx}/{total_runs}  |  "
                  f"Model={model_name}  "
                  f"Train={train_modality}/{train_condition}")
            print(f"{'#'*72}")

            rows = run_training(model_name, train_modality, train_condition)

            if rows:
                all_rows.extend(rows)

                # Save intermediate results after every run
                df_tmp = pd.DataFrame(all_rows)
                df_tmp.to_csv(
                    RESULT_DIR / "dn_results_intermediate.csv",
                    index=False
                )

    # ── Final CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)

    if len(df) > 0:
        df_sorted = df.sort_values("Accuracy", ascending=False)
        df_sorted.to_csv(
            RESULT_DIR / "dn_result_final.csv", index=False
        )
        df_sorted.to_csv(
            RESULT_DIR / "dn_results_ranked.csv", index=False
        )

        # ── Summary table ──────────────────────────────────────────────────────
        print(f"\n{'='*72}")
        print("  FINAL RESULTS SUMMARY")
        print(f"{'='*72}")
        print(df_sorted.to_string(index=False))

        # ── Cross-condition heatmap (accuracy + macro-F1) ──────────────────────
        for model_name in MODEL_NAMES:
            for metric in ("Accuracy", "Macro_F1"):
                sub = df[df["Model"] == model_name].copy()
                if sub.empty:
                    continue
                sub["Train"] = (sub["Train_Modality"] + "/"
                                + sub["Train_Condition"])
                sub["Test"]  = (sub["Test_Modality"]  + "/"
                                + sub["Test_Condition"])
                try:
                    pivot = sub.pivot(index="Train", columns="Test",
                                      values=metric).fillna(0)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                                vmin=0, vmax=100, ax=ax,
                                annot_kws={"size": 12, "weight": "bold"})
                    ax.set_title(f"{model_name} — {metric} (%)")
                    plt.tight_layout()
                    fig.savefig(
                        RESULT_DIR / f"{model_name}_{metric}_heatmap.png",
                        dpi=150, bbox_inches='tight'
                    )
                    plt.close(fig)
                    print(f"  Heatmap saved: "
                          f"{model_name}_{metric}_heatmap.png")
                except Exception as e:
                    print(f"  [WARN] Could not create heatmap: {e}")

    # ── Timing ─────────────────────────────────────────────────────────────────
    elapsed = str(datetime.timedelta(seconds=int(time.time() - GRAND_START)))
    print(f"\n{'='*72}")
    print(f"  ALL {total_runs} TRAINING RUNS COMPLETE")
    print(f"  Total wall time  : {elapsed}")
    print(f"  Total result rows: {len(df)}")
    print(f"  Results saved to : {RESULT_DIR.resolve()}")
    print(f"  Checkpoints in   : {CKPT_DIR.resolve()}")
    print(f"{'='*72}")