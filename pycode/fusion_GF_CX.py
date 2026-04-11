
CPU_DEBUG_MODE = False   # True=local CPU debug | False=RTX A4000 GPU

import sys, os, gc, time, random, math, datetime, warnings, json
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
warnings.filterwarnings('ignore')

# ================================================================
#  DEVICE
# ================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.deterministic    = False
torch.backends.cudnn.benchmark        = True

# ================================================================
#  PATHS
# ================================================================

PROJECT_ROOT     = Path(".")
FUSION_MANIFEST  = PROJECT_ROOT / "manifests" / "fusion_paired_manifest.csv"
FUSION_SPLIT_DIR = PROJECT_ROOT / "manifests" / "fusion_splits"

# ================================================================
#  LOAD FUSION MANIFEST
# ================================================================

assert FUSION_MANIFEST.exists(), \
    f"Run fusion_splitB.py first to create: {FUSION_MANIFEST}"

manifest = pd.read_csv(FUSION_MANIFEST)
for col in ['pair_id', 'subject_id', 'base_subject_id', 'distance']:
    manifest[col] = manifest[col].astype(str)

GESTURES       = sorted(manifest['gesture'].unique().tolist())
GESTURE_TO_IDX = {g: i for i, g in enumerate(GESTURES)}
NUM_CLASSES    = len(GESTURES)
DISTANCES      = sorted(manifest['distance'].unique().tolist())

print(f"Fusion manifest: {len(manifest)} pairs | {NUM_CLASSES} gestures | {len(DISTANCES)} distances")
print(f"Gestures:  {GESTURES}")
print(f"Distances: {DISTANCES}")

# ================================================================
#  FUSION TYPES
# ================================================================

FUSION_TYPES = {
    "rgb_thermal": {
        "primary_col":   "rgb_frame_dir",
        "secondary_col": "thermal_frame_dir",
        "filter_col":    None,
    },
    "rgb_bgrem_thermal": {
        "primary_col":   "rgb_bgrem_frame_dir",
        "secondary_col": "thermal_frame_dir",
        "filter_col":    "bgrem_frames_exist",
    },
}

# ================================================================
#  21 PROTOCOL DEFINITIONS
# ================================================================

TRAIN_COMBINATIONS = [
    ["4_feet"],
    ["6_feet"],
    ["8_feet"],
    ["4_feet", "6_feet"],
    ["4_feet", "8_feet"],
    ["6_feet", "8_feet"],
    ["4_feet", "6_feet", "8_feet"],
]
TEST_DISTANCES = ["4_feet", "6_feet", "8_feet"]

# ================================================================
#  TEMPORAL SAMPLING
# ================================================================

def sample_uniform(n, w):
    if n <= 0: return [0] * w
    if n <= w:
        idx = list(range(n))
        while len(idx) < w: idx += idx
        return sorted(idx[:w])
    return [int(i * (n / w)) for i in range(w)]

def sample_random(n, w):
    if n <= w: return sample_uniform(n, w)
    seg = n / w
    return [random.randint(int(i*seg), max(int(i*seg), int((i+1)*seg)-1)) for i in range(w)]

# ================================================================
#  FUSION DATASET
#
#  Returns a 6-channel tensor: [6, T, H, W]
#    channels 0..2 = primary modality (RGB or RGB_BGREM)
#    channels 3..5 = thermal (grayscale replicated to 3ch)
#
#  Both GestFormerFusion and ConvNeXtTinyGRU receive the same
#  6-channel tensor — they split internally at channels [:3] / [3:].
#  This means a single dataloader serves both models with no changes.
# ================================================================

class FusionGestureClipDataset(Dataset):
    EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

    def __init__(self, df, primary_col, secondary_col,
                 num_frames, img_size, is_train):
        self.df            = df.reset_index(drop=True)
        self.primary_col   = primary_col
        self.secondary_col = secondary_col
        self.num_frames    = num_frames
        self.img_size      = img_size
        self.is_train      = is_train

        if is_train:
            self.spatial_aug = T.RandomResizedCrop(img_size, scale=(0.8, 1.0))
            self.hflip_p     = 0.5
        else:
            self.resize = T.Resize(int(img_size * 1.14))
            self.crop   = T.CenterCrop(img_size)

        self.to_tensor = T.ToTensor()
        self.norm_rgb  = T.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
        self.norm_thm  = T.Normalize([0.456, 0.456, 0.456],
                                     [0.224, 0.224, 0.224])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = GESTURE_TO_IDX[row['gesture']]
        p_dir = str(row[self.primary_col])
        s_dir = str(row[self.secondary_col])

        p_files  = self._sorted_frames(p_dir)
        s_files  = self._sorted_frames(s_dir)
        n_frames = max(len(p_files), len(s_files), 1)

        sampler = sample_random if self.is_train else sample_uniform
        indices = sampler(n_frames, self.num_frames)

        # Same spatial seed ensures aligned crop/flip for both streams
        seed   = random.randint(0, 2**31) if self.is_train else 0
        p_clip = self._load_clip(p_files, indices, seed, is_thermal=False)
        s_clip = self._load_clip(s_files, indices, seed, is_thermal=True)

        fused = torch.cat([p_clip, s_clip], dim=0)   # [6, T, H, W]
        return fused, torch.tensor(label, dtype=torch.long)

    def _sorted_frames(self, frame_dir):
        p = Path(frame_dir)
        if not p.exists(): return []
        return sorted([f for f in p.iterdir() if f.suffix.lower() in self.EXTS])

    def _load_clip(self, files, indices, seed, is_thermal):
        zero = torch.zeros(3, self.num_frames, self.img_size, self.img_size)
        if not files: return zero
        frames = []
        for i in indices:
            fi = min(i, len(files) - 1)
            try:
                img = Image.open(str(files[fi]))
                img = img.convert('L') if is_thermal else img.convert('RGB')
                if is_thermal:
                    img = img.convert('RGB')
                t = self._apply_spatial(img, seed, is_thermal)
            except Exception:
                t = torch.zeros(3, self.img_size, self.img_size)
            frames.append(t)
        return torch.stack(frames, dim=0).permute(1, 0, 2, 3)  # [3, T, H, W]

    def _apply_spatial(self, img, seed, is_thermal):
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        if self.is_train:
            img = self.spatial_aug(img)
            if random.random() < self.hflip_p:
                img = T.functional.hflip(img)
            if not is_thermal:
                img = T.ColorJitter(0.3, 0.3, 0.2)(img)
        else:
            img = self.resize(img)
            img = self.crop(img)
        torch.set_rng_state(state)
        t = self.to_tensor(img)
        return self.norm_thm(t) if is_thermal else self.norm_rgb(t)


def make_fusion_loaders(tr, va, te, primary_col, secondary_col,
                        nf, sz, bs, nw):
    kw = dict(num_workers=nw, pin_memory=False)
    return (
        DataLoader(FusionGestureClipDataset(tr, primary_col, secondary_col,
                                            nf, sz, True),
                   bs, shuffle=True, drop_last=len(tr) > bs, **kw),
        DataLoader(FusionGestureClipDataset(va, primary_col, secondary_col,
                                            nf, sz, False),
                   bs, shuffle=False, **kw),
        DataLoader(FusionGestureClipDataset(te, primary_col, secondary_col,
                                            nf, sz, False),
                   bs, shuffle=False, **kw),
    )

# ================================================================
#  SEED
# ================================================================

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


class SpatialEncoder(nn.Module):
    """EfficientNet-B0 feature extractor (or MobileNetV3-small in debug)."""
    def __init__(self):
        super().__init__()
        if CPU_DEBUG_MODE:
            base = tvm.mobilenet_v3_small(
                weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.out_dim = 576
        else:
            base = tvm.efficientnet_b0(
                weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.out_dim = 1280
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)   # [N, out_dim]


class TemporalPE(nn.Module):
    """Sinusoidal positional encoding for temporal dimension."""
    def __init__(self, d, max_len=72, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention (CM-Diff CFC module, arXiv:2503.09514 §3.1).

    RGB tokens attend to Thermal (gain thermal context).
    Thermal tokens attend to RGB (gain RGB context).
    Each stream goes through FFN + residual + LN → richer representations.
    Both enhanced streams are returned; the caller fuses them.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.rgb2thm  = nn.MultiheadAttention(d_model, nhead,
                                               dropout=dropout, batch_first=True)
        self.thm2rgb  = nn.MultiheadAttention(d_model, nhead,
                                               dropout=dropout, batch_first=True)
        self.norm_rgb = nn.LayerNorm(d_model)
        self.norm_thm = nn.LayerNorm(d_model)
        self.ffn_rgb  = nn.Sequential(
            nn.Linear(d_model, d_model*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model), nn.Dropout(dropout))
        self.ffn_thm  = nn.Sequential(
            nn.Linear(d_model, d_model*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model), nn.Dropout(dropout))
        self.norm_ffn_rgb = nn.LayerNorm(d_model)
        self.norm_ffn_thm = nn.LayerNorm(d_model)

    def forward(self, rgb_tokens, thm_tokens):
        rgb_ca, _ = self.rgb2thm(rgb_tokens, thm_tokens, thm_tokens)
        thm_ca, _ = self.thm2rgb(thm_tokens, rgb_tokens, rgb_tokens)
        rgb_ca    = self.norm_rgb(rgb_tokens + rgb_ca)
        thm_ca    = self.norm_thm(thm_tokens + thm_ca)
        rgb_out   = self.norm_ffn_rgb(rgb_ca + self.ffn_rgb(rgb_ca))
        thm_out   = self.norm_ffn_thm(thm_ca + self.ffn_thm(thm_ca))
        return rgb_out, thm_out


class GestFormerFusion(nn.Module):
    """
    Dual-stream GestFormer with Cross-Modal Attention Fusion.
    Input:  x [B, 6, T, H, W]   ch 0-2=primary, ch 3-5=thermal
    Output: logits [B, nc]
    """
    def __init__(self, nc):
        super().__init__()
        self.rgb_encoder = SpatialEncoder()
        self.thm_encoder = SpatialEncoder()
        bd = self.rgb_encoder.out_dim  # 1280

        td = 128 if CPU_DEBUG_MODE else 256
        nh = 4   if CPU_DEBUG_MODE else 8
        nl = 2   if CPU_DEBUG_MODE else 4
        fd = 256 if CPU_DEBUG_MODE else 512
        Tf = 8   if CPU_DEBUG_MODE else 16

        self.rgb_proj        = nn.Sequential(nn.Linear(bd, td), nn.LayerNorm(td))
        self.thm_proj        = nn.Sequential(nn.Linear(bd, td), nn.LayerNorm(td))
        self.cross_modal_attn = CrossModalAttention(td, nhead=nh)
        self.fusion_norm     = nn.LayerNorm(td)
        self.cls             = nn.Parameter(torch.zeros(1, 1, td))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.pe = TemporalPE(td, max_len=Tf + 8)
        enc = nn.TransformerEncoderLayer(
            d_model=td, nhead=nh, dim_feedforward=fd,
            dropout=0.1, activation='gelu',
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=nl)
        self.head = nn.Sequential(nn.LayerNorm(td), nn.Dropout(0.1),
                                  nn.Linear(td, nc))
        nn.init.trunc_normal_(self.head[-1].weight, std=0.02)
        nn.init.zeros_(self.head[-1].bias)
        n = sum(p.numel() for p in self.parameters())
        print(f'  GestFormerFusion: {nl}Lx{nh}Hx{td}D  {n:,} params  {nc} classes')

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert C == 6, f"Expected 6-channel input, got {C}"
        rgb = x[:, :3]; thm = x[:, 3:]

        rgb_flat  = rgb.permute(0,2,1,3,4).reshape(B*T, 3, H, W)
        thm_flat  = thm.permute(0,2,1,3,4).reshape(B*T, 3, H, W)
        rgb_feats = self.rgb_encoder(rgb_flat).reshape(B, T, -1)
        thm_feats = self.thm_encoder(thm_flat).reshape(B, T, -1)

        rgb_tokens = self.rgb_proj(rgb_feats)
        thm_tokens = self.thm_proj(thm_feats)

        rgb_enh, thm_enh = self.cross_modal_attn(rgb_tokens, thm_tokens)
        fused = self.fusion_norm(rgb_enh + thm_enh)

        fused = self.pe(torch.cat([self.cls.expand(B, -1, -1), fused], dim=1))
        out   = self.transformer(fused)
        return self.head(out[:, 0])


def build_gestformer_fusion(nc):
    return GestFormerFusion(nc)




class ConvNeXtTinyEncoder(nn.Module):
    """
    ConvNeXt-Tiny feature extractor.
    Removes the classification head; returns pooled spatial features.
    Top 40% of parameters are trainable (freeze_60 strategy from reference).
    """
    def __init__(self):
        super().__init__()
        base = tvm.convnext_tiny(
            weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # Remove classifier head — keep feature extractor only
        self.features  = base.features
        self.pool      = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim   = 768   # ConvNeXt-Tiny final stage channels

        # freeze_60: freeze bottom 60% of parameters by count
        params = list(self.parameters())
        n_freeze = int(len(params) * 0.6)
        for i, p in enumerate(params):
            p.requires_grad = (i >= n_freeze)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f'  ConvNeXtTinyEncoder: {total:,} total params, '
              f'{trainable:,} trainable ({100*trainable/total:.0f}%)')

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)   # [N, 768]


class ConvNeXtTinyGRU(nn.Module):
    """
    Dual-stream ConvNeXt-Tiny + 1-layer GRU fusion model.

    Input:  x [B, 6, T, H, W]   ch 0-2=primary, ch 3-5=thermal
    Output: logits [B, nc]

    Fusion strategy: feature-level concatenation (2×768=1536-d)
    fed into a 1-layer GRU.  Last hidden state → linear classifier.
    """
    def __init__(self, nc):
        super().__init__()
        self.rgb_encoder = ConvNeXtTinyEncoder()
        self.thm_encoder = ConvNeXtTinyEncoder()

        feat_dim   = self.rgb_encoder.out_dim   # 768
        fused_dim  = feat_dim * 2               # 1536 (concat of both streams)
        gru_hidden = 256

        # 1-layer GRU (matches best config from reference ablation)
        self.gru = nn.GRU(fused_dim, gru_hidden,
                          num_layers=1, batch_first=True,
                          dropout=0.0)   # no dropout for single layer

        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden),
            nn.Dropout(0.3),
            nn.Linear(gru_hidden, nc))

        n = sum(p.numel() for p in self.parameters())
        print(f'  ConvNeXtTinyGRU: feat={feat_dim}×2→GRU({gru_hidden})  '
              f'{n:,} params  {nc} classes')

    def forward(self, x):
        """
        x: [B, 6, T, H, W]
        """
        B, C, T, H, W = x.shape
        assert C == 6, f"Expected 6-channel input, got {C}"

        rgb = x[:, :3]   # [B, 3, T, H, W]
        thm = x[:, 3:]   # [B, 3, T, H, W]

        # Reshape to [B*T, 3, H, W] for efficient per-frame encoding
        rgb_flat  = rgb.permute(0,2,1,3,4).reshape(B*T, 3, H, W)
        thm_flat  = thm.permute(0,2,1,3,4).reshape(B*T, 3, H, W)

        rgb_feats = self.rgb_encoder(rgb_flat).reshape(B, T, -1)  # [B,T,768]
        thm_feats = self.thm_encoder(thm_flat).reshape(B, T, -1)  # [B,T,768]

        # Concatenate on feature dim → [B, T, 1536]
        fused = torch.cat([rgb_feats, thm_feats], dim=-1)

        # GRU temporal modelling → use last time-step's hidden state
        gru_out, _ = self.gru(fused)          # [B, T, 256]
        last_hidden = gru_out[:, -1, :]       # [B, 256]

        return self.head(last_hidden)


def build_convnext_tiny_gru(nc):
    return ConvNeXtTinyGRU(nc)


# ---- Smoke tests ----
print('\nGestFormerFusion smoke test...')
_T = 8 if CPU_DEBUG_MODE else 16; _H = 112 if CPU_DEBUG_MODE else 224
with torch.no_grad():
    _o = build_gestformer_fusion(NUM_CLASSES).eval()(torch.randn(2, 6, _T, _H, _H))
assert _o.shape == (2, NUM_CLASSES), f'GestFormerFusion smoke failed: {_o.shape}'
print(f'  GestFormerFusion smoke test passed: {tuple(_o.shape)}')
del _o

print('\nConvNeXtTinyGRU smoke test...')
with torch.no_grad():
    _o = build_convnext_tiny_gru(NUM_CLASSES).eval()(torch.randn(2, 6, _T, _H, _H))
assert _o.shape == (2, NUM_CLASSES), f'ConvNeXtTinyGRU smoke failed: {_o.shape}'
print(f'  ConvNeXtTinyGRU smoke test passed: {tuple(_o.shape)}')
del _o

# ================================================================
#  HYPERPARAMETERS
# ================================================================

if CPU_DEBUG_MODE:
    PARAMS_GESTFORMER = dict(
        num_epochs=3, batch_size=2, num_frames=8,  img_size=112,
        lr=5e-4, weight_decay=0.01, warmup_epochs=1, patience=3,
        num_workers=0)
    PARAMS_CONVNEXT = dict(
        num_epochs=3, batch_size=2, num_frames=8,  img_size=112,
        lr=1e-4, weight_decay=0.01, warmup_epochs=1, patience=3,
        num_workers=0)
else:
    # GestFormerFusion: larger model, needs smaller batch
    PARAMS_GESTFORMER = dict(
        num_epochs=50, batch_size=8,  num_frames=16, img_size=224,
        lr=5e-4, weight_decay=0.01, warmup_epochs=8, patience=5,
        num_workers=8)
    # ConvNeXtTinyGRU: two ConvNeXt streams + GRU, similar memory footprint
    PARAMS_CONVNEXT = dict(
        num_epochs=50, batch_size=8,  num_frames=16, img_size=224,
        lr=1e-4, weight_decay=0.01, warmup_epochs=5, patience=5,
        num_workers=8)

print('\nGestFormerFusion params:')
for k, v in PARAMS_GESTFORMER.items(): print(f'  {k:<18}: {v}')
print('\nConvNeXtTinyGRU params:')
for k, v in PARAMS_CONVNEXT.items():   print(f'  {k:<18}: {v}')

# ================================================================
#  TRAINING ENGINE  (shared by both models)
# ================================================================

def run_epoch(model, loader, optimizer, criterion, is_train, scaler=None):
    model.train() if is_train else model.eval()
    total_loss, all_labels, all_preds = 0.0, [], []
    use_amp  = (scaler is not None) and torch.cuda.is_available()
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()

    with grad_ctx:
        for b, (clips, labels) in enumerate(loader):
            clips  = clips.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                    device_type='cuda' if torch.cuda.is_available() else 'cpu',
                    dtype=torch.float16 if use_amp else torch.float32,
                    enabled=use_amp):
                logits = model(clips)
                loss   = criterion(logits, labels)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(logits.detach().argmax(1).cpu().tolist())
            total_loss += loss.item()
            if is_train and (b + 1) % 10 == 0:
                print(f'    [{b+1}/{len(loader)}] loss={total_loss/(b+1):.4f}', end='\r')

    avg = total_loss / max(len(loader), 1)
    acc = accuracy_score(all_labels, all_preds) * 100
    mf1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    return avg, acc, mf1, all_labels, all_preds


def train_and_eval(model, trl, vll, tel, p, ckpt_path):
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    if scaler: print('  AMP (FP16) active')

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Separate LRs: backbone at lr*0.1, head/fusion layers at lr
    head_keys = ['head', 'cls', 'fusion_norm', 'gru']
    head_p = [x for n, x in model.named_parameters()
              if x.requires_grad and any(k in n for k in head_keys)]
    back_p = [x for n, x in model.named_parameters()
              if x.requires_grad and not any(k in n for k in head_keys)]
    groups = (([{'params': back_p, 'lr': p['lr'] * 0.1}] if back_p else []) +
              ([{'params': head_p, 'lr': p['lr']}]        if head_p else []))
    if not groups: groups = [{'params': model.parameters(), 'lr': p['lr']}]
    opt = torch.optim.AdamW(groups, weight_decay=p['weight_decay'])

    n_ep = p['num_epochs']; n_wu = p['warmup_epochs']
    def lr_lam(ep):
        if ep < n_wu: return ep / max(1, n_wu)
        q = (ep - n_wu) / max(1, n_ep - n_wu)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * q)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lam)

    best_val, best_ep, p_cnt, history = 0.0, 0, 0, []

    for epoch in range(1, n_ep + 1):
        t0 = time.time()
        tl, ta, tf, _, _ = run_epoch(model, trl, opt, criterion, True,  scaler)
        sched.step()
        vl, va, vf, _, _ = run_epoch(model, vll, None, criterion, False, None)
        el = time.time() - t0
        history.append(dict(epoch=epoch,
            tr_loss=round(tl,4), tr_acc=round(ta,2), tr_f1=round(tf,2),
            vl_loss=round(vl,4), vl_acc=round(va,2), vl_f1=round(vf,2),
            time_s=round(el,1)))
        print(f'  Epoch {epoch:3d}/{n_ep}  tr {tl:.4f}/{ta:.1f}%  '
              f'vl {vl:.4f}/{va:.1f}%  [{el:.0f}s]')

        if va > best_val + 0.1:
            best_val, best_ep, p_cnt = va, epoch, 0
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_acc': va, 'val_f1': vf}, ckpt_path)
            print(f'  >> Saved best  val_acc={va:.1f}%')
        else:
            p_cnt += 1
        if p_cnt >= p['patience']:
            print(f'  Early stop at epoch {epoch}'); break

    if Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck['model_state'])

    model.eval()
    te_loss, te_acc, te_mf1, labels, preds = run_epoch(
        model, tel, None, criterion, False, None)
    wf1 = f1_score(labels, preds, average='weighted', zero_division=0) * 100
    cm  = confusion_matrix(labels, preds)
    return pd.DataFrame(history), dict(
        accuracy=round(te_acc, 2),    macro_f1=round(te_mf1, 2),
        weighted_f1=round(wf1, 2),    test_loss=round(te_loss, 4),
        best_val_acc=round(best_val, 2), best_epoch=best_ep,
        confusion_matrix=cm)

# ================================================================
#  SINGLE PROTOCOL RUN  (OOM retry + skip-if-done)
#  Shared by both models — model identity tracked via tag prefix.
# ================================================================

def run_one_fusion(tag, tr, va, te, primary_col, secondary_col,
                   rdir, cdir, params, build_fn):
    print(f'  train={len(tr)}  val={len(va)}  test={len(te)}')
    if len(tr) == 0 or len(te) == 0:
        print('  [SKIP] empty split'); return None

    ckpt_dir = Path(cdir) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt         = str(ckpt_dir / 'best_model.pt')
    Path(rdir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(rdir) / f'{tag}_metrics.json'

    # ---- Skip if already completed ----
    if Path(ckpt).exists() and metrics_path.exists():
        print(f'  [SKIP] Already completed – loading cached metrics')
        try:
            met = json.load(open(metrics_path))
            print(f'  CACHED  acc={met.get("accuracy","?")}  '
                  f'macro_f1={met.get("macro_f1","?")}')
            return {**met, 'name': tag, 'primary_col': primary_col}
        except Exception:
            pass

    if Path(ckpt).exists(): Path(ckpt).unlink()

    bs = params['batch_size']
    for attempt, batch_size in enumerate([bs, bs//2, bs//4, 1], 1):
        if batch_size < 1: break
        if attempt > 1:
            print(f'  [OOM RETRY] attempt {attempt}  batch_size={batch_size}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.synchronize()
            gc.collect()

        try:
            p_run = dict(params); p_run['batch_size'] = batch_size
            trl, vll, tel = make_fusion_loaders(
                tr, va, te, primary_col, secondary_col,
                p_run['num_frames'], p_run['img_size'],
                p_run['batch_size'], p_run['num_workers'])

            set_seed(42)
            model = build_fn(NUM_CLASSES)
            hist, m = train_and_eval(model, trl, vll, tel, p_run, ckpt)

            try: model.cpu()
            except Exception: pass
            del model, trl, vll, tel
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.synchronize()
            gc.collect()

            hist.to_csv(Path(rdir) / f'{tag}_history.csv', index=False)
            pd.DataFrame(m['confusion_matrix'],
                         index=GESTURES, columns=GESTURES
                         ).to_csv(Path(rdir) / f'{tag}_cm.csv')

            met = {k: v for k, v in m.items() if k != 'confusion_matrix'}
            met.update({'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
                        'batch_size_used': batch_size,
                        'timestamp': str(datetime.datetime.now())})
            json.dump(met, open(metrics_path, 'w'), indent=2, default=str)

            print(f'  RESULT  acc={m["accuracy"]:.1f}%  macro_f1={m["macro_f1"]:.1f}%')
            return dict(
                name=tag, primary_col=primary_col,
                accuracy=m['accuracy'],        macro_f1=m['macro_f1'],
                weighted_f1=m['weighted_f1'],  test_loss=m['test_loss'],
                best_val_acc=m['best_val_acc'], best_epoch=m['best_epoch'],
                n_train=len(tr), n_val=len(va), n_test=len(te),
                batch_size_used=batch_size)

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
                print(f'  [OOM] batch={batch_size}: {str(e)[:80]}')
                try: model.cpu()
                except Exception: pass
                try: del model
                except Exception: pass
                try: del trl, vll, tel
                except Exception: pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache(); torch.cuda.synchronize()
                gc.collect(); continue
            else:
                raise

    print(f'  [FAIL] {tag} – OOM even at batch=1')
    return None

# ================================================================
#  LOAD PROTOCOL SPLITS  (from fusion_splitB.py output)
# ================================================================

def load_fusion_protocol_csv(fusion_type, train_combo, test_dist):
    split_dir  = FUSION_SPLIT_DIR / fusion_type
    train_name = "+".join([d.replace("_feet", "ft") for d in train_combo])
    test_name  = test_dist.replace("_feet", "ft")
    base       = f"{train_name}_to_{test_name}"

    tr_path = split_dir / f"{base}_train.csv"
    va_path = split_dir / f"{base}_val.csv"
    te_path = split_dir / f"{base}_test.csv"

    if not (tr_path.exists() and va_path.exists() and te_path.exists()):
        print(f'  [SKIP] Split CSVs not found for {fusion_type}/{base}')
        return None, None, None

    return pd.read_csv(tr_path), pd.read_csv(va_path), pd.read_csv(te_path)

# ================================================================
#  VISUALISATION HELPERS
# ================================================================

def save_heatmap(df, label, metric, path):
    if df is None or len(df) == 0: return
    mat = df.pivot(index='train_combo', columns='test_distance',
                   values=metric).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(mat, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5,
                vmin=0, vmax=100, ax=ax, annot_kws={'size': 13, 'weight': 'bold'})
    ax.set_title(f'{label}\n{metric} (%)', fontsize=11, pad=10)
    ax.set_xlabel('Test Distance'); ax.set_ylabel('Train Combo')
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


def save_comparison_bar(gest_rows, conv_rows, fusion_type, rdir):
    """Side-by-side accuracy bar: GestFormerFusion vs ConvNeXtTinyGRU."""
    if not gest_rows and not conv_rows: return
    all_protos = sorted(set(
        [r['protocol'] for r in gest_rows] +
        [r['protocol'] for r in conv_rows]))
    gest_acc = {r['protocol']: r['accuracy'] for r in gest_rows}
    conv_acc = {r['protocol']: r['accuracy'] for r in conv_rows}

    x     = np.arange(len(all_protos))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(all_protos) * 0.7), 5))
    b1 = ax.bar(x - width/2,
                [gest_acc.get(p, 0) for p in all_protos],
                width, label='GestFormerFusion', color='steelblue')
    b2 = ax.bar(x + width/2,
                [conv_acc.get(p, 0) for p in all_protos],
                width, label='ConvNeXtTinyGRU',  color='coral')
    ax.bar_label(b1, fmt='%.1f', padding=2, fontsize=7)
    ax.bar_label(b2, fmt='%.1f', padding=2, fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(all_protos, rotation=30, ha='right', fontsize=8)
    ax.set_ylim(0, 115); ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'GestFormerFusion vs ConvNeXtTinyGRU\n{fusion_type}')
    ax.legend(); plt.tight_layout()
    p = Path(rdir) / f'{fusion_type}_model_comparison.pdf'
    fig.savefig(p, bbox_inches='tight', dpi=150); plt.close(fig)
    print(f'  Saved: {p}')

# ================================================================
#  MASTER RUNNER
#  Outer loop: fusion_type (2)
#  Inner loop: protocols (21)
#  Each protocol runs BOTH models on the SAME split CSVs.
#  Results stored separately per model for clean analysis.
# ================================================================

GRAND_START = time.time()

# Output roots — separate per model
GEST_ROOT  = PROJECT_ROOT / 'Gresults'     / 'gestformer_fusion'
GEST_CKPTS = PROJECT_ROOT / 'Gcheckpoints' / 'gestformer_fusion'
CONV_ROOT  = PROJECT_ROOT / 'Cresults'     / 'convnext_gru_fusion'
CONV_CKPTS = PROJECT_ROOT / 'Ccheckpoints' / 'convnext_gru_fusion'

for d in [GEST_ROOT, GEST_CKPTS, CONV_ROOT, CONV_CKPTS]:
    d.mkdir(parents=True, exist_ok=True)

all_gest_rows = []
all_conv_rows = []

for fusion_type, cfg in FUSION_TYPES.items():
    print('\n' + '=' * 60)
    print(f'  FUSION TYPE: {fusion_type.upper()}')
    print('=' * 60)

    primary_col   = cfg["primary_col"]
    secondary_col = cfg["secondary_col"]

    # Separate result / checkpoint dirs per model per fusion type
    gest_rdir = str(GEST_ROOT  / fusion_type)
    gest_cdir = str(GEST_CKPTS / fusion_type)
    conv_rdir = str(CONV_ROOT  / fusion_type)
    conv_cdir = str(CONV_CKPTS / fusion_type)

    gest_rows_ftype = []
    conv_rows_ftype = []

    total   = len(TRAIN_COMBINATIONS) * len(TEST_DISTANCES)
    counter = 0

    for train_combo in TRAIN_COMBINATIONS:
        for test_dist in TEST_DISTANCES:
            counter += 1
            train_name = "+".join([d.replace("_feet", "ft") for d in train_combo])
            test_name  = test_dist.replace("_feet", "ft")
            proto_name = f"{train_name}_to_{test_name}"

            print(f'\n  [{counter}/{total}] {fusion_type} | {proto_name}')

            tr, va, te = load_fusion_protocol_csv(
                fusion_type, train_combo, test_dist)
            if tr is None:
                continue

            print(f'    train={len(tr)}  val={len(va)}  test={len(te)}')

            # ---- Run GestFormerFusion ----
            print(f'  >> GestFormerFusion')
            gest_tag = f'gestformer_{fusion_type}_{proto_name}'
            r_gest = run_one_fusion(
                tag           = gest_tag,
                tr=tr, va=va, te=te,
                primary_col   = primary_col,
                secondary_col = secondary_col,
                rdir          = gest_rdir,
                cdir          = gest_cdir,
                params        = PARAMS_GESTFORMER,
                build_fn      = build_gestformer_fusion,
            )
            if r_gest:
                r_gest.update({'fusion_type': fusion_type,
                               'protocol':    proto_name,
                               'train_combo': train_name,
                               'test_distance': test_dist,
                               'model':       'GestFormerFusion'})
                gest_rows_ftype.append(r_gest)
                all_gest_rows.append(r_gest)

            # ---- Run ConvNeXtTinyGRU (same split, same loaders) ----
            print(f'  >> ConvNeXtTinyGRU')
            conv_tag = f'convnext_{fusion_type}_{proto_name}'
            r_conv = run_one_fusion(
                tag           = conv_tag,
                tr=tr, va=va, te=te,
                primary_col   = primary_col,
                secondary_col = secondary_col,
                rdir          = conv_rdir,
                cdir          = conv_cdir,
                params        = PARAMS_CONVNEXT,
                build_fn      = build_convnext_tiny_gru,
            )
            if r_conv:
                r_conv.update({'fusion_type': fusion_type,
                               'protocol':    proto_name,
                               'train_combo': train_name,
                               'test_distance': test_dist,
                               'model':       'ConvNeXtTinyGRU'})
                conv_rows_ftype.append(r_conv)
                all_conv_rows.append(r_conv)

    # ---- Per-fusion-type heatmaps ----
    if gest_rows_ftype:
        fdf = pd.DataFrame(gest_rows_ftype)
        save_heatmap(fdf, f'GestFormerFusion | {fusion_type}', 'accuracy',
                     Path(gest_rdir) / f'{fusion_type}_accuracy_heatmap.pdf')
    if conv_rows_ftype:
        fdf = pd.DataFrame(conv_rows_ftype)
        save_heatmap(fdf, f'ConvNeXtTinyGRU | {fusion_type}', 'accuracy',
                     Path(conv_rdir) / f'{fusion_type}_accuracy_heatmap.pdf')

    # ---- Side-by-side comparison bar chart ----
    save_comparison_bar(gest_rows_ftype, conv_rows_ftype,
                        fusion_type, str(GEST_ROOT))

# ================================================================
#  SAVE MASTER SUMMARIES
# ================================================================

gest_master = pd.DataFrame(all_gest_rows)
conv_master = pd.DataFrame(all_conv_rows)

gest_master.to_csv(GEST_ROOT / 'gestformer_fusion_master.csv', index=False)
conv_master.to_csv(CONV_ROOT / 'convnext_gru_fusion_master.csv', index=False)

# Combined table for direct comparison
if len(all_gest_rows) > 0 and len(all_conv_rows) > 0:
    combined = pd.concat([gest_master, conv_master], ignore_index=True)
    combined.to_csv(PROJECT_ROOT / 'CCresults' / 'fusion_combined_comparison.csv',
                    index=False)

# ================================================================
#  PRINT COMBINED SUMMARY
# ================================================================

print('\n' + '=' * 60)
print('  COMBINED FUSION RESULTS SUMMARY')
print('=' * 60)

for model_name, rows in [('GestFormerFusion', all_gest_rows),
                          ('ConvNeXtTinyGRU',  all_conv_rows)]:
    if not rows: continue
    df_m = pd.DataFrame(rows)
    print(f'\n  {model_name}:')
    summary = (df_m.groupby(['fusion_type'])['accuracy']
               .agg(['mean', 'max', 'min'])
               .round(2))
    print(summary.to_string())

# Head-to-head per protocol
if all_gest_rows and all_conv_rows:
    print('\n  Head-to-head (mean accuracy by fusion type):')
    for ftype in FUSION_TYPES:
        g = np.mean([r['accuracy'] for r in all_gest_rows
                     if r.get('fusion_type') == ftype])
        c = np.mean([r['accuracy'] for r in all_conv_rows
                     if r.get('fusion_type') == ftype])
        print(f'  {ftype:25s}  GestFormer={g:.1f}%  ConvNeXtGRU={c:.1f}%  '
              f'gap={g-c:+.1f}%')

grand_elapsed = str(datetime.timedelta(seconds=int(time.time() - GRAND_START)))
print(f'\n{"="*60}')
print(f'  ALL FUSION MODELS | ALL 42 PROTOCOLS × 2 MODELS DONE')
print(f'  Total wall time: {grand_elapsed}')
print(f'  GestFormerFusion Gresults : {GEST_ROOT}')
print(f'  ConvNeXtTinyGRU  Cresults : {CONV_ROOT}')
print(f'  Combined comparison      : {PROJECT_ROOT}/CCresults/fusion_combined_comparison.csv')
print(f'{"="*60}')
