

CPU_DEBUG_MODE = False  # True=local CPU debug | False=RTX A4000 GPU

import sys, os, gc, time, random, math, datetime, warnings
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
warnings.filterwarnings('ignore')

# ================================================================
#  DEVICE
# ================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
# benchmark=True only makes sense with deterministic=False.
# We keep deterministic=True for reproducibility; benchmark set to False here,
# and set_seed() will also enforce it before each run.


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

# ================================================================
#  PATHS
# ================================================================

PROJECT_ROOT  = Path(".")
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "paired_manifest.csv"
SPLIT_DIR     = PROJECT_ROOT / "manifests" / "splits"
SPLIT_3X3     = SPLIT_DIR / "3x3"

# ================================================================
#  LOAD MANIFEST  (single load -- no duplicate block)
# ================================================================
manifest = pd.read_csv(MANIFEST_PATH)

# FIX: create distance_canon if missing
if 'distance_canon' not in manifest.columns:
    print("[INFO] 'distance_canon' not found → creating from 'distance'")
    manifest['distance_canon'] = manifest['distance']


for col in ['pair_id', 'subject_id', 'base_subject_id', 'distance', 'distance_canon']:
    manifest[col] = manifest[col].astype(str)

REQUIRED = {'pair_id', 'subject_id', 'base_subject_id', 'gesture',
            'distance', 'distance_canon', 'rgb_frame_dir', 'thermal_frame_dir'}
assert not (REQUIRED - set(manifest.columns)), \
    f'Missing columns: {REQUIRED - set(manifest.columns)}'

GESTURES        = sorted(manifest['gesture'].unique().tolist())
GESTURE_TO_IDX  = {g: i for i, g in enumerate(GESTURES)}
NUM_CLASSES     = len(GESTURES)
DISTANCES_CANON = sorted(manifest['distance_canon'].unique().tolist())

print(f'Project root  : {PROJECT_ROOT}')
print(f'Manifest      : {len(manifest)} pairs | {NUM_CLASSES} gestures | {len(DISTANCES_CANON)} distances')
print(f'Gestures      : {GESTURES}')
print(f'Distances     : {DISTANCES_CANON}')
print(f'CPU debug mode: {CPU_DEBUG_MODE}')

# ================================================================
#  21 PROTOCOL DEFINITIONS (FIXED)
# ================================================================

TRAIN_COMBINATIONS = [
    ["4_feet"],
    ["6_feet"],
    ["8_feet"],
    ["4_feet","6_feet"],
    ["4_feet","8_feet"],
    ["6_feet","8_feet"],
    ["4_feet","6_feet","8_feet"]
]

TEST_DISTANCES = ["4_feet","6_feet","8_feet"]


def load_21_protocol(train_combo, test_dist):
    """
    Build train/val/test splits for 21 protocols
    """

    # --- TRAIN ---
    tr = manifest[manifest['distance'].isin(train_combo)].copy()

    # --- TEST ---
    te = manifest[manifest['distance'] == test_dist].copy()

    # --- VALIDATION (from TRAIN subjects) ---
    if len(tr) == 0 or len(te) == 0:
        print("  [ERROR] Empty split")
        return None, None, None

    subjs = sorted(tr['base_subject_id'].unique())
    random.seed(42)
    random.shuffle(subjs)

    n = len(subjs)
    n_val = max(1, int(n * 0.15))

    val_subj = set(subjs[:n_val])
    train_subj = set(subjs[n_val:])

    va = tr[tr['base_subject_id'].isin(val_subj)].reset_index(drop=True)
    tr = tr[tr['base_subject_id'].isin(train_subj)].reset_index(drop=True)
    te = te.reset_index(drop=True)

    # ---- FINAL SAFETY CHECK ----
    if len(tr) == 0 or len(te) == 0:
        print("  [SKIP] Empty after split")
        return None, None, None

    return tr, va, te
# ================================================================
#  MODALITY PATH RESOLVER
#
#  rgb       -> rgb_frame_dir  column
#  thermal       -> thermal_frame_dir  column  (thermal camera; was 'thermal' in old version)
#  rgb_bgrem -> derived from rgb_frame_dir by replacing /RGB/ with /RGB_BGREM/
# ================================================================

def get_frame_dir(row, modality):
    if modality == 'thermal':
        return str(row['thermal_frame_dir'])
    p = str(row['rgb_frame_dir'])
    if modality == 'rgb':
        return p
    # rgb_bgrem: replace the RGB path segment
    p_fwd = p.replace(os.sep, '/')
    if '/RGB/' in p_fwd:
        return p_fwd.replace('/RGB/', '/RGB_BGREM/', 1)
    return p.replace('RGB', 'RGB_BGREM', 1) if 'RGB' in p else p

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
#  LAZY RAM FRAME CACHE
# ================================================================



# ================================================================
#  CLIP DATASET
# ================================================================

class GestureClipDataset(Dataset):
    EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

    def __init__(self, df, modality, num_frames, img_size, is_train):
        self.df        = df.reset_index(drop=True)
        self.modality  = modality
        self.num_frames = num_frames
        self.img_size  = img_size
        self.is_train  = is_train

        if is_train:
            self.spatial = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(0.5)])
        else:
            self.spatial = T.Compose([
                T.Resize(int(img_size * 1.14)),
                T.CenterCrop(img_size)])

        # colour jitter only for RGB (not thermal, not bgrem)
        ex = [T.ColorJitter(0.3, 0.3, 0.2)] if (is_train and modality == 'rgb') else []
        is_thermal = (modality == 'thermal')

        self.cpu_tf = T.Compose([
            T.Grayscale(3) if is_thermal else T.Lambda(lambda x: x),
            *([T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
               T.RandomHorizontalFlip(0.5)] if is_train else
              [T.Resize(int(img_size * 1.14)), T.CenterCrop(img_size)]),
            *ex,
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = GESTURE_TO_IDX[row['gesture']]
        fdir  = get_frame_dir(row, self.modality)
        return self._load_clip(fdir), torch.tensor(label, dtype=torch.long)

    def _load_clip(self, frame_dir):
        p    = Path(frame_dir)
        zero = torch.zeros(3, self.num_frames, self.img_size, self.img_size)
        if not p.exists():
            print(f"[WARNING] Missing dir: {frame_dir}")
            return zero
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in self.EXTS])
        if not files: return zero
        sampler = sample_random if self.is_train else sample_uniform
        frames  = []
        for i in sampler(len(files), self.num_frames):
            img = Image.open(str(files[i]))
            if img is None: img = Image.open(str(files[i]))
            else:
                img = img.convert('L') if self.modality == 'thermal' else img.convert('RGB')
                t   = self.cpu_tf(img)
            frames.append(t)
        return torch.stack(frames, dim=0).permute(1, 0, 2, 3)


def make_loaders(tr, va, te, mod, nf, sz, bs, nw):
    kw = dict(num_workers=nw, pin_memory=False)
    return (DataLoader(GestureClipDataset(tr, mod, nf, sz, True),  bs, shuffle=True,
                       drop_last=len(tr) > bs, **kw),
            DataLoader(GestureClipDataset(va, mod, nf, sz, False), bs, shuffle=False, **kw),
            DataLoader(GestureClipDataset(te, mod, nf, sz, False), bs, shuffle=False, **kw))


# ================================================================
#  SEED
# ================================================================

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

# ================================================================
#  TRAINING ENGINE
# ================================================================

def run_epoch(model, loader, optimizer, criterion, is_train, scaler=None):
    model.train() if is_train else model.eval()
    total_loss, all_labels, all_preds = 0.0, [], []
    use_amp  = (scaler is not None) and torch.cuda.is_available()
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()

    with grad_ctx:
        for b, (clips, labels) in enumerate(loader):
            clips, labels = (clips.to(DEVICE, non_blocking=True),
                             labels.to(DEVICE, non_blocking=True))
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
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
    mf1 = f1_score(all_labels, all_preds, average='macro',    zero_division=0) * 100
    return avg, acc, mf1, all_labels, all_preds


def train_and_eval(model, trl, vll, tel, p, ckpt_path):
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    if scaler: print('  AMP (FP16) active')

    model = model.to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    head_p = [x for n, x in model.named_parameters()
              if x.requires_grad and any(k in n for k in ['head', 'cls', 'proj'])]
    back_p = [x for n, x in model.named_parameters()
              if x.requires_grad and not any(k in n for k in ['head', 'cls', 'proj'])]
    groups = ([{'params': back_p, 'lr': p['lr'] * 0.1}] if back_p else []) + \
             ([{'params': head_p, 'lr': p['lr']}]        if head_p else [])
    if not groups: groups = [{'params': model.parameters(), 'lr': p['lr']}]
    opt = torch.optim.AdamW(groups, weight_decay=p['weight_decay'])

    # LR schedule: stepped per EPOCH (warmup + cosine decay over num_epochs epochs)
    n_ep = p['num_epochs']; n_wu = p['warmup_epochs']
    def lr_lam(ep):
        if ep < n_wu: return ep / max(1, n_wu)
        q = (ep - n_wu) / max(1, n_ep - n_wu)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * q)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lam)

    best_val, best_ep, p_cnt, history = 0.0, 0, 0, []

    for epoch in range(1, n_ep + 1):
        t0 = time.time()
        ua = p.get('unfreeze_at')
        if ua and epoch == ua and hasattr(model, 'unfreeze'):
            model.unfreeze()
            opt = torch.optim.AdamW(model.parameters(),
                                    lr=p['lr'] * 0.1, weight_decay=p['weight_decay'])

        tl, ta, tf, _, _ = run_epoch(model, trl, opt, criterion, True,  scaler)
        sched.step()
        vl, va, vf, _, _ = run_epoch(model, vll, None, criterion, False, None)
        el = time.time() - t0
        history.append(dict(epoch=epoch,
            tr_loss=round(tl,4), tr_acc=round(ta,2), tr_f1=round(tf,2),
            vl_loss=round(vl,4), vl_acc=round(va,2), vl_f1=round(vf,2),
            time_s=round(el,1)))
        print(f'  Epoch {epoch:3d}/{n_ep}  tr {tl:.4f}/{ta:.1f}%  vl {vl:.4f}/{va:.1f}%  [{el:.0f}s]')

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
    te_loss, te_acc, te_mf1, labels, preds = run_epoch(model, tel, None, criterion, False, None)
    wf1 = f1_score(labels, preds, average='weighted', zero_division=0) * 100
    cm  = confusion_matrix(labels, preds)
    return pd.DataFrame(history), dict(
        accuracy=round(te_acc, 2), macro_f1=round(te_mf1, 2),
        weighted_f1=round(wf1, 2), test_loss=round(te_loss, 4),
        best_val_acc=round(best_val, 2), best_epoch=best_ep,
        confusion_matrix=cm)

# ================================================================
#  run_one  -- single protocol run with OOM retry

def run_one(name, tr, va, te, modality, rdir, cdir, params, build_fn, subname=''):
    print(f'  train={len(tr)}  val={len(va)}  test={len(te)}')
    if len(tr) == 0 or len(te) == 0:
        print('  [SKIP] empty split'); return None

    tag      = f'{name}_{subname}' if subname else name
    ckpt_dir = Path(cdir) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt     = str(ckpt_dir / 'best_model.pt')
    Path(rdir).mkdir(parents=True, exist_ok=True)

    # ---- SKIP IF ALREADY COMPLETED (checkpoint + metrics both exist) ----
    metrics_path = Path(rdir) / f'{tag}_metrics.json'
    if Path(ckpt).exists() and metrics_path.exists():
        print(f'  [SKIP] Already completed -- loading saved metrics')
        import json as _j
        try:
            met = _j.load(open(metrics_path))
            print(f'  CACHED  acc={met.get("accuracy","?")}  macro_f1={met.get("macro_f1","?")}')
            return dict(
                name=tag, modality=modality,
                accuracy=met['accuracy'],       macro_f1=met['macro_f1'],
                weighted_f1=met['weighted_f1'], test_loss=met['test_loss'],
                best_val_acc=met['best_val_acc'], best_epoch=met['best_epoch'],
                n_train=met['n_train'], n_val=met['n_val'], n_test=met['n_test'],
                batch_size_used=met['batch_size_used'])
        except Exception as e:
            print(f'  [WARN] Could not load cached metrics ({e}) -- rerunning')

    # ---- REMOVE ONLY STALE PARTIAL FILES (no checkpoint = not completed) ----
    for f in [Path(rdir)/f'{tag}_history.csv',
              metrics_path,
              Path(rdir)/f'{tag}_cm.csv']:
        if f.exists(): f.unlink()
    if Path(ckpt).exists(): Path(ckpt).unlink()

    # ---- TRAIN WITH OOM RETRY ----
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
            trl, vll, tel = make_loaders(tr, va, te, modality,
                                         p_run['num_frames'], p_run['img_size'],
                                         p_run['batch_size'], p_run['num_workers'])
            set_seed(42)
            model = build_fn(NUM_CLASSES)
            hist, m = train_and_eval(model, trl, vll, tel, p_run, ckpt)

            # cleanup GPU memory
            try: model.cpu()
            except Exception: pass
            del model, trl, vll, tel
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.synchronize()
            gc.collect()

            # save outputs
            hist.to_csv(Path(rdir)/f'{tag}_history.csv', index=False)
            pd.DataFrame(m['confusion_matrix'],
                         index=GESTURES, columns=GESTURES
                         ).to_csv(Path(rdir)/f'{tag}_cm.csv')

            import json as _j
            met = {k: v for k, v in m.items() if k != 'confusion_matrix'}
            met.update({'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
                        'batch_size_used': batch_size,
                        'timestamp': str(datetime.datetime.now())})
            _j.dump(met, open(metrics_path, 'w'), indent=2, default=str)

            print(f'  RESULT  acc={m["accuracy"]:.1f}%  macro_f1={m["macro_f1"]:.1f}%')

            return dict(
                name=tag, modality=modality,
                accuracy=m['accuracy'],       macro_f1=m['macro_f1'],
                weighted_f1=m['weighted_f1'], test_loss=m['test_loss'],
                best_val_acc=m['best_val_acc'], best_epoch=m['best_epoch'],
                n_train=len(tr), n_val=len(va), n_test=len(te),
                batch_size_used=batch_size)

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
                print(f'  [OOM] batch={batch_size} failed: {str(e)[:80]}')
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

    print(f'  [FAIL] {tag} -- OOM even at batch=1')
    return None

#  VISUALISATION HELPERS
# ================================================================

def save_heatmap(df, modality, metric, path):
    if df is None or len(df) == 0: return
    mat = (df.pivot(index='train_distance', columns='test_distance', values=metric)
             .reindex(index=DISTANCES_CANON, columns=DISTANCES_CANON).fillna(0))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(mat, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5,
                vmin=0, vmax=100, ax=ax, annot_kws={'size': 13, 'weight': 'bold'})
    for i in range(len(DISTANCES_CANON)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=2.5))
    ax.set_title(f'{modality.upper()} -- 3x3\n{metric} (%)', fontsize=11, pad=10)
    ax.set_xlabel('Test Distance'); ax.set_ylabel('Train Distance')
    plt.tight_layout(); fig.savefig(path, bbox_inches='tight', dpi=150); plt.close(fig)
    print(f'  Saved: {path}')

def save_bar(rows, tag, rdir):
    if not rows: return
    names = [r.get('split_name', r['name']) for r in rows]
    vals  = [r['accuracy'] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 4))
    bars = ax.bar(names, vals, color='steelblue', edgecolor='white')
    ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
    ax.set_ylim(0, 110); ax.set_ylabel('Accuracy (%)'); ax.set_title(tag)
    plt.xticks(rotation=25, ha='right', fontsize=8); plt.tight_layout()
    p = Path(rdir) / f'{tag}_accuracy_bar.pdf'
    fig.savefig(p, bbox_inches='tight', dpi=150); plt.close(fig)
    print(f'  Saved: {p}')

# ================================================================
#  MASTER RUNNER -- all 21 protocols for one model
# ================================================================
# ================================================================
#  MASTER RUNNER -- FIXED 21 PROTOCOL LOOP
# ================================================================

def run_all_protocols(model_name, build_fn, modalities, results_root, ckpt_root, params):

    results_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    t0 = time.time()

    for modality in modalities:
        print("\n" + "#"*60)
        print(f"  {model_name} -- {modality.upper()} -- 21 PROTOCOLS")
        print("#"*60)

        rdir = str(results_root / modality)
        cdir = str(ckpt_root / modality)

        total = len(TRAIN_COMBINATIONS) * len(TEST_DISTANCES)
        counter = 0

        for train_combo in TRAIN_COMBINATIONS:
            for test_dist in TEST_DISTANCES:

                counter += 1
                name = f"{'+'.join(train_combo)}_to_{test_dist}"

                print(f"\n  [{counter}/{total}] {name}")

                tr, va, te = load_21_protocol(train_combo, test_dist)

                if tr is None:
                    continue

                print(f"    train={len(tr)} val={len(va)} test={len(te)}")

                r = run_one(
                    "protocol21",
                    tr, va, te,
                    modality,
                    rdir,
                    cdir,
                    params,
                    build_fn,
                    subname=name
                )

                if r:
                    r['protocol'] = '21'
                    r['train_combo'] = "+".join(train_combo)
                    r['test_distance'] = test_dist
                    r['model'] = model_name
                    all_rows.append(r)

    # SAVE
    master = pd.DataFrame(all_rows)
    master.to_csv(results_root / 'master_summary.csv', index=False)

    print(f"\n  Saved: {results_root}/master_summary.csv")

    elapsed = str(datetime.timedelta(seconds=int(time.time()-t0)))
    print(f"\n  {model_name} DONE. Time: {elapsed}")

    return master
#  MODEL 1: GestFormer
# ================================================================

import torchvision.models as tvm

class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        if CPU_DEBUG_MODE:
            base = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.out_dim = 576
        else:
            base = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.out_dim = 1280
        self.features = base.features; self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x): return self.pool(self.features(x)).flatten(1)

class TemporalPE(nn.Module):
    def __init__(self, d, max_len=72, dropout=0.1):
        super().__init__(); self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.drop(x + self.pe[:, :x.size(1)])

class GestFormer(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.spatial = SpatialEncoder(); bd = self.spatial.out_dim
        td = 128 if CPU_DEBUG_MODE else 256
        nh = 4   if CPU_DEBUG_MODE else 8
        nl = 2   if CPU_DEBUG_MODE else 4
        fd = 256 if CPU_DEBUG_MODE else 512
        Tf = 8   if CPU_DEBUG_MODE else 16
        self.proj = nn.Sequential(nn.Linear(bd, td), nn.LayerNorm(td))
        self.cls  = nn.Parameter(torch.zeros(1, 1, td))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.pe = TemporalPE(td, max_len=Tf+8)
        enc = nn.TransformerEncoderLayer(d_model=td, nhead=nh, dim_feedforward=fd,
                                          dropout=0.1, activation='gelu',
                                          batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=nl)
        self.head = nn.Sequential(nn.LayerNorm(td), nn.Dropout(0.1), nn.Linear(td, nc))
        nn.init.trunc_normal_(self.head[-1].weight, std=0.02)
        nn.init.zeros_(self.head[-1].bias)
        n = sum(p.numel() for p in self.parameters())
        print(f'  GestFormer: {nl}Lx{nh}Hx{td}D  {n:,} params  {nc} classes')

    def forward(self, x):
        B, C, T, H, W = x.shape
        feats  = self.spatial(x.permute(0,2,1,3,4).reshape(B*T, C, H, W)).reshape(B, T, -1)
        tokens = self.proj(feats)
        tokens = self.pe(torch.cat([self.cls.expand(B,-1,-1), tokens], dim=1))
        return self.head(self.transformer(tokens)[:, 0])

def build_gestformer(nc): return GestFormer(nc)

# smoke test
print('\nGestFormer smoke test ...')
_T = 8 if CPU_DEBUG_MODE else 16; _H = 112 if CPU_DEBUG_MODE else 224
with torch.no_grad():
    _o = build_gestformer(NUM_CLASSES).eval()(torch.randn(2, 3, _T, _H, _H))
assert _o.shape == (2, NUM_CLASSES), f'Smoke test failed: {_o.shape}'
print(f'GestFormer smoke test passed: {tuple(_o.shape)}')
del _o

# ================================================================
#  MODEL 2: VideoMAE
# ================================================================

class _LightFallback(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3,32,3,padding=1), nn.BatchNorm3d(32), nn.ReLU(True), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(32,64,3,padding=1), nn.BatchNorm3d(64), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(64,128,3,padding=1), nn.BatchNorm3d(128), nn.ReLU(True), nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, nc))
        print(f'  LightFallback: {sum(p.numel() for p in self.parameters()):,} params')
    def forward(self, x): return self.head(self.net(x))

class _FallbackSignal(Exception):
    """Raised inside _VM.__init__ to trigger LightFallback cleanly."""
    pass

def build_videomae(nc):
    if CPU_DEBUG_MODE: return _LightFallback(nc)
    try:
        from transformers import VideoMAEModel
        class _VM(nn.Module):
            def __init__(self, nc):
                super().__init__()
                print('  Loading facebook/videomae-base ...')
                # Try loading from local cache first, then remote
                try:
                    self.encoder = VideoMAEModel.from_pretrained(
                        'facebook/videomae-base',
                        local_files_only=False
                    )
                except (OSError, EnvironmentError) as e:
                    print(f'  [WARN] VideoMAE download failed: {str(e)[:120]}')
                    print('  [WARN] Falling back to LightFallback (3D-CNN)')
                    raise _FallbackSignal()
                h = self.encoder.config.hidden_size
                self.head = nn.Sequential(nn.LayerNorm(h), nn.Dropout(0.3), nn.Linear(h, nc))
                for p in self.encoder.parameters(): p.requires_grad = False
                print('  Backbone frozen.')
            def unfreeze(self):
                for p in self.encoder.parameters(): p.requires_grad = True
                print('  VideoMAE backbone unfrozen')
            def forward(self, x):
                return self.head(self.encoder(pixel_values=x).last_hidden_state.mean(1))
        return _VM(nc)
    except _FallbackSignal:
        print('  Using LightFallback instead of VideoMAE')
        return _LightFallback(nc)
    except ImportError:
        print('  transformers not installed -- pip install transformers')
        return _LightFallback(nc)
# ================================================================
#  HYPERPARAMETERS
# ================================================================

if CPU_DEBUG_MODE:
    PARAMS_GEST = dict(num_epochs=3,  batch_size=2,  num_frames=8,  img_size=112,
                       lr=5e-4, weight_decay=0.01, warmup_epochs=1, patience=3,
                       unfreeze_at=None, num_workers=8)
    PARAMS_VMAE = dict(num_epochs=3,  batch_size=2,  num_frames=8,  img_size=112,
                       lr=1e-4, weight_decay=0.05, warmup_epochs=1, patience=3,
                       unfreeze_at=None, num_workers=8)
else:
    PARAMS_GEST = dict(num_epochs=50, batch_size=16, num_frames=16, img_size=224,
                       lr=5e-4, weight_decay=0.01, warmup_epochs=8, patience=5,
                       unfreeze_at=None, num_workers=8)   # Windows: keep num_workers=0
    PARAMS_VMAE = dict(num_epochs=50, batch_size=16, num_frames=16, img_size=224,
                       lr=1e-4, weight_decay=0.05, warmup_epochs=5, patience=5,
                       unfreeze_at=10, num_workers=8)

print('\nGestFormer params:')
for k, v in PARAMS_GEST.items(): print(f'  {k:<18}: {v}')
print('\nVideoMAE params:')
for k, v in PARAMS_VMAE.items(): print(f'  {k:<18}: {v}')

# ================================================================
#  MAIN RUN
# ================================================================

GRAND_START = time.time()
all_masters = []

# ---- GestFormer ----
print('\n' + '='*60)
print('  GESTFORMER  |  RGB + RGB_BGREM + thermal  |  ALL 21 PROTOCOLS')
print('='*60)

GEST_RESULTS = PROJECT_ROOT / 'results'     / 'gestformer'
GEST_CKPTS   = PROJECT_ROOT / 'checkpoints' / 'gestformer'

gest_master = run_all_protocols(
    model_name   = 'GestFormer',
    build_fn     = build_gestformer,
    modalities   = ['rgb', 'rgb_bgrem', 'thermal'],
    results_root = GEST_RESULTS,
    ckpt_root    = GEST_CKPTS,
    params       = PARAMS_GEST,
)
all_masters.append(gest_master)

# ---- VideoMAE ----
print('\n' + '='*60)
print('  VIDEOMAE  |  RGB + RGB_BGREM + thermal  |  ALL 21 PROTOCOLS')
print('='*60)

VMAE_RESULTS = PROJECT_ROOT / 'results'     / 'videomae'
VMAE_CKPTS   = PROJECT_ROOT / 'checkpoints' / 'videomae'

vmae_master = run_all_protocols(
    model_name   = 'VideoMAE',
    build_fn     = build_videomae,
    modalities   = ['rgb', 'rgb_bgrem', 'thermal'],
    results_root = VMAE_RESULTS,
    ckpt_root    = VMAE_CKPTS,
    params       = PARAMS_VMAE,
)
all_masters.append(vmae_master)

# ================================================================
#  COMBINED SUMMARY
# ================================================================

combined = pd.concat(all_masters, ignore_index=True)
combined.to_csv(PROJECT_ROOT / 'results' / 'combined_all_models.csv', index=False)

print('\n' + '='*60)
print('  COMBINED RESULTS SUMMARY')
print('='*60)
if len(combined) > 0:
    summary = (combined
        .groupby(['model', 'modality', 'protocol'])['accuracy']
        .agg(['mean', 'max', 'min'])
        .round(2)
        .reset_index())
    print(summary.to_string(index=False))

if 'model' in combined.columns:
    for mod in ['rgb', 'rgb_bgrem', 'thermal']:
        g = combined[(combined['model'] == 'GestFormer') & (combined['modality'] == mod)]['accuracy'].mean()
        v = combined[(combined['model'] == 'VideoMAE')   & (combined['modality'] == mod)]['accuracy'].mean()
        if not (pd.isna(g) or pd.isna(v)):
            print(f'  {mod.upper():12s}: VideoMAE={v:.1f}%  GestFormer={g:.1f}%  gap={v-g:.1f}%')

grand_elapsed = str(datetime.timedelta(seconds=int(time.time()-GRAND_START)))
print(f'\n{"="*60}')
print(f'  ALL MODELS | ALL MODALITIES | ALL 21 PROTOCOLS DONE')
print(f'  Total wall time : {grand_elapsed}')
print(f'  Results saved to:')
print(f'    {GEST_RESULTS}')
print(f'    {VMAE_RESULTS}')
print(f'    {PROJECT_ROOT}/results/combined_all_models.csv')
print(f'{"="*60}')
