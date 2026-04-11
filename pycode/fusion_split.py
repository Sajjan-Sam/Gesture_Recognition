

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path



TRIMMED_ROOT  = Path("Trimmed")
RGB_ROOT      = TRIMMED_ROOT / "RGB"
THERMAL_ROOT  = TRIMMED_ROOT / "thermal"

MANIFEST_DIR  = Path("manifests")
FUSION_SPLIT_DIR = MANIFEST_DIR / "fusion_splits"

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
FUSION_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

# ================================================================
#  UTIL
# ================================================================

def count_frames(folder):
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)

def base_subject(name):
    m = re.match(r'^.*?_?(S\d+)$', name, re.IGNORECASE)
    return m.group(1).upper() if m else name

def derive_bgrem_dir(rgb_path: str) -> str:
    """Derive RGB_BGREM path from RGB path (same logic as modelB.get_frame_dir)."""
    p = rgb_path.replace(os.sep, '/')
    if '/RGB/' in p:
        return p.replace('/RGB/', '/RGB_BGREM/', 1)
    return p.replace('RGB', 'RGB_BGREM', 1) if 'RGB' in p else p



BASE_MANIFEST_PATH = MANIFEST_DIR / "paired_manifest.csv"

if BASE_MANIFEST_PATH.exists():
    print(f"\n🔹 Loading existing manifest: {BASE_MANIFEST_PATH}")
    base_manifest = pd.read_csv(BASE_MANIFEST_PATH)
    print(f"   Loaded {len(base_manifest)} rows")
else:
    # ---- Build from scratch (identical to original splitB.py Step 1) ----
    print("\n🔹 Building base manifest from scratch...")
    rows = []
    for dist_dir in sorted(RGB_ROOT.iterdir()):
        if not dist_dir.is_dir():
            continue
        distance = dist_dir.name
        for gesture_dir in dist_dir.iterdir():
            if not gesture_dir.is_dir():
                continue
            gesture = gesture_dir.name
            for subj_dir in gesture_dir.iterdir():
                if not subj_dir.is_dir():
                    continue
                subject = subj_dir.name
                for pair_dir in subj_dir.iterdir():
                    if not pair_dir.is_dir():
                        continue
                    thermal_dir = THERMAL_ROOT / distance / gesture / subject / pair_dir.name
                    rgb_n     = count_frames(pair_dir)
                    thermal_n = count_frames(thermal_dir)
                    if rgb_n == 0 or thermal_n == 0:
                        continue
                    rows.append({
                        "pair_id":          pair_dir.name,
                        "subject_id":       subject,
                        "base_subject_id":  base_subject(subject),
                        "gesture":          gesture,
                        "distance":         distance,
                        "rgb_frame_dir":    str(pair_dir),
                        "thermal_frame_dir":str(thermal_dir),
                        "rgb_n_frames":     rgb_n,
                        "thermal_n_frames": thermal_n
                    })
    base_manifest = pd.DataFrame(rows)
    base_manifest.to_csv(BASE_MANIFEST_PATH, index=False)
    print(f"   Built {len(base_manifest)} rows → saved to {BASE_MANIFEST_PATH}")

# ================================================================
#  
#
#  Adds:
#    rgb_bgrem_frame_dir   – derived from rgb_frame_dir
#    bgrem_frames_exist    – bool: does the RGB_BGREM directory have frames?
#
#  Fusion types defined:
#    "rgb_thermal"       → (rgb_frame_dir,       thermal_frame_dir)
#    "rgb_bgrem_thermal" → (rgb_bgrem_frame_dir,  thermal_frame_dir)
# ================================================================

print("\n🔹 Building fusion manifest...")

fusion_manifest = base_manifest.copy()

# Derive rgb_bgrem paths
fusion_manifest['rgb_bgrem_frame_dir'] = fusion_manifest['rgb_frame_dir'].apply(derive_bgrem_dir)

# Check whether bgrem directories actually have frames
def bgrem_has_frames(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    return any(f.suffix.lower() in IMAGE_EXTS for f in p.iterdir())

print("   Checking RGB_BGREM directories (this may take a moment)...")
fusion_manifest['bgrem_frames_exist'] = fusion_manifest['rgb_bgrem_frame_dir'].apply(bgrem_has_frames)

n_bgrem = fusion_manifest['bgrem_frames_exist'].sum()
print(f"   RGB_BGREM available: {n_bgrem}/{len(fusion_manifest)} samples")

# Save fusion manifest
fusion_manifest_path = MANIFEST_DIR / "fusion_paired_manifest.csv"
fusion_manifest.to_csv(fusion_manifest_path, index=False)
print(f"✅ Fusion manifest saved: {fusion_manifest_path}")
print(f"   Total valid fusion samples (rgb+thermal): {len(fusion_manifest)}")
print(f"   Total valid fusion samples (bgrem+thermal): {n_bgrem}")

# ================================================================
#  STEP 3: DISTANCES
# ================================================================

DISTANCES = ['4_feet', '6_feet', '8_feet']

# ================================================================
#  STEP 4: SUBJECT-LEVEL TRAIN/VAL SPLIT
# ================================================================

def split_subject(df, val_ratio=0.15, seed=42):
    """Split by unique base_subject_id, 85/15 train/val."""
    np.random.seed(seed)
    subjects = df['base_subject_id'].unique().copy()
    np.random.shuffle(subjects)
    n_val = max(1, int(len(subjects) * val_ratio))
    val_subj   = set(subjects[:n_val])
    train_subj = set(subjects[n_val:])
    train_df = df[df['base_subject_id'].isin(train_subj)].reset_index(drop=True)
    val_df   = df[df['base_subject_id'].isin(val_subj)].reset_index(drop=True)
    return train_df, val_df

# ================================================================
#  STEP 5: 21 PROTOCOL DEFINITIONS
# ================================================================

TRAIN_COMBINATIONS = [
    ['4_feet'],
    ['6_feet'],
    ['8_feet'],
    ['4_feet', '6_feet'],
    ['4_feet', '8_feet'],
    ['6_feet', '8_feet'],
    ['4_feet', '6_feet', '8_feet'],
]

TEST_DISTANCES = ['4_feet', '6_feet', '8_feet']

# ================================================================
#  STEP 6: FUSION TYPES TO GENERATE SPLITS FOR
#
#  Each fusion type selects a different "primary" (non-thermal)
#  modality column.  We only keep rows where both modalities exist.
# ================================================================

FUSION_TYPES = {
    "rgb_thermal": {
        "primary_col":   "rgb_frame_dir",
        "secondary_col": "thermal_frame_dir",
        "filter_col":    None,   # all rows already have RGB
    },
    "rgb_bgrem_thermal": {
        "primary_col":   "rgb_bgrem_frame_dir",
        "secondary_col": "thermal_frame_dir",
        "filter_col":    "bgrem_frames_exist",  # only rows where bgrem exists
    },
}

# ================================================================
#  STEP 7: GENERATE 21 × 2 = 42 PROTOCOL SPLITS
# ================================================================

print("\n🔹 Generating 42 Fusion Protocols (21 × 2 fusion types)...\n")

all_summaries = []

for fusion_type, cfg in FUSION_TYPES.items():
    print(f"\n{'='*50}")
    print(f"  Fusion type: {fusion_type.upper()}")
    print(f"{'='*50}")

    # Select valid rows for this fusion type
    if cfg["filter_col"] is not None:
        df_ftype = fusion_manifest[fusion_manifest[cfg["filter_col"]]].copy()
    else:
        df_ftype = fusion_manifest.copy()

    print(f"  Valid samples: {len(df_ftype)}")

    # Output directory for this fusion type
    type_dir = FUSION_SPLIT_DIR / fusion_type
    type_dir.mkdir(parents=True, exist_ok=True)

    proto_id = 0

    for train_combo in TRAIN_COMBINATIONS:
        for test_dist in TEST_DISTANCES:
            proto_id += 1

            # Filter by distance
            train_pool = df_ftype[df_ftype['distance'].isin(train_combo)]
            test_df    = df_ftype[df_ftype['distance'] == test_dist]

            if len(train_pool) == 0 or len(test_df) == 0:
                print(f"  [{proto_id:02d}] SKIP – empty split")
                continue

            # Subject-level train/val
            train_df, val_df = split_subject(train_pool)

            if len(train_df) == 0:
                print(f"  [{proto_id:02d}] SKIP – empty train after subject split")
                continue

            # Protocol naming
            train_name = "+".join([d.replace("_feet", "ft") for d in train_combo])
            test_name  = test_dist.replace("_feet", "ft")
            base_name  = f"{train_name}_to_{test_name}"

            # Save splits (same schema as original splitB.py for compatibility)
            train_df.to_csv(type_dir / f"{base_name}_train.csv", index=False)
            val_df.to_csv(  type_dir / f"{base_name}_val.csv",   index=False)
            test_df.to_csv( type_dir / f"{base_name}_test.csv",  index=False)

            print(f"  [{proto_id:02d}] {fusion_type} | {train_name} → {test_name} | "
                  f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

            all_summaries.append({
                "fusion_type":    fusion_type,
                "protocol":       f"{train_name}->{test_name}",
                "train_combo":    train_name,
                "test_distance":  test_dist,
                "train_size":     len(train_df),
                "val_size":       len(val_df),
                "test_size":      len(test_df),
                "primary_col":    cfg["primary_col"],
                "secondary_col":  cfg["secondary_col"],
            })

# ================================================================
#  STEP 8: SAVE SUMMARY
# ================================================================

summary_df = pd.DataFrame(all_summaries)
summary_path = FUSION_SPLIT_DIR / "fusion_protocol_summary.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\n All fusion protocols generated!")
print(f"   Total splits: {len(all_summaries)}")
print(f"   Summary saved: {summary_path}")
print(f"\n   Run fusion_modelB.py next.")
