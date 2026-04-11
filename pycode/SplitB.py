

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ================================================================
#  CONFIG
# ================================================================

TRIMMED_ROOT = Path("Trimmed")

RGB_ROOT     = TRIMMED_ROOT / "RGB"
THERMAL_ROOT = TRIMMED_ROOT / "thermal"

MANIFEST_DIR = Path("manifests")
SPLIT_DIR    = MANIFEST_DIR / "splits"

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

# ================================================================
#  UTIL FUNCTIONS
# ================================================================

def count_frames(folder):
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)

def base_subject(name):
    m = re.match(r'^.*?_?(S\d+)$', name, re.IGNORECASE)
    return m.group(1).upper() if m else name

# ================================================================
#  STEP 1: BUILD MANIFEST
# ================================================================

print("\n🔹 Building Manifest...")

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

                rgb_n = count_frames(pair_dir)
                thermal_n = count_frames(thermal_dir)

                if rgb_n == 0 or thermal_n == 0:
                    continue

                rows.append({
                    "pair_id": pair_dir.name,
                    "subject_id": subject,
                    "base_subject_id": base_subject(subject),
                    "gesture": gesture,
                    "distance": distance,
                    "rgb_frame_dir": str(pair_dir),
                    "thermal_frame_dir": str(thermal_dir),
                    "rgb_n_frames": rgb_n,
                    "thermal_n_frames": thermal_n
                })

manifest = pd.DataFrame(rows)
manifest_path = MANIFEST_DIR / "paired_manifest.csv"
manifest.to_csv(manifest_path, index=False)

print(f" Manifest saved: {manifest_path}")
print(f"Total valid samples: {len(manifest)}")

# ================================================================
#  STEP 2: DEFINE DISTANCES
# ================================================================

DISTANCES = ['4_feet', '6_feet', '8_feet']

# ================================================================
#  STEP 3: TRAIN-VAL SPLIT BY SUBJECT
# ================================================================

def split_subject(df, val_ratio=0.15, seed=42):
    np.random.seed(seed)

    subjects = df['base_subject_id'].unique()
    np.random.shuffle(subjects)

    n_val = max(1, int(len(subjects) * val_ratio))

    val_subj = set(subjects[:n_val])
    train_subj = set(subjects[n_val:])

    train_df = df[df['base_subject_id'].isin(train_subj)]
    val_df   = df[df['base_subject_id'].isin(val_subj)]

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

# ================================================================
#  STEP 4: DEFINE 21 PROTOCOLS
# ================================================================

TRAIN_COMBINATIONS = [
    ['4_feet'],
    ['6_feet'],
    ['8_feet'],
    ['4_feet','6_feet'],
    ['4_feet','8_feet'],
    ['6_feet','8_feet'],
    ['4_feet','6_feet','8_feet']
]

TEST_DISTANCES = ['4_feet','6_feet','8_feet']

# ================================================================
#  STEP 5: GENERATE SPLITS
# ================================================================

print("\n🔹 Generating 21 Protocols...\n")

protocol_id = 0
summary = []

for train_combo in TRAIN_COMBINATIONS:
    for test_dist in TEST_DISTANCES:

        protocol_id += 1

        # Filter data
        train_pool = manifest[manifest['distance'].isin(train_combo)]
        test_df    = manifest[manifest['distance'] == test_dist]

        # Split train → train + val
        train_df, val_df = split_subject(train_pool)

        # Naming
        train_name = "+".join([d.replace("_feet", "ft") for d in train_combo])
        test_name  = test_dist.replace("_feet", "ft")

        base_name = f"{train_name}_to_{test_name}"

        # Save CSV
        train_df.to_csv(SPLIT_DIR / f"{base_name}_train.csv", index=False)
        val_df.to_csv(  SPLIT_DIR / f"{base_name}_val.csv",   index=False)
        test_df.to_csv( SPLIT_DIR / f"{base_name}_test.csv",  index=False)

        print(f"[{protocol_id:02d}] {train_name} → {test_name} | "
              f"Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}")

        summary.append({
            "Protocol": f"{train_name}->{test_name}",
            "Train_Size": len(train_df),
            "Val_Size": len(val_df),
            "Test_Size": len(test_df)
        })

# ================================================================
#  STEP 6: SAVE SUMMARY
# ================================================================

summary_df = pd.DataFrame(summary)
summary_path = SPLIT_DIR / "protocol_21_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\n✅ All 21 protocols generated successfully!")
print(f"Summary saved at: {summary_path}")
