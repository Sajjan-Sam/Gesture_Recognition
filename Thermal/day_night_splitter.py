"""
Day_Night_Dataset Splitter  —  Person-wise Train/Val/Test Split
===============================================================

Day / Night rules (same as before):
  4_feet : S001–S015  → day   |  S016+     → night
  6_feet : S034–S035  → day   |  S001–S033 → night
  8_feet : ALL        → day

Person-wise split (70 % train | 20 % test | 10 % val):
  • Subjects are shuffled with a fixed seed (SEED = 42) for reproducibility.
  • ALL pair-folders belonging to one subject stay together in the same split.
  • Ratios are computed on the subject list, not on individual pairs/frames.

Output structure  (mirrors thermal_split exactly):
  Day_Night_Dataset/
  ├── rgb/
  │   ├── day/
  │   │   ├── train/
  │   │   │   ├── doctor/
  │   │   │   │   ├── 4F_S001_PAIR_00001/   ← original pair folder copied wholesale
  │   │   │   │   ├── 4F_S002_PAIR_00003/
  │   │   │   │   └── ...
  │   │   │   ├── emergency/
  │   │   │   └── ... (7 classes)
  │   │   ├── test/
  │   │   │   └── ... (same 7 classes)
  │   │   └── val/
  │   │       └── ... (same 7 classes)
  │   └── night/
  │       ├── train/ test/ val/  (same 7 classes each)
  └── thermal/
      ├── day/
      │   └── train/ test/ val/  (same 7 classes each)
      └── night/
          └── train/ test/ val/  (same 7 classes each)

Notes:
  • Pair folders are COPIED (original Trimmed tree is untouched).
  • The pair folder name is prefixed with the distance abbreviation to avoid
    collisions when pairs from different distances land in the same class folder:
      4ft_4F_S001_PAIR_00001/
  • Re-running is safe: existing destinations are skipped.
"""

import os
import re
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────

BASE_DIR   = Path("Trimmed")
OUTPUT_DIR = Path("Day_Night_Dataset")

MODALITIES = {
    "rgb":     "RGB_BGREM",
    "thermal": "thermal",
}

DISTANCES = ["4_feet", "6_feet", "8_feet"]

CLASSES = [
    "doctor", "emergency", "fire", "help",
    "robbery", "sit_down", "stand_up",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Person-wise split ratios
TRAIN_RATIO = 0.70
TEST_RATIO  = 0.20
VAL_RATIO   = 0.10   # remainder after train + test

SEED = 42   # fixed seed → reproducible splits


# ─────────────────────────────────────────────────────────────────
#  RULE: day or night
# ─────────────────────────────────────────────────────────────────

def get_day_night(distance: str, subject_num: int) -> str:
    """Return 'day' or 'night'."""
    if distance == "4_feet":
        return "day" if 1 <= subject_num <= 15 else "night"
    elif distance == "6_feet":
        return "day" if subject_num in (34, 35) else "night"
    elif distance == "8_feet":
        return "day"
    else:
        raise ValueError(f"Unknown distance: {distance}")


def parse_subject_number(folder_name: str) -> int:
    """4F_S011 → 11   |   6F_S034 → 34   |   returns -1 on failure."""
    m = re.search(r"S(\d+)$", folder_name, re.IGNORECASE)
    return int(m.group(1)) if m else -1


# ─────────────────────────────────────────────────────────────────
#  PERSON-WISE SPLIT  (subject-level, not frame-level)
# ─────────────────────────────────────────────────────────────────

def person_wise_split(subject_list: list) -> dict:
    """
    Given a sorted list of subject folder names (strings), shuffle them with
    a fixed seed and assign each to train / val / test.

    Returns: {"train": [...], "val": [...], "test": [...]}

    Split proportions: 70 % train | 20 % test | 10 % val
    """
    subjects = sorted(set(subject_list))   # deterministic starting order
    rng = random.Random(SEED)
    rng.shuffle(subjects)

    n       = len(subjects)
    n_train = max(1, round(n * TRAIN_RATIO))
    n_test  = max(1, round(n * TEST_RATIO))
    # val gets the remainder (≥ 1 if n ≥ 3)
    n_val   = max(0, n - n_train - n_test)

    train_s = set(subjects[:n_train])
    test_s  = set(subjects[n_train : n_train + n_test])
    val_s   = set(subjects[n_train + n_test :])

    return {"train": train_s, "val": val_s, "test": test_s}


# ─────────────────────────────────────────────────────────────────
#  CREATE OUTPUT SKELETON
# ─────────────────────────────────────────────────────────────────

def create_output_structure():
    for modality in ("rgb", "thermal"):
        for day_night in ("day", "night"):
            for split in ("train", "val", "test"):
                for cls in CLASSES:
                    (OUTPUT_DIR / modality / day_night / split / cls).mkdir(
                        parents=True, exist_ok=True
                    )
    print(f"[OK] Output directory tree created under: {OUTPUT_DIR.resolve()}\n")


# ─────────────────────────────────────────────────────────────────
#  COPY ONE PAIR FOLDER
# ─────────────────────────────────────────────────────────────────

def copy_pair_folder(src_pair: Path, dest_cls_dir: Path,
                     dist_abbrev: str, subject_name: str) -> tuple:
    """
    Copy all image files inside src_pair into dest_cls_dir/<new_pair_name>/.

    The destination pair folder is renamed to:
        <dist_abbrev>_<subject_name>_<original_pair_folder_name>
    e.g.  4ft_4F_S001_PAIR_00001

    Returns (copied_count, skipped_count).
    """
    new_pair_name = f"{dist_abbrev}_{subject_name}_{src_pair.name}"
    dest_pair     = dest_cls_dir / new_pair_name
    dest_pair.mkdir(parents=True, exist_ok=True)

    copied  = 0
    skipped = 0

    for img in src_pair.iterdir():
        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
            dest_img = dest_pair / img.name
            if dest_img.exists():
                skipped += 1
            else:
                shutil.copy2(img, dest_img)
                copied += 1

    return copied, skipped


# ─────────────────────────────────────────────────────────────────
#  MAIN COPY LOGIC
# ─────────────────────────────────────────────────────────────────

def copy_all():
    total_copied  = 0
    total_skipped = 0

    # ── per-modality ─────────────────────────────────────────────────────────
    for out_modality, src_folder_name in MODALITIES.items():
        modality_src = BASE_DIR / src_folder_name

        if not modality_src.exists():
            print(f"[WARN] Source not found: {modality_src} — skipping modality.")
            continue

        print(f"\n{'='*65}")
        print(f"  Modality: {out_modality}  (source: {src_folder_name})")
        print(f"{'='*65}")

        # ── per-distance ─────────────────────────────────────────────────────
        for distance in DISTANCES:
            dist_dir    = modality_src / distance
            dist_abbrev = distance.replace("_feet", "ft")   # "4ft", "6ft", "8ft"

            if not dist_dir.exists():
                print(f"  [WARN] Not found: {dist_dir} — skipping distance.")
                continue

            print(f"\n  Distance: {distance}")

            # ── per-class ────────────────────────────────────────────────────
            for cls in CLASSES:
                cls_dir = dist_dir / cls

                if not cls_dir.exists():
                    print(f"    [WARN] Not found: {cls_dir} — skipping class.")
                    continue

                # ── Discover subject folders and collect pair folders ─────────
                # Structure inside cls_dir:
                #   cls_dir/
                #     <subject_folder>/          e.g. 4F_S001
                #       <pair_folder>/           e.g. PAIR_00001
                #         frame_001.jpg
                #         frame_002.jpg
                #         ...

                # Group pair folders by subject
                subject_to_pairs = defaultdict(list)

                for item in cls_dir.iterdir():
                    if not item.is_dir():
                        continue
                    subject_name = item.name          # e.g. "4F_S011"
                    subj_num     = parse_subject_number(subject_name)

                    if subj_num == -1:
                        print(f"      [WARN] Cannot parse subject from "
                              f"'{subject_name}' — skipping.")
                        continue

                    # Each item can itself be a subject folder containing
                    # pair sub-folders, OR it can directly be a pair folder.
                    # Detect: if it has image files directly → it IS a pair folder.
                    direct_imgs = [
                        f for f in item.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                    ]

                    if direct_imgs:
                        # item is itself a pair folder; subject = parent logic
                        # (shouldn't happen in normal structure, handle gracefully)
                        subject_to_pairs[subject_name].append(item)
                    else:
                        # item is a subject folder; look for pair sub-folders
                        for pair_item in sorted(item.iterdir()):
                            if pair_item.is_dir():
                                subject_to_pairs[subject_name].append(pair_item)

                if not subject_to_pairs:
                    print(f"    [WARN] No subject/pair folders found in {cls_dir}")
                    continue

                # ── Determine day/night per subject ──────────────────────────
                # Build two lists: day-subjects, night-subjects
                day_subjects   = []
                night_subjects = []

                for subj_name in subject_to_pairs:
                    subj_num  = parse_subject_number(subj_name)
                    day_night = get_day_night(distance, subj_num)
                    if day_night == "day":
                        day_subjects.append(subj_name)
                    else:
                        night_subjects.append(subj_name)

                # ── Person-wise train/val/test split per day/night group ──────
                splits_map = {}   # subj_name → "train" | "val" | "test"

                for group_name, group_subjects in [
                    ("day",   day_subjects),
                    ("night", night_subjects),
                ]:
                    if not group_subjects:
                        continue

                    assignment = person_wise_split(group_subjects)
                    # assignment = {"train": set, "val": set, "test": set}

                    for tv_split, subj_set in assignment.items():
                        for s in subj_set:
                            # Store as  (day_night, tv_split)
                            splits_map[s] = (group_name, tv_split)

                # ── Copy pair folders to correct destination ──────────────────
                cls_copied  = 0
                cls_skipped = 0

                for subj_name, pair_list in subject_to_pairs.items():
                    if subj_name not in splits_map:
                        continue

                    day_night, tv_split = splits_map[subj_name]

                    dest_cls_dir = (
                        OUTPUT_DIR / out_modality / day_night / tv_split / cls
                    )

                    for pair_dir in pair_list:
                        c, s = copy_pair_folder(
                            pair_dir, dest_cls_dir, dist_abbrev, subj_name
                        )
                        cls_copied  += c
                        cls_skipped += s

                total_copied  += cls_copied
                total_skipped += cls_skipped

                # ── Per-class summary ─────────────────────────────────────────
                day_counts   = {sp: 0 for sp in ("train", "val", "test")}
                night_counts = {sp: 0 for sp in ("train", "val", "test")}

                for subj_name, (dn, sp) in splits_map.items():
                    n_pairs = len(subject_to_pairs.get(subj_name, []))
                    if dn == "day":
                        day_counts[sp]   += n_pairs
                    else:
                        night_counts[sp] += n_pairs

                print(
                    f"    [{cls}]  "
                    f"day  → train={day_counts['train']} "
                    f"val={day_counts['val']} "
                    f"test={day_counts['test']}  |  "
                    f"night → train={night_counts['train']} "
                    f"val={night_counts['val']} "
                    f"test={night_counts['test']}  "
                    f"(copied={cls_copied}, skipped={cls_skipped})"
                )

    print(f"\n{'='*65}")
    print(f"  ALL DONE")
    print(f"  Total frames copied : {total_copied:,}")
    print(f"  Total frames skipped: {total_skipped:,}  (already existed)")
    print(f"  Output root         : {OUTPUT_DIR.resolve()}")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────────
#  SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────

def print_summary():
    """Print image-file counts for every leaf folder."""
    print("\n── DATASET SUMMARY (image counts) ──────────────────────────")
    for modality in ("rgb", "thermal"):
        print(f"\n  {modality.upper()}")
        for day_night in ("day", "night"):
            print(f"    {day_night.upper()}")
            for tv_split in ("train", "val", "test"):
                split_total = 0
                for cls in CLASSES:
                    leaf = OUTPUT_DIR / modality / day_night / tv_split / cls
                    if not leaf.exists():
                        continue
                    # count recursively (images inside pair sub-folders)
                    n = sum(
                        1 for _, _, files in os.walk(leaf)
                        for f in files
                        if Path(f).suffix.lower() in IMAGE_EXTS
                    )
                    split_total += n
                    print(f"      [{tv_split}/{cls}]  {n}")
                print(f"    --> {tv_split} subtotal: {split_total}")
    print()


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nDay_Night_Dataset Builder  —  Person-wise 70/20/10 Split")
    print("=" * 65)
    print(f"Source root : {BASE_DIR.resolve()}")
    print(f"Output root : {OUTPUT_DIR.resolve()}")
    print(f"Seed        : {SEED}")
    print()
    print("Day/Night rules:")
    print("  4_feet  S001–S015 → day  |  S016+     → night")
    print("  6_feet  S034–S035 → day  |  S001–S033 → night")
    print("  8_feet  ALL       → day")
    print()
    print("Person-wise split (applied separately inside each day/night group):")
    print(f"  train={int(TRAIN_RATIO*100)}%  "
          f"test={int(TEST_RATIO*100)}%  "
          f"val={int(VAL_RATIO*100)}%")
    print("=" * 65)

    create_output_structure()
    copy_all()
    print_summary()