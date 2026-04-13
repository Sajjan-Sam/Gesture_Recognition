#  Gesture Recognition   RGB + Thermal Dual-Modality Pipeline

A complete research pipeline for **multi-modality hand gesture recognition** using paired RGB and thermal video data captured at three distances (4 ft, 6 ft, 8 ft) and across day/night conditions. The pipeline covers everything from raw data auditing to fusion model training, result collection, and paper-ready figures.

**7 gesture classes:** `doctor`, `emergency`, `fire`, `help`, `robbery`, `sit_down`, `stand_up`  
**Modalities:** RGB · Thermal · RGB-Background-Removed (RGB_BGREM)  
**Models:** CNN-LSTM · CNN-GRU · GestFormerFusion · ConvNeXtTinyGRU  

---

## 📁 Repository Structure

```
Gesture_Recognition/
│
├── 📓 Notebooks (run in numerical order)
│   ├── 01_dataset_audit.ipynb
│   ├── 02_video_pairing_manifest.ipynb
│   ├── 03_frame_extraction.ipynb
│   ├── 03_frame_extraction_bgremove.ipynb
│   ├── 04_subject_split_creation.ipynb
│   ├── 05_baseline_rgb.ipynb
│   ├── 05_baseline_rgb1.ipynb
│   ├── 06_baseline_nir.ipynb
│   ├── 10_intermediate_report_result_collector.ipynb
│   ├── 10_paper_figures.ipynb
│   └── rgb_train_thermaltest.ipynb
│
├──  Python Scripts (training / evaluation / splitting)
│   ├── splitB-1.py
│   ├── fusion_splitB.py  /  fusion_splitB-1.py
│   ├── Pre_train_model.py
│   ├── all_lstm_final.py
│   ├── all_gru.py
│   ├── modelB.py
│   ├── fusion_modelB.py
│   ├── day_night_final.py
│   ├── testing_rgb.py
│   └── testing_rgb_gru1.py
│
├── 📊 Results & Logs (CSV + TXT)
│   ├── results_pre_train.csv
│   ├── results_lstm.csv
│   ├── results_gru.csv
│   ├── dn_results_intermediate.csv
│   ├── convnext_gru_fusion_master.csv
│   ├── gestformer_fusion_master.csv
│   ├── combined_all_models.csv
│   ├── all_experiment_results-1.csv
│   ├── Pre_train_model.txt
│   ├── all_lstm_final.txt
│   ├── all_gru.txt
│   ├── day_night_final.txt
│   ├── testing_rgb.txt
│   ├── testing_rgb_geu.txt
│   ├── fusion_modelB.txt
│   └── modelB.txt
│
└── 📄 Figures & Documents
    ├── rgb_thermal_model_comparison.pdf
    ├── rgb_bgrem_thermal_model_comparison.pdf
    └── sajjan2.tex
```

---

## 🔢 Step-by-Step Navigation Guide

Follow this exact sequence to reproduce the full pipeline from raw data to paper figures.

---

### PHASE 1   Data Preparation

#### `01_dataset_audit.ipynb`
**Purpose:** Inspect the raw dataset structure.  
- Counts videos/frames per gesture class, distance, and subject.  
- Flags missing or mismatched RGB ↔ Thermal pairs.  
- Run this **first** to verify your data is complete before any processing.

#### `02_video_pairing_manifest.ipynb`
**Purpose:** Build a master pairing manifest (`paired_manifest.csv`).  
- Walks `Trimmed/RGB/` and `Trimmed/thermal/` directories.  
- Creates `manifests/paired_manifest.csv` with columns: `pair_id`, `subject_id`, `gesture`, `distance`, `rgb_frame_dir`, `thermal_frame_dir`.  
- All downstream scripts depend on this manifest file.

#### `03_frame_extraction.ipynb`
**Purpose:** Extract frames from raw RGB and thermal video clips.  
- Reads video files and saves individual `.jpg` frames into `Trimmed/RGB/` and `Trimmed/thermal/` subdirectories.  
- Output folder structure: `<root>/<distance>/<gesture>/<subject>/<pair_id>/frame_XXXX.jpg`

#### `03_frame_extraction_bgremove.ipynb`
**Purpose:** Generate background-removed RGB frames (RGB_BGREM modality).  
- Takes extracted RGB frames as input.  
- Applies background subtraction/segmentation and saves results to `Trimmed/RGB_BGREM/` using the same folder structure.  
- Enables the `rgb_bgrem_thermal` fusion experiment.

#### `04_subject_split_creation.ipynb`
**Purpose:** Create subject-wise train / val / test splits.  
- Reads `paired_manifest.csv`.  
- Performs **subject-stratified splitting** (85% train subjects / 15% val subjects; test is always the target distance).  
- Generates 21 protocols (7 train-distance combos × 3 test distances).  
- Saves CSVs under `manifests/splits/`.

---

### PHASE 2   Baseline Training (Single Modality)

#### `Pre_train_model.py`
**Purpose:** Frame-level pretrained CNN baseline (no temporal modelling).  
- Uses `ImageFolder` over `thermal_split/`.  
- Backbones: ResNet-18, EfficientNet-B0, MobileNetV3-Small, ViT-B/16, ConvNeXt-Tiny (via `timm`).  
- Freezes bottom 60% of layers; trains top 40% with Adam (lr=1e-4).  
- Saves best `.pth` models to `saved_models_60/`, results to `results_60/results.csv`.  
- Log: `Pre_train_model.txt` · Results: `results_pre_train.csv`

**Run:**
```bash
python Pre_train_model.py
```

#### `05_baseline_rgb.ipynb` / `05_baseline_rgb1.ipynb`
**Purpose:** Interactive notebook versions of the RGB baseline training.  
- `05_baseline_rgb.ipynb`   full run with training curves and confusion matrices.  
- `05_baseline_rgb1.ipynb`   variant experiment (alternative hyperparameters or augmentation).  
- Both use RGB frames and test across the 21 distance protocols.

#### `06_baseline_nir.ipynb`
**Purpose:** Near-infrared / thermal-only baseline experiment (stub / early version).  
- Minimal content; reserved for NIR-only modality experiments.

---

### PHASE 3   Sequential Model Training (CNN + RNN)

#### `all_lstm_final.py`
**Purpose:** Exhaustive CNN-LSTM sweep over all model-distance-depth combinations.  
- Architecture: `CNN_LSTM`   CNN feature extractor (timm) → LSTM (1–5 layers, hidden=256) → FC.  
- Backbones: ResNet-50, EfficientNet-B0, MobileNetV3-Small, ViT-B/16, ConvNeXt-Tiny.  
- Data root: `thermal_split_new/`, Sequence length: 10 frames, Batch: 8.  
- Trains across 7 train-distance combos × 3 test distances × 5 LSTM depths = **525 runs**.  
- Saves models to `per_models_all_models/`, graphs to `per_graphs_all_models/`.  
- Log: `all_lstm_final.txt` · Results: `results_lstm.csv`

**Run:**
```bash
python all_lstm_final.py
```

#### `all_gru.py`
**Purpose:** Identical sweep as `all_lstm_final.py` but replaces LSTM with GRU.  
- Architecture: `CNN_LSTM` class (reused name) with `nn.GRU` substituted for `nn.LSTM`.  
- GRU depths: 1–5 layers; all other hyperparameters identical.  
- Outputs to `per_gru_models_all_models/`, `per_gru_graphs_all_models/`.  
- Log: `all_gru.txt` · Results: `results_gru.csv`

**Run:**
```bash
python all_gru.py
```

---

### PHASE 4   Advanced Single-Stream Models (GestFormer + ConvNeXtGRU)

#### `splitB-1.py`
**Purpose:** Generate the 21-protocol manifest and split CSVs for the advanced pipeline.  
- Re-walks `Trimmed/RGB/` to build `manifests/paired_manifest.csv`.  
- Defines 21 protocols and saves per-protocol `_train.csv`, `_val.csv`, `_test.csv` under `manifests/splits/`.  
- Must be run **before** `modelB.py`.

**Run:**
```bash
python splitB-1.py
```

#### `modelB.py`
**Purpose:** Single-stream benchmark with two advanced architectures.  
- **GestFormerFusion**   EfficientNet-B0 spatial encoder → Projection + LayerNorm → [CLS] token → Transformer encoder (CM-Diff / CFC inspired) → MLP head.  
- **ConvNeXtTinyGRU**   ConvNeXt-Tiny (60% frozen) → 1-layer GRU → FC.  
- Reads `manifests/paired_manifest.csv` and the 21 protocol splits.  
- Modalities: `rgb`, `thermal`, `rgb_bgrem` (resolved via `get_frame_dir()`).  
- Training: AdamW + warmup cosine LR schedule + AMP (FP16) + OOM retry logic.  
- Saves checkpoints to `checkpoints/`, metrics JSON + history CSV per protocol.  
- Log: `modelB.txt`

**Run:**
```bash
python modelB.py
```

#### `rgb_train_thermaltest.ipynb`
**Purpose:** Cross-modality generalisation experiment.  
- Trains a model exclusively on RGB data and evaluates on thermal test sets (and vice versa).  
- Tests whether features learned from RGB transfer to the thermal domain without retraining.

---

### PHASE 5   Dual-Stream Fusion Models

#### `fusion_splitB.py` / `fusion_splitB-1.py`
**Purpose:** Generate fusion-specific manifests and split CSVs.  
- Reads `manifests/paired_manifest.csv` and derives `rgb_bgrem_frame_dir` via `derive_bgrem_dir()`.  
- Checks existence of RGB_BGREM directories and flags `bgrem_frames_exist`.  
- Saves `manifests/fusion_paired_manifest.csv`.  
- Generates splits for two fusion types: `rgb_thermal` and `rgb_bgrem_thermal`.  
- Outputs 42 protocol splits (21 × 2) under `manifests/fusion_splits/<fusion_type>/`.  
- Must be run **before** `fusion_modelB.py`.

**Run:**
```bash
python fusion_splitB.py
```

#### `fusion_modelB.py`
**Purpose:** Full dual-stream fusion benchmark (84 total runs).  
- **GestFormerFusion**   two independent EfficientNet-B0 encoders (RGB + Thermal) → bidirectional cross-modal attention (RGB↔Thermal) → Add+LayerNorm → Transformer → logits.  
- **ConvNeXtTinyGRU**   two independent ConvNeXt-Tiny encoders (60% frozen) → concatenation on feature dim → 1-layer GRU → FC.  
- Input: 6-channel tensor `[6, T, H, W]` (channels 0–2 = primary, 3–5 = thermal).  
- Runs all 21 protocols × 2 fusion types × 2 models = 84 experiment runs.  
- Includes skip-if-done caching, OOM-retry, and per-run JSON metrics.  
- Results: `convnext_gru_fusion_master.csv`, `gestformer_fusion_master.csv`  
- Log: `fusion_modelB.txt`

**Run:**
```bash
python fusion_modelB.py
```

---

### PHASE 6   Day/Night Robustness Experiments

#### `day_night_final.py`
**Purpose:** Train and test both models under varying lighting conditions.  
- Dataset: `Day_Night_Dataset/` with `rgb/` and `thermal/` subdirectories, each split by `day/` and `night/`.  
- Single-stream (no fusion); each model takes `[B, 3, H, W]`.  
- 4 training configs: (rgb/day), (rgb/night), (thermal/day), (thermal/night).  
- Tests all 4 cross-condition combinations → 32 result rows total.  
- Saves accuracy/F1 heatmaps per model.  
- Results: `dn_results_intermediate.csv` (rolling), `dn_result_final/dn_result_final.csv`  
- Log: `day_night_final.txt`

**Run:**
```bash
python day_night_final.py
```

---

### PHASE 7   Cross-Modality Testing

#### `testing_rgb.py`
**Purpose:** Load a trained thermal CNN-LSTM model and evaluate it on RGB test data.  
- Loads specific `.pth` files from `per_models_all_models/` (ConvNeXt-Tiny, ViT-B/16).  
- Evaluates at 4 ft, 6 ft, 8 ft distances from `RGB_split/`.  
- Saves accuracy per distance to `final_results.csv`.  
- Log: `testing_rgb.txt`

**Run:**
```bash
python testing_rgb.py
```

#### `testing_rgb_gru1.py`
**Purpose:** Same cross-modality test but for the GRU model.  
- Loads `per_gru_models_all_models/convnext_tiny_4_feet_6_feet_8_feet_gru1.pth`.  
- Defines `CNN_GRU` class (ConvNeXt-Tiny + 1-layer GRU) to match training architecture.  
- Saves per-distance accuracy to `rgb_test_results.csv`.  
- Log: `testing_rgb_geu.txt`

**Run:**
```bash
python testing_rgb_gru1.py
```

---

### PHASE 8   Result Aggregation & Figures

#### `10_intermediate_report_result_collector.ipynb`
**Purpose:** Aggregate raw experiment CSVs into a unified results table.  
- Reads outputs from `Pre_train_model.py`, `all_lstm_final.py`, `all_gru.py`, `modelB.py`.  
- Produces `all_experiment_results-1.csv`   master comparison table.  
- Generates preliminary bar charts and heatmaps for the intermediate report.

#### `10_paper_figures.ipynb`
**Purpose:** Produce all publication-quality figures.  
- Uses `combined_all_models.csv`, `convnext_gru_fusion_master.csv`, `gestformer_fusion_master.csv`.  
- Generates accuracy heatmaps, cross-distance generalisation plots, fusion vs. single-stream comparisons.  
- Outputs figures directly embedded in `sajjan2.tex`.

---

## 📊 Key Results Files

| File | Contents |
|------|----------|
| `results_pre_train.csv` | Frame-level pretrained CNN baseline results |
| `results_lstm.csv` | CNN-LSTM sweep (525 runs) |
| `results_gru.csv` | CNN-GRU sweep (525 runs) |
| `dn_results_intermediate.csv` | Day/night experiment rolling results |
| `convnext_gru_fusion_master.csv` | Fusion results   ConvNeXtTinyGRU |
| `gestformer_fusion_master.csv` | Fusion results   GestFormerFusion |
| `combined_all_models.csv` | All models combined for paper comparison |
| `all_experiment_results-1.csv` | Master aggregated results file |

---

## 📄 Paper

The LaTeX source for the corresponding research paper is in **`sajjan2.tex`** (compiled via Overleaf). Figures referenced in the paper are generated by `10_paper_figures.ipynb`. Visual comparisons across RGB-only, Thermal-only, and RGB+Thermal fusion conditions are summarised in:
- `rgb_thermal_model_comparison.pdf`
- `rgb_bgrem_thermal_model_comparison.pdf`

---

## ⚙️ Environment

```bash
pip install torch torchvision timm pandas matplotlib seaborn tqdm scikit-learn Pillow
```

| Requirement | Version |
|-------------|---------|
| PyTorch | ≥ 2.0 |
| timm | ≥ 0.9 |
| Python | ≥ 3.9 |
| GPU | CUDA-enabled (RTX A4000 recommended) |

---

## 🗺️ Recommended Execution Order

```
01_dataset_audit.ipynb
        ↓
02_video_pairing_manifest.ipynb
        ↓
03_frame_extraction.ipynb  →  03_frame_extraction_bgremove.ipynb
        ↓
04_subject_split_creation.ipynb
        ↓
splitB-1.py  →  fusion_splitB.py
        ↓
Pre_train_model.py          (frame-level baseline)
        ↓
all_lstm_final.py            (CNN-LSTM sweep)
all_gru.py                   (CNN-GRU sweep)
        ↓
modelB.py                    (GestFormer + ConvNeXtGRU, single-stream)
        ↓
fusion_modelB.py             (dual-stream fusion)
        ↓
day_night_final.py           (robustness: day/night)
testing_rgb.py               (cross-modality: LSTM)
testing_rgb_gru1.py          (cross-modality: GRU)
        ↓
10_intermediate_report_result_collector.ipynb
        ↓
10_paper_figures.ipynb
```
