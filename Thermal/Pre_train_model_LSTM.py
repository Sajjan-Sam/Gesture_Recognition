# ==========================================================
# MULTI-MODEL + LSTM (FULL PERMUTATION + SAVING SYSTEM)
# ==========================================================

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
import timm
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# ===============================
# CONFIG
# ===============================
DATA_ROOT = "thermal_split_new"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 100
NUM_CLASSES = 7
PATIENCE = 7
SEQ_LEN = 10

RESULT_DIR = "results_all_models"
MODEL_DIR = "models_all_models"
GRAPH_DIR = "graphs_all_models"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# ===============================
# TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# ===============================
# FREEZE 60%
# ===============================
def freeze_60(model):
    params = list(model.parameters())
    total = len(params)
    trainable = int(total * 0.6)

    for i, p in enumerate(params):
        p.requires_grad = (i >= total - trainable)

# ===============================
# DATASET
# ===============================
class SequenceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        if not os.path.exists(root_dir):
            print(f"[WARNING] Missing: {root_dir}")
            return

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c:i for i,c in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)

            if not os.path.isdir(cls_path):
                continue

            for seq in os.listdir(cls_path):
                seq_path = os.path.join(cls_path, seq)

                if not os.path.isdir(seq_path):
                    continue

                for pair in os.listdir(seq_path):
                    pair_path = os.path.join(seq_path, pair)

                    if os.path.isdir(pair_path):
                        self.samples.append((pair_path, cls))

        print(f"[INFO] {root_dir} → {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pair_path, cls = self.samples[idx]

        imgs = sorted(os.listdir(pair_path))[:SEQ_LEN]

        frames = []
        for img in imgs:
            img_path = os.path.join(pair_path, img)
            img = plt.imread(img_path)
            img = transform(img)
            frames.append(img)

        while len(frames) < SEQ_LEN:
            frames.append(frames[-1])

        return torch.stack(frames), self.class_to_idx[cls]

# ===============================
# LOADER
# ===============================
def get_loader(paths, split):
    datasets = []

    for p in paths:
        path = os.path.join(DATA_ROOT, p, split)
        dataset = SequenceDataset(path)

        if len(dataset) > 0:
            datasets.append(dataset)
        else:
            print(f"[WARNING] Empty skipped: {path}")

    if len(datasets) == 0:
        print(f"[CRITICAL] No data for {paths} {split}")
        dummy_x = torch.zeros(1, SEQ_LEN, 3, 224, 224)
        dummy_y = torch.zeros(1, dtype=torch.long)
        return DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=1)

    return DataLoader(
        ConcatDataset(datasets),
        batch_size=BATCH_SIZE,
        shuffle=(split=="train")
    )

# ===============================
# MODEL (GENERIC)
# ===============================
class CNN_LSTM(nn.Module):
    def __init__(self, backbone, lstm_layers):
        super().__init__()

        self.cnn = timm.create_model(backbone, pretrained=True)
        freeze_60(self.cnn)

        # ===== Feature extraction handling =====
        if "vit" in backbone:
            self.feat_dim = self.cnn.num_features
            self.feature_type = "vit"
        elif "convnext" in backbone:
            self.feat_dim = self.cnn.num_features
            self.feature_type = "convnext"
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.cnn.head = nn.Identity()
        else:
            self.feature_type = "cnn"
            if hasattr(self.cnn, "fc"):
                self.feat_dim = self.cnn.fc.in_features
                self.cnn.fc = nn.Identity()
            else:
                self.feat_dim = self.cnn.classifier.in_features
                self.cnn.classifier = nn.Identity()

        # ===== LSTM =====
        self.lstm = nn.LSTM(
            self.feat_dim,
            256,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.reshape(B*T, C, H, W)

        if self.feature_type == "convnext":
            feat = self.cnn.forward_features(x)
            feat = self.pool(feat)
            feat = feat.flatten(1)

        elif self.feature_type == "vit":
            feat = self.cnn.forward_features(x)
            feat = feat[:,0]  # CLS token

        else:
            feat = self.cnn(x)
            if len(feat.shape) == 4:
                feat = torch.mean(feat, dim=[2,3])

        feat = feat.reshape(B, T, self.feat_dim)

        out,_ = self.lstm(feat)

        return self.fc(out[:, -1, :])

# ===============================
# TRAIN / EVAL
# ===============================
def evaluate(model, loader):
    model.eval()
    correct,total = 0,0

    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)

    return correct/total if total > 0 else 0

def train_model(model, train_loader, val_loader):

    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    best_val = 0
    patience = 0

    history = {"train": [], "val": []}

    for epoch in range(EPOCHS):
        model.train()
        correct,total = 0,0

        for x,y in tqdm(train_loader, leave=False):
            x,y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out,y)
            loss.backward()
            optimizer.step()

            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)

        train_acc = correct/total if total>0 else 0
        val_acc = evaluate(model, val_loader)

        history["train"].append(train_acc)
        history["val"].append(val_acc)

        print(f"Epoch {epoch}: Train={train_acc:.4f} | Val={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            patience = 0
        else:
            patience += 1

        if patience >= PATIENCE:
            print("Early stopping")
            break

    return best_val, history

# ===============================
# PERMUTATIONS
# ===============================
models = [
    "resnet50",
    "efficientnet_b0",
    "mobilenetv3_small_100",
    "vit_base_patch16_224",
    "convnext_tiny"
]

train_sets = [
    ["4_feet"], ["6_feet"], ["8_feet"],
    ["4_feet","6_feet"], ["4_feet","8_feet"],
    ["6_feet","8_feet"], ["4_feet","6_feet","8_feet"]
]

test_sets = ["4_feet","6_feet","8_feet"]
lstm_layers_list = [1,2,3,4,5]

results = []

# ===============================
# MAIN LOOP
# ===============================
for model_name in models:
    for train_combo in train_sets:
        for lstm_layers in lstm_layers_list:

            name = f"{model_name}_{'_'.join(train_combo)}_lstm{lstm_layers}"

            print(f"\nModel={model_name} | Train={train_combo} | LSTM={lstm_layers}")

            train_loader = get_loader(train_combo, "train")
            val_loader = get_loader(train_combo, "val")

            model = CNN_LSTM(model_name, lstm_layers)

            val_acc, history = train_model(model, train_loader, val_loader)

            # SAVE MODEL
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, f"{name}.pth"))

            # SAVE GRAPH
            plt.figure()
            plt.plot(history["train"], label="Train")
            plt.plot(history["val"], label="Val")
            plt.legend()
            plt.title(name)
            plt.savefig(os.path.join(GRAPH_DIR, f"{name}.png"))
            plt.close()

            # TEST
            for test_set in test_sets:
                test_loader = get_loader([test_set], "test")
                acc = evaluate(model, test_loader)

                results.append({
                    "Model": model_name,
                    "LSTM_Layers": lstm_layers,
                    "Train_Set": "+".join(train_combo),
                    "Test_Set": test_set,
                    "Test_Acc": acc
                })

# ===============================
# SAVE RESULTS
# ===============================
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULT_DIR, "results.csv"), index=False)

plt.figure(figsize=(14,6))
sns.barplot(data=df, x="Train_Set", y="Test_Acc", hue="Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "final_comparison.png"))

print("\n DONE")
