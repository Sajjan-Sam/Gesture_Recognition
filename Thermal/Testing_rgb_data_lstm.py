import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import timm

# ===============================
# CONFIG
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "per_models_all_models"
DATA_ROOT = "RGB_split"
OUTPUT_CSV = "final_results.csv"

IMG_SIZE = 224
SEQ_LEN = 10
BATCH_SIZE = 8
NUM_CLASSES = 7

# ===============================
# TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ===============================
# FIXED DATASET (IMPORTANT FIX)
# ===============================
class SequenceDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root, cls)

            for seq in os.listdir(cls_path):
                seq_path = os.path.join(cls_path, seq)

                #  HANDLE SUBFOLDER (PAIR_xxx)
                for sub in os.listdir(seq_path):
                    sub_path = os.path.join(seq_path, sub)

                    if os.path.isdir(sub_path):
                        self.samples.append((sub_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_path, label = self.samples[idx]

        frames = sorted([
            f for f in os.listdir(seq_path)
            if f.endswith((".jpg", ".png"))
        ])[:SEQ_LEN]

        imgs = []
        for f in frames:
            img_path = os.path.join(seq_path, f)

            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            imgs.append(img)

        # padding if less frames
        while len(imgs) < SEQ_LEN:
            imgs.append(imgs[-1])

        return torch.stack(imgs), label

# ===============================
# MODEL (MATCH TRAINING)
# ===============================
class CNN_LSTM(nn.Module):
    def __init__(self, backbone, lstm_layers):
        super().__init__()

        self.cnn = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.feature_dim = self.cnn.num_features

        self.lstm = nn.LSTM(self.feature_dim, 256, lstm_layers, batch_first=True)
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        feat = self.cnn(x)
        feat = feat.view(B, T, -1)

        out, _ = self.lstm(feat)
        return self.fc(out[:, -1, :])

# ===============================
# EVALUATION
# ===============================
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            pred = model(x).argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total

# ===============================
# MODELS TO TEST
# ===============================
models_to_test = [
    "convnext_tiny_4_feet_8_feet_lstm1.pth",
    "vit_base_patch16_224_4_feet_6_feet_8_feet_lstm1.pth"
]

# ===============================
# MAIN LOOP
# ===============================
results = []

for model_file in models_to_test:

    print(f"\n Testing {model_file}")

    backbone = "convnext_tiny" if "convnext" in model_file else "vit_base_patch16_224"

    model = CNN_LSTM(backbone, lstm_layers=1).to(DEVICE)

    state_dict = torch.load(os.path.join(MODEL_DIR, model_file), map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)

    for dist in ["4_feet", "6_feet", "8_feet"]:

        test_path = os.path.join(DATA_ROOT, dist, "test")

        dataset = SequenceDataset(test_path)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        acc = evaluate(model, loader)

        print(f"{dist} → {acc:.4f}")

        results.append({
            "Model": model_file,
            "Distance": dist,
            "Accuracy": acc
        })

# ===============================
# SAVE RESULTS
# ===============================
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print("\n DONE! Results saved to:", OUTPUT_CSV)
