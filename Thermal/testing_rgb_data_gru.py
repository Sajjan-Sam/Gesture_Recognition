import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import timm
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "per_gru_models_all_models/convnext_tiny_4_feet_6_feet_8_feet_gru1.pth"

DATA_ROOT = "RGB_split"   #  CHANGE: RGB instead of thermal
SEQ_LEN = 10
NUM_CLASSES = 7

SAVE_RESULT = "rgb_test_results.csv"

TEST_SETS = ["4_feet", "6_feet", "8_feet"]

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# =========================
# DATASET (SAME AS TRAINING)
# =========================
class SequenceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c:i for i,c in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)

            for seq in os.listdir(cls_path):
                seq_path = os.path.join(cls_path, seq)

                for pair in os.listdir(seq_path):
                    pair_path = os.path.join(seq_path, pair)

                    if os.path.isdir(pair_path):
                        self.samples.append((pair_path, cls))

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

# =========================
# MODEL (MATCH TRAINING)
# =========================
class CNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = timm.create_model("convnext_tiny", pretrained=False)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.cnn.head = nn.Identity()

        self.feat_dim = self.cnn.num_features

        self.gru = nn.GRU(self.feat_dim, 256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.reshape(B*T, C, H, W)

        feat = self.cnn.forward_features(x)
        feat = self.pool(feat)
        feat = feat.flatten(1)

        feat = feat.reshape(B, T, self.feat_dim)

        out,_ = self.gru(feat)
        return self.fc(out[:, -1, :])

# =========================
# LOAD MODEL
# =========================
model = CNN_GRU().to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt, strict=False)

model.eval()
print(" Model loaded")

# =========================
# TEST FUNCTION
# =========================
def test_loader(loader):
    correct,total = 0,0

    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            pred = out.argmax(1)

            correct += (pred==y).sum().item()
            total += y.size(0)

    return correct/total

# =========================
# RUN TEST
# =========================
results = []

for test_set in TEST_SETS:
    path = os.path.join(DATA_ROOT, test_set, "test")

    dataset = SequenceDataset(path)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    acc = test_loader(loader)

    print(f"{test_set} → {acc:.4f}")

    results.append({
        "Test_Set": test_set,
        "Accuracy": acc
    })

# =========================
# SAVE
# =========================
df = pd.DataFrame(results)
df.to_csv(SAVE_RESULT, index=False)

print("\n Results saved:", SAVE_RESULT)
