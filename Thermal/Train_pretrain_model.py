import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
import timm
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
DATA_ROOT = "thermal_split"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 7
PATIENCE = 7

RESULT_DIR = "results_60"
MODEL_DIR = "saved_models_60"
CHECKPOINT_DIR = "checkpoints_60"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = True

# ===============================
# TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# ===============================
# DATA LOADER
# ===============================
def get_loader(paths, split):
    datasets = []
    for p in paths:
        datasets.append(ImageFolder(os.path.join(DATA_ROOT, p, split), transform))

    dataset = ConcatDataset(datasets)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True
    )

# ===============================
# MODEL
# ===============================
def get_model(name):
    try:
        model = timm.create_model(name, pretrained=True, num_classes=NUM_CLASSES)
    except:
        print(f" Pretrained failed for {name}")
        model = timm.create_model(name, pretrained=False, num_classes=NUM_CLASSES)
    return model

# ===============================
# FREEZE 60%
# ===============================
def freeze_layers(model, ratio=0.6):
    params = list(model.parameters())
    total = len(params)
    trainable_count = int(total * ratio)

    for i, p in enumerate(params):
        p.requires_grad = (i >= total - trainable_count)

    print(f"Trainable Params: {trainable_count}/{total} ({ratio*100:.0f}%)")

# ===============================
# EVALUATE
# ===============================
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total

# ===============================
# CHECKPOINT LOAD
# ===============================
def load_checkpoint(model, optimizer, scheduler, path):
    if os.path.exists(path):
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        print(f" Resuming from epoch {checkpoint['epoch']}")

        return checkpoint["epoch"], checkpoint["best_val"], checkpoint["history"]

    return 0, 0, {"train": [], "val": []}

# ===============================
# CHECKPOINT SAVE
# ===============================
def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val, history):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val": best_val,
        "history": history
    }, path)

# ===============================
# TRAIN
# ===============================
def train_model(model, train_loader, val_loader, save_path, checkpoint_path):

    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=3
    )

    start_epoch, best_val, history = load_checkpoint(
        model, optimizer, scheduler, checkpoint_path
    )

    patience_counter = 0

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        correct, total = 0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        val_acc = evaluate(model, val_loader)

        history["train"].append(train_acc)
        history["val"].append(val_acc)

        print(f"Epoch {epoch}: Train={train_acc:.4f}, Val={val_acc:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        save_checkpoint(
            checkpoint_path, model, optimizer, scheduler,
            epoch, best_val, history
        )

        if patience_counter >= PATIENCE:
            print(" Early stopping")
            break

    return best_val, history

# ===============================
# EXPERIMENT
# ===============================
models = [
    "resnet18",
    "efficientnet_b0",
    "mobilenetv3_small_100",
    "vit_base_patch16_224",
    "convnext_tiny"
]

train_sets = [
    ["4_feet"],
    ["6_feet"],
    ["8_feet"],
    ["4_feet", "6_feet"],
    ["4_feet", "8_feet"],
    ["6_feet", "8_feet"],
    ["4_feet", "6_feet", "8_feet"]
]

test_sets = ["4_feet", "6_feet", "8_feet"]

results = []

# ===============================
# MAIN LOOP
# ===============================
for model_name in models:
    for train_combo in train_sets:

        print(f"\n {model_name} | Train: {train_combo}")

        train_loader = get_loader(train_combo, "train")
        val_loader = get_loader(train_combo, "val")

        model = get_model(model_name)
        freeze_layers(model, 0.6)

        name = f"{model_name}_{'_'.join(train_combo)}"

        save_path = os.path.join(MODEL_DIR, name + ".pth")
        checkpoint_path = os.path.join(CHECKPOINT_DIR, name + ".pth")

        val_acc, history = train_model(
            model, train_loader, val_loader,
            save_path, checkpoint_path
        )

        model.load_state_dict(torch.load(save_path))

        # Save graph
        plt.figure()
        plt.plot(history["train"], label="Train")
        plt.plot(history["val"], label="Val")
        plt.legend()
        plt.title(name)
        plt.savefig(os.path.join(RESULT_DIR, name + ".png"))
        plt.close()

        print("\n TEST RESULTS:")
        print("-" * 40)

        for test_set in test_sets:
            test_loader = get_loader([test_set], "test")
            test_acc = evaluate(model, test_loader)

            print(f" Test on {test_set}: {test_acc:.4f}")

            results.append({
                "Model": model_name,
                "Train_Set": "+".join(train_combo),
                "Test_Set": test_set,
                "Val_Acc": val_acc,
                "Test_Acc": test_acc
            })

        print("-" * 40)

# ===============================
# SAVE RESULTS
# ===============================
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULT_DIR, "results.csv"), index=False)

# ===============================
# FINAL GRAPH
# ===============================
import seaborn as sns

plt.figure(figsize=(12,6))
sns.barplot(data=df, x="Train_Set", y="Test_Acc", hue="Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "comparison.png"))

print("\n DONE")
