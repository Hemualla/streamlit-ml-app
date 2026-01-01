# Generated from: 05_segmentation_model.ipynb
# Converted at: 2026-01-01T14:08:18.098Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


IMG_SIZE = 128
NUM_SAMPLES = 120

images = []
masks = []

for _ in range(NUM_SAMPLES):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # vegetation (class 1)
    for _ in range(5):
        x, y = np.random.randint(0, 100, 2)
        cv2.circle(img, (x, y), 20, (0, 180, 0), -1)
        cv2.circle(mask, (x, y), 20, 1, -1)

    # ruins (class 2)
    for _ in range(3):
        x, y = np.random.randint(0, 90, 2)
        cv2.rectangle(img, (x, y), (x+30, y+30), (150,150,150), -1)
        cv2.rectangle(mask, (x, y), (x+30, y+30), 2, -1)

    images.append(img)
    masks.append(mask)

images = np.array(images) / 255.0
masks = np.array(masks)

print("Images:", images.shape)
print("Masks :", masks.shape)


X_train, X_val, y_train, y_val = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)


class SegDataset(Dataset):
    def __init__(self, imgs, masks):
        self.imgs = imgs
        self.masks = masks

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx]).permute(2,0,1).float()
        mask = torch.tensor(self.masks[idx]).long()
        return img, mask

train_loader = DataLoader(SegDataset(X_train, y_train), batch_size=8, shuffle=True)
val_loader   = DataLoader(SegDataset(X_val, y_val), batch_size=8, shuffle=False)

print("Dataloaders ready")


# =====================================
# VISUALIZE SAMPLE DATASET IMAGES
# =====================================

import matplotlib.pyplot as plt

# Get one batch from training loader
images, masks = next(iter(train_loader))

# Move to CPU
images = images.cpu().numpy()
masks = masks.cpu().numpy()

# Plot 3 samples
num_samples = min(3, images.shape[0])

plt.figure(figsize=(12, 4 * num_samples))

for i in range(num_samples):
    # Input image
    plt.subplot(num_samples, 3, i*3 + 1)
    plt.imshow(images[i].transpose(1,2,0))
    plt.title("Input Image")
    plt.axis("off")

    # Ground truth mask
    plt.subplot(num_samples, 3, i*3 + 2)
    plt.imshow(masks[i], cmap="jet")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # Overlay
    plt.subplot(num_samples, 3, i*3 + 3)
    plt.imshow(images[i].transpose(1,2,0))
    plt.imshow(masks[i], cmap="jet", alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")

plt.tight_layout()
plt.show()


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()

        self.d1 = DoubleConv(3, 32)
        self.d2 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c2 = DoubleConv(128, 64)

        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.c1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        b  = self.bottleneck(self.pool(d2))

        u2 = self.u2(b)
        c2 = self.c2(torch.cat([u2, d2], dim=1))

        u1 = self.u1(c2)
        c1 = self.c1(torch.cat([u1, d1], dim=1))

        return self.out(c1)

model = UNet(n_classes=3).to(device)
print(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def compute_iou(pred, true):
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return intersection / (union + 1e-6)

def compute_dice(pred, true):
    intersection = (pred * true).sum()
    return (2. * intersection) / (pred.sum() + true.sum() + 1e-6)


EPOCHS = 8

for epoch in range(EPOCHS):
    model.train()
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            if epoch == EPOCHS - 1:
                val_img_sample = imgs[0].cpu().permute(1,2,0).numpy()
                true_mask = masks[0].cpu().numpy()
                pred_mask = preds[0].cpu().numpy()

    print(f"Epoch {epoch+1}/{EPOCHS} completed")


iou = compute_iou(pred_mask > 0, true_mask > 0)
dice = compute_dice(pred_mask > 0, true_mask > 0)

print("===== SEGMENTATION METRICS =====")
print(f"IoU  Score : {iou:.4f}")
print(f"Dice Score : {dice:.4f}")


plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(val_img_sample)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(true_mask, cmap="jet")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(val_img_sample)
plt.imshow(pred_mask, cmap="jet", alpha=0.5)
plt.title("Predicted Mask Overlay")
plt.axis("off")

plt.show()


from google.colab import files

uploaded = files.upload()
img_path = list(uploaded.keys())[0]

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128,128)) / 255.0

tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device)

with torch.no_grad():
    pred = model(tensor)
    pred_mask_test = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Uploaded Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(pred_mask_test, cmap="jet")
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img)
plt.imshow(pred_mask_test, cmap="jet", alpha=0.5)
plt.title("Overlay")
plt.axis("off")

plt.show()


# ================================
# SEGMENTATION EVALUATION METRICS
# ================================

import numpy as np
import matplotlib.pyplot as plt

def compute_iou(pred, target, num_classes=3):
    ious = []
    for cls in range(num_classes):
        pred_c = (pred == cls)
        target_c = (target == cls)

        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

def compute_dice(pred, target, num_classes=3):
    dices = []
    for cls in range(num_classes):
        pred_c = (pred == cls)
        target_c = (target == cls)

        intersection = (pred_c & target_c).sum()
        denom = pred_c.sum() + target_c.sum()

        if denom == 0:
            dices.append(np.nan)
        else:
            dices.append(2 * intersection / denom)
    return np.nanmean(dices)


# -------------------------------
# Take ONE validation sample
# -------------------------------
model.eval()
images, masks = next(iter(val_loader))

images = images.to(device)
masks = masks.to(device)

with torch.no_grad():
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

# Move to CPU
val_img_sample = images[0].cpu().numpy().transpose(1, 2, 0)
true_mask = masks[0].cpu().numpy()
pred_mask = preds[0].cpu().numpy()

# -------------------------------
# Compute metrics
# -------------------------------
iou_score = compute_iou(pred_mask, true_mask, num_classes=3)
dice_score = compute_dice(pred_mask, true_mask, num_classes=3)

print("===== SEGMENTATION METRICS =====")
print(f"Mean IoU  : {iou_score:.4f}")
print(f"Dice Score: {dice_score:.4f}")

# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(val_img_sample)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(true_mask, cmap="jet")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(val_img_sample)
plt.imshow(pred_mask, cmap="jet", alpha=0.5)
plt.title("Predicted Mask Overlay")
plt.axis("off")

plt.show()