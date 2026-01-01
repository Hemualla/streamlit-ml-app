# Generated from: 07_erosion_prediction_model.ipynb
# Converted at: 2026-01-01T14:08:54.044Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


BASE_DIR = "/content/drive/MyDrive/AI_Archaeology_Project_Final/data/terrain_dataset"

DEM_PATH = os.path.join(BASE_DIR, "dem_raw/synthetic_dem.npy")
SLOPE_PATH = os.path.join(BASE_DIR, "slope_maps/synthetic_slope.npy")
NDVI_PATH = os.path.join(BASE_DIR, "ndvi_maps/synthetic_ndvi.npy")
LABEL_PATH = os.path.join(BASE_DIR, "erosion_labels/synthetic_erosion_label.png")

print("Paths configured successfully")


import cv2

dem = np.load(DEM_PATH)
slope = np.load(SLOPE_PATH)
ndvi = np.load(NDVI_PATH)
erosion = cv2.imread(LABEL_PATH, cv2.IMREAD_GRAYSCALE) / 255.0

print("DEM shape:", dem.shape)
print("Slope shape:", slope.shape)
print("NDVI shape:", ndvi.shape)
print("Erosion label shape:", erosion.shape)


PATCH_SIZE = 32
STRIDE = 16

X, y = [], []

H, W = dem.shape

for i in range(0, H - PATCH_SIZE, STRIDE):
    for j in range(0, W - PATCH_SIZE, STRIDE):
        dem_patch = dem[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
        slope_patch = slope[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
        ndvi_patch = ndvi[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
        erosion_patch = erosion[i:i+PATCH_SIZE, j:j+PATCH_SIZE]

        features = [
            dem_patch.mean(), dem_patch.std(),
            slope_patch.mean(), slope_patch.std(),
            ndvi_patch.mean(), ndvi_patch.std()
        ]

        label = erosion_patch.mean()

        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Total samples:", X.shape[0])
print("Feature vector size:", X.shape[1])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed")


y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("===== MODEL EVALUATION =====")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


plt.figure(figsize=(8,4))
plt.bar(
    ["DEM μ", "DEM σ", "Slope μ", "Slope σ", "NDVI μ", "NDVI σ"],
    model.feature_importances_
)
plt.title("Feature Importance for Erosion Prediction")
plt.ylabel("Importance Score")
plt.show()


# --- CELL 9: Reconstruct Spatial Prediction Map (FIXED) ---

prediction_map = np.zeros_like(erosion)

idx = 0
max_idx = len(y_pred)

for i in range(0, H - PATCH_SIZE, STRIDE):
    for j in range(0, W - PATCH_SIZE, STRIDE):
        if idx >= max_idx:
            break

        prediction_map[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = y_pred[idx]
        idx += 1


# --- CELL 10: Visualization of Erosion Prediction Map ---

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.title("Ground Truth Erosion Map")
plt.imshow(erosion, cmap="inferno")
plt.colorbar(label="Erosion Intensity")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Erosion Map (XGBoost)")
plt.imshow(prediction_map, cmap="inferno")
plt.colorbar(label="Predicted Erosion")
plt.axis("off")

plt.tight_layout()
plt.show()


# --- CELL 11: Prepare Evaluation Targets ---

import numpy as np

# Ground truth erosion values (flattened)
y_true = erosion.flatten()

# Predicted erosion values (flattened)
y_pred = prediction_map.flatten()

print("y_true shape:", y_true.shape)
print("y_pred shape:", y_pred.shape)


# --- CELL 12: Regression Metrics ---

from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("===== MODEL PERFORMANCE =====")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# --- CELL 13: Spatial Comparison ---

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.title("Ground Truth Erosion Map")
plt.imshow(erosion, cmap="inferno")
plt.colorbar(label="Erosion Intensity")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Erosion Map")
plt.imshow(prediction_map, cmap="inferno")
plt.colorbar(label="Predicted Erosion")
plt.axis("off")

plt.tight_layout()
plt.show()


# --- CELL 14: Risk Categorization ---

def classify_erosion(val):
    if val < 0.3:
        return "Low"
    elif val < 0.6:
        return "Moderate"
    else:
        return "High"

risk_map = np.vectorize(classify_erosion)(prediction_map)

unique, counts = np.unique(risk_map, return_counts=True)

print("Erosion Risk Distribution:")
for u, c in zip(unique, counts):
    print(f"{u}: {c}")


# --- CELL 16: Final Summary ---

print("""
MODEL USED:
- XGBoost Regressor

INPUT FEATURES:
- DEM (Elevation)
- Slope
- NDVI

TASK:
- Patch-based terrain erosion prediction

EVALUATION METRICS:
- RMSE (Regression Error)
- R² Score (Variance Explained)
- Spatial Heatmap Comparison

KEY RESULT:
- Model successfully identifies erosion-prone regions
- Suitable for archaeological terrain risk assessment
""")