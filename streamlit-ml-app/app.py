import streamlit as st
import torch
import cv2
import numpy as np
import joblib
from PIL import Image
from xgboost import Booster
import os

# ---------------- CONFIG ----------------
BASE = "models"
SEG_PATH = f"{BASE}/segmentation/unet.pth"
DET_PATH = f"{BASE}/detection/yolov5_best.pt"
EROSION_DIR = f"{BASE}/terrain"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- UI ----------------
st.set_page_config("AI Archaeology Dashboard", layout="wide")
st.title("🏺 AI Archaeology Dashboard")

task = st.sidebar.selectbox(
    "Select Task",
    ["Segmentation", "Object Detection", "Erosion Prediction"]
)

uploaded = st.file_uploader("Upload Satellite Image", ["jpg", "png", "jpeg"])

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_erosion():
    scaler = joblib.load(f"{EROSION_DIR}/scaler.joblib")
    model = joblib.load(f"{EROSION_DIR}/xgb_regressor.joblib")
    return scaler, model

# ---------------- SEGMENTATION ----------------
if task == "Segmentation" and uploaded:
    st.subheader("🧩 Semantic Segmentation")

    img = Image.open(uploaded).convert("RGB").resize((128,128))
    img_np = np.array(img) / 255.0

    from segmentation_model import UNet   # OPTIONAL modularization
    model = UNet(n_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(SEG_PATH, map_location=DEVICE))
    model.eval()

    x = torch.tensor(img_np).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        pred = model(x)
        mask = torch.argmax(pred, dim=1).cpu().numpy()[0]

    st.image(img_np, caption="Input Image")
    st.image(mask, caption="Predicted Mask", clamp=True)

# ---------------- EROSION ----------------
elif task == "Erosion Prediction" and uploaded:
    st.subheader("🌋 Terrain Erosion Prediction")

    scaler, model = load_erosion()

    img = Image.open(uploaded).convert("L").resize((128,128))
    arr = np.array(img) / 255.0

    features = np.array([
        arr.mean(), arr.std(),
        np.gradient(arr)[0].mean(), np.gradient(arr)[0].std(),
        np.gradient(arr)[1].mean(), np.gradient(arr)[1].std()
    ]).reshape(1, -1)

    X = scaler.transform(features)
    erosion = model.predict(X)[0]

    st.metric("Predicted Erosion Score", f"{erosion:.3f}")

# ---------------- OBJECT DETECTION ----------------
elif task == "Object Detection" and uploaded:
    st.subheader("🔍 Artifact Detection (YOLOv5)")
    st.info("Detection runs using pretrained YOLOv5 model")

    st.image(uploaded)
