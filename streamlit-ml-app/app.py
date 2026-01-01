%%writefile app.py
import streamlit as st

st.set_page_config(page_title="ML Models Dashboard", layout="wide")

st.title("🌍 ML Models Dashboard")

st.write("Choose a model from the sidebar")

model = st.sidebar.selectbox(
    "Select Model",
    ["Segmentation", "Object Detection", "Erosion Prediction"]
)

uploaded_file = st.file_uploader(
    "Upload Image", type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image")

    if model == "Segmentation":
        st.success("Segmentation model will run here")

    elif model == "Object Detection":
        st.success("Object Detection model will run here")

    elif model == "Erosion Prediction":
        st.success("Erosion prediction model will run here")
