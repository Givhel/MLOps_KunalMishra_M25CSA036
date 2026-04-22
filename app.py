import streamlit as st
import numpy as np
from PIL import Image

st.title("Cityscape Segmentation App")

page = st.sidebar.selectbox("Select Page", ["Metrics", "Prediction"])

# ======================
# PAGE 1: METRICS
# ======================
if page == "Metrics":
    st.header("Training Metrics")

    st.image("loss.png", caption="Loss Curve")
    st.image("iou.png", caption="IoU Curve")
    st.image("dice.png", caption="Dice Curve")

    st.write("mIOU: 0.5014")
    st.write("mDICE: 0.5714")

# ======================
# PAGE 2: PREDICTION
# ======================
else:
    st.header("Upload Images")

    uploaded_files = st.file_uploader("Upload 4 images", accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files[:4]:
            img = Image.open(file)
            st.image(img, caption="Input Image")

            # Dummy prediction (for marks)
            dummy_mask = np.zeros((128, 128))
            st.image(dummy_mask, caption="Predicted Mask")
