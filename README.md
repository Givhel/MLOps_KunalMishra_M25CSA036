# MLDLOPs Exam 2026

## 📌 Branch: MLDLOPs-Exam2026

---

# ✅ Question 1: NLP Translation Task

### 🔹 Model Used
HuggingFace pretrained model: Helsinki-NLP/opus-mt-bn-en

### 🔹 Task
- Translated Bengali text to English
- Saved output in `output.txt`

### 🔹 Evaluation
- Used `sacrebleu` to compute BLEU score

### 🔹 Result
BLEU Score: 0.49

---

# ✅ Question 2: Cityscape Image Segmentation

### 🔹 Dataset
- CameraRGB → Input images  
- CameraMask → Segmentation masks  
- Total Classes: 23  

### 🔹 Approach
- Train-test split: 80-20 (seed=42)
- Custom Dataset + DataLoader
- Model: UNet-based architecture
- Loss: CrossEntropyLoss
- Optimizer: Adam

### 🔹 Training
- Epochs: 7 (optimized for time)
- Batch size: 8

### 🔹 Evaluation Metrics
- mIOU (Mean Intersection over Union)
- mDICE Score

### 🔹 Final Results
Question2: mIOU: 0.5014 and mDICE: 0.5714

---

### 🔹 Plots
- Training Loss Curve → `loss.png`
- IoU Curve → `iou.png`
- Dice Curve → `dice.png`

---

# ✅ Streamlit App

### 🔹 Features

## Page 1:
- Displays:
  - Training Loss Curve
  - IoU Curve
  - Dice Curve
  - Final mIOU and mDICE values

## Page 2:
- Upload 4 test images
- Shows:
  - Input Image
  - Predicted Segmentation Mask (dummy visualization)

---

# ⚙️ How to Run

```bash
# Install dependencies
pip install torch torchvision matplotlib streamlit scikit-learn

# Run training
python train.py

# Run Streamlit app
streamlit run app.py
