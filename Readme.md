# Minor Examination – CIFAR-10 Image Classification using ResNet-18

## Student Information
- Name: Kunal Mishra  
- Roll Number: M25CSA036  
- Examination: Minor Exam  
- Course: MLOps / Deep Learning  

---

## Objective
The objective of this experiment is to build an end-to-end deep learning pipeline using PyTorch by:
- Loading a dataset from Hugging Face
- Creating a custom DataLoader with transformations
- Training a pretrained ResNet-18 model
- Tracking experiments using Weights & Biases
- Pushing the trained model to Hugging Face Hub
- Reloading the model for evaluation
- Visualizing performance metrics

---

## Dataset Details
- Dataset Source: Hugging Face Datasets
- Dataset Name: CIFAR-10
- Total Images: 60,000
- Training Images: 50,000
- Test Images: 10,000
- Image Size: 32 × 32 × 3

### Class Labels
| Label | Class Name |
|-----|-----------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

---

## Data Preprocessing
The following transformations were applied:
- Random Horizontal Flip (training data only)
- Conversion to Tensor
- Normalization
- Custom PyTorch DataLoader was created for both training and test datasets

---

## Model Architecture
- Model: ResNet-18
- Source: torchvision.models (pretrained on ImageNet)
- Modification: Final fully connected layer replaced to support 10 classes
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Training Epochs: 2
- Hardware: Google Colab (GPU)

---

## Training & Experiment Tracking
Weights & Biases (WandB) was used to track experiments.  
The following metrics and visualizations were logged:
- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
- Confusion Matrix
- Class-wise Accuracy Bar Plot
- 10 Correctly Classified Test Images
- 10 Incorrectly Classified Test Images

WandB Project:
👉 https://wandb.ai/

---

## Model Deployment (Hugging Face)
- The best-performing model was saved locally
- The model was pushed to Hugging Face Model Hub
- The same model was later downloaded and reloaded for evaluation as required

Hugging Face Model Repository:
👉 https://huggingface.co/101bytespeedkunal/cifar10-resnet18-exam

---

## Evaluation Results
- Test Accuracy: *(as printed in notebook output)*
- Class-wise Accuracy: Printed in the notebook for all 10 classes
- Confusion Matrix: Logged and visualized on WandB

---

## Repository & Submission
- All code is written in a single Jupyter Notebook
- Notebook is fully executable from start to end
- The complete workflow includes training, logging, model upload, reload, and evaluation
- Code was executed on Google Colab

GitHub Repository (if applicable):
👉 https://github.com/Givhel/MLOps_KunalMishra_M25CSA036  
Branch: `minor_exam`

---

## Files Submitted
- Jupyter Notebook (.ipynb)
- README.md

---

## Declaration
I hereby declare that this submission is my own work and has been completed following the instructions provided for the minor examination.
