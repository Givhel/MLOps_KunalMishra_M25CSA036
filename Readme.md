# DLOps Lab 2 – CNN on CIFAR-10

## Overview
This repository contains the implementation for **DLOps Lab Worksheet – Lab 2**, where a Convolutional Neural Network (CNN) is trained on the CIFAR-10 dataset.  
The assignment focuses on model training, computational analysis, and training observability using modern DLOps practices.

---

## Objectives
- Train a CNN model on the CIFAR-10 dataset  
- Implement a custom PyTorch Dataset and DataLoader  
- Calculate FLOPs for the selected CNN model  
- Visualize gradient flow and weight update flow  
- Track all metrics and visualizations using Weights & Biases (WandB)  

---

## Dataset
- **Name:** CIFAR-10  
- **Images:** 60,000 RGB images  
- **Image Size:** 32 × 32  
- **Classes:** 10  
- **Train/Test Split:** 50,000 / 10,000  

A custom dataset class was implemented instead of directly using the built-in CIFAR-10 loader to simulate real-world data pipelines.

---

## Model Architecture
The CNN model consists of:
- Three convolutional layers with increasing channel depth  
- ReLU activation functions  
- Batch Normalization for training stability  
- Max Pooling layers for spatial downsampling  
- Two fully connected layers for classification  

The architecture is lightweight and suitable for CIFAR-10 classification.

---

## FLOPs Analysis
- FLOPs were computed using the **fvcore** library  
- Input size used: **3 × 32 × 32**  
- FLOPs were calculated for a single forward pass  
- The model was found to be computationally efficient for small-scale image classification tasks  

---

## Training Configuration
- **Optimizer:** Adam  
- **Loss Function:** Cross-Entropy Loss  
- **Batch Size:** 64  
- **Epochs:** 25  
- **Hardware:** GPU (Google Colab)  

Training progress was monitored using a live progress bar, and epoch-level metrics were logged to WandB.

---

## Gradient Flow & Weight Update Analysis
- Gradient flow was visualized once per epoch to analyze gradient propagation  
- Weight update norms were computed by comparing parameters before and after optimizer steps  
- Stable gradients and decreasing weight update magnitudes were observed, indicating healthy training and convergence  

---

## Experiment Tracking (Weights & Biases)
All experiments were tracked using **Weights & Biases (WandB)**, including:
- Training loss  
- Training accuracy  
- Validation accuracy  
- Gradient flow visualizations  
- Weight update flow metrics  

All visualizations are available through the provided WandB project link.

---

