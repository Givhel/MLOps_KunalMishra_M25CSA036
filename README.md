# HuggingFace Docker MLOps Assignment

This repository demonstrates an end-to-end Machine Learning Operations (MLOps) workflow using Docker and Hugging Face. The project covers model training, evaluation, containerization, and deployment to ensure reproducibility and portability of machine learning experiments.

---

## Model Link

https://huggingface.co/101bytespeedkunal/distilbert-review-genres

---

## Project Overview

This assignment implements a complete machine learning lifecycle, starting from model training and ending with evaluation and deployment. Docker is used to containerize both training and evaluation pipelines, Hugging Face Hub is used for model hosting and sharing, and the entire workflow is designed to be reproducible and production-ready.

---

## Model Details

- Model Architecture: DistilBERT  
- Task: Multi-class text classification  
- Framework: Hugging Face Transformers  
- Hardware Support: GPU-enabled training and evaluation  

---

## Training Pipeline

Build the Docker image for training:

```bash
docker build -t hf-train:v1 .
