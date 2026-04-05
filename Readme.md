# Assignment 5 - Deep Learning Operations
## ViT Fine-tuning with LoRA & Adversarial Attacks

**Name:** Kunal Mishra  
**Roll Number:** M25CSA036  
**Institution:** Indian Institute of Technology Jodhpur

---

## 🔗 Important Links
- 📊 **WandB:** https://wandb.ai/kunalmishra8765-indian-institute-of-technology-jodhpur/DLops-Assignment-5
- 🤗 **HuggingFace:** https://huggingface.co/101bytespeedkunal/dlops-assignment5-models
- 📋 **Report:** M25CSA036_Kunal_Ass5.pdf

---

```text

## 📁 Repository Structure
Assignment_5/
├── Q1/
│   ├── model.py           # ViT model with LoRA
│   ├── utils.py           # Data loading utilities
│   ├── train.py           # Training script
│   ├── test.py            # Testing script
│   └── optuna_search.py   # Hyperparameter search
├── Q2/
│   ├── train_clean.py     # Train ResNet18 on CIFAR10
│   ├── fgsm_scratch.py    # FGSM attack from scratch
│   ├── fgsm_art.py        # FGSM attack using IBM ART
│   ├── detector_pgd.py    # PGD adversarial detector
│   ├── detector_bim.py    # BIM adversarial detector
│   ├── detector_pgd.pth   # PGD detector weights
│   └── detector_bim.pth   # BIM detector weights
├── report/
│   └── M25CSA036_Kunal_Mishra_Ass5.pdf
├── Dockerfile
└── requirements.txt
└── M25CSA036_Kunal_Ass5.pdf

---

## 🐳 Docker Setup

### Build container:
```bash
docker build -t dlops-assignment5 .
```

### Run container:
```bash
docker run --gpus all -it dlops-assignment5 bash
```

---

## ⚙️ Installation
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch==2.1.0 torchvision==0.16.0 timm==0.9.12 peft==0.7.1 wandb==0.16.0 optuna==3.4.0 adversarial-robustness-toolbox==1.16.0 huggingface_hub==0.20.1 matplotlib==3.8.0 numpy==1.24.0 scikit-learn==1.3.0 tqdm==4.66.0
```

---

## 🚀 Q1 - ViT Fine-tuning on CIFAR-100

#### 📈 Training Graphs
All training graphs, loss curves and accuracy plots are available on WandB:
👉 https://wandb.ai/kunalmishra8765-indian-institute-of-technology-jodhpur/DLops-Assignment-5

### Train without LoRA:
```bash
cd Q1
python train.py --epochs 10 --lr 1e-3 --batch_size 64
```

### Train with LoRA:
```bash
python train.py --use_lora --rank 8 --alpha 8 --dropout 0.1 --epochs 10 --lr 1e-3 --batch_size 64
```

### Test best model:
```bash
python test.py --model_path best_model_lora_r8_a8.pth --use_lora --rank 8 --alpha 8 --dropout 0.1
```

### Optuna hyperparameter search:
```bash
python optuna_search.py
```

---

## 📊 Q1 Results

### Experiment Summary Table

| LoRA | Rank | Alpha | Dropout | Best Val Acc | Test Acc | Trainable Params |
|------|------|-------|---------|-------------|----------|-----------------|
| No   | -    | -     | -       | 81.82%      | -        | 38,500          |
| Yes  | 2    | 2     | 0.1     | 75.22%      | -        | 36,864          |
| Yes  | 2    | 4     | 0.1     | 78.02%      | -        | 36,864          |
| Yes  | 2    | 8     | 0.1     | 79.78%      | -        | 36,864          |
| Yes  | 4    | 2     | 0.1     | 81.56%      | -        | 73,728          |
| Yes  | 4    | 4     | 0.1     | 83.72%      | -        | 73,728          |
| Yes  | 4    | 8     | 0.1     | 84.50%      | -        | 73,728          |
| Yes  | 8    | 2     | 0.1     | 85.66%      | -        | 147,456         |
| Yes  | 8    | 4     | 0.1     | 85.08%      | -        | 147,456         |
| Yes  | 8    | 8     | 0.1     | 85.74%      | 86.30%   | 147,456         |

### No LoRA Training Progress

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1     | 0.9876    | 0.6947   | 73.55%    | 79.44%  |
| 2     | 0.5890    | 0.6816   | 82.60%    | 80.02%  |
| 3     | 0.5156    | 0.6625   | 84.23%    | 80.56%  |
| 4     | 0.4647    | 0.6699   | 85.63%    | 80.98%  |
| 5     | 0.4247    | 0.6551   | 86.85%    | 80.90%  |
| 6     | 0.3928    | 0.6465   | 87.77%    | 80.90%  |
| 7     | 0.3660    | 0.6332   | 88.54%    | 81.02%  |
| 8     | 0.3420    | 0.6092   | 89.30%    | 81.82%  |
| 9     | 0.3259    | 0.6212   | 89.90%    | 81.48%  |
| 10    | 0.3179    | 0.6168   | 90.22%    | 81.62%  |

### Best LoRA (Rank=8, Alpha=8) Training Progress

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1     | 1.8100    | 0.8095   | 56.91%    | 79.22%  |
| 2     | 0.6490    | 0.6500   | 82.96%    | 82.38%  |
| 3     | 0.4636    | 0.5529   | 87.72%    | 84.72%  |
| 4     | 0.3583    | 0.5217   | 90.77%    | 85.42%  |
| 5     | 0.2960    | 0.5047   | 92.70%    | 85.78%  |

### Optuna Best Configuration
- **Rank:** 8
- **Alpha:** 8  
- **Dropout:** 0.028
- **LR:** 0.00038
- **Best Val Accuracy:** 71.94%

---

## 🚀 Q2 - Adversarial Attacks

### Train ResNet18 on CIFAR-10:
```bash
cd Q2
python train_clean.py
```

### FGSM attack from scratch:
```bash
python fgsm_scratch.py
```

### FGSM attack using IBM ART:
```bash
python fgsm_art.py
```

### Train PGD detector:
```bash
python detector_pgd.py
```

### Train BIM detector:
```bash
python detector_bim.py
```

---

## 📊 Q2 Results

#### 📈 Training Graphs
All training graphs, loss curves and accuracy plots are available on WandB:
👉 [Wandb Link](https://wandb.ai/kunalmishra8765-indian-institute-of-technology-jodhpur/DLops-Assignment-5)

### ResNet18 Clean Accuracy: 93.16% (Target: ≥72%) ✅

### FGSM Attack Results

| Epsilon | Clean Acc | Scratch Acc | ART Acc | Scratch Drop | ART Drop |
|---------|-----------|-------------|---------|--------------|----------|
| 0.01    | 93.16%    | 46.10%      | 35.00%  | 47.06%       | 58.16%   |
| 0.05    | 93.16%    | 35.68%      | 14.30%  | 57.48%       | 78.86%   |
| 0.10    | 93.16%    | 29.40%      | 10.00%  | 63.76%       | 83.16%   |
| 0.20    | 93.16%    | 23.06%      | 10.00%  | 70.10%       | 83.16%   |
| 0.30    | 93.16%    | 19.40%      | 10.00%  | 73.76%       | 83.16%   |

### Adversarial Detector Results

| Attack | Model    | Detection Accuracy | Target | Status    |
|--------|----------|--------------------|--------|-----------|
| PGD    | ResNet34 | 100.00%            | ≥70%   | ✅ Passed |
| BIM    | ResNet34 | 100.00%            | ≥70%   | ✅ Passed |

---


## 🏆 Model Weights

> **Note:** Best Q1 model weight (best_model_lora_r8_a8.pth) exceeds 
> GitHub's 25MB file size limit and has been uploaded to HuggingFace instead.
> All Q2 weights are available on GitHub directly.

| Model | Size | Location |
|-------|------|----------|
| Best Q1 (LoRA Rank=8, Alpha=8) | >25MB | 🤗 HuggingFace: Q1/best_model_lora_r8_a8.pth |
| Q2 ResNet18 Clean | >25MB | 🤗 HuggingFace: Q2/resnet18_clean.pth |
| Q2 PGD Detector | <25MB | GitHub: Q2/detector_pgd.pth |
| Q2 BIM Detector | <25MB | GitHub: Q2/detector_bim.pth |

🤗 **HuggingFace Repository:** https://huggingface.co/101bytespeedkunal/dlops-assignment5-models
| Model | Location |
|-------|----------|
| Best Q1 (LoRA Rank=8, Alpha=8) | 🤗 HuggingFace: Q1/best_model_lora_r8_a8.pth |
| Q2 ResNet18 Clean | 🤗 HuggingFace: Q2/resnet18_clean.pth |
| Q2 PGD Detector | GitHub: Q2/detector_pgd.pth |
| Q2 BIM Detector | GitHub: Q2/detector_bim.pth |

