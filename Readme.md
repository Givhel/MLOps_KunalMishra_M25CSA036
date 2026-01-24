# DL-Ops Assignment 1

Name: Kunal Mishra    

Roll Number: M25CSA036 

Colab Notebook Link:  
https://colab.research.google.com/drive/1KvirxsQvSsGZPJsTleeCR7-_l7cJIVzW?usp=sharing

---
## Q1(a): CNN Experiments on MNIST and FashionMNIST**

Trained ResNet-18 and ResNet-50 (pretrained = False) on both datasets with 70-10-20 split.

### MNIST Results

| Batch Size | Optimizer | Learning Rate | ResNet-18 (%) | ResNet-50 (%) |
|-----------|----------|--------------|-------------|-------------|
| 16 | SGD | 0.001 | 98.37 %| 97.94 % |
| 16 | SGD | 0.0001 | 95.07 % | 86.62 %  |
| 16 | Adam | 0.001 | 98.18 % | 97.94 % |
| 16 | Adam | 0.0001 | 98.65 % | 96.72 % |
| 32 | SGD | 0.001 | 97.53 % | 96.55 % |
| 32 | SGD | 0.0001 | 92.99 % | _ |
| 32 | Adam | 0.001 | 98.91 % | _ |
| 32 | Adam | 0.0001 | 92.99 % | 96.78 %  |

### FashionMNIST Results

| Batch Size | Optimizer | Learning Rate | ResNet-18 (%) | ResNet-50 (%) |
|-----------|----------|--------------|-------------|-------------|
| 16 | SGD | 0.001 | 88.18 % | 84.71 % |
| 16 | SGD | 0.0001 | 82.25 % | 77.39 % |
| 16 | Adam | 0.001 | 88.35 % | 87.18 % |
| 16 | Adam | 0.0001 | 82.76 % | 87.92 % |
| 32 | SGD | 0.001 | 87.22 % | 83.83 % |
| 32 | SGD | 0.0001 | _ | _ |
| 32 | Adam | 0.001 | _ | _ |
| 32 | Adam | 0.0001 | 90.33 % | 86.24 % |

---

## Q1(b): SVM Experiments

| Dataset | Kernel | Test Accuracy (%) | Training Time (ms) |
|--------|-------|------------------|-------------------|
| MNIST | poly | 98.09 | 97204 |
| MNIST | rbf | 97.86 | 122653 |
| FashionMNIST | poly | 89.22 | 139078 |
| FashionMNIST | rbf | 88.60 | 155279 |

---

## Q2: CPU vs GPU Comparison (AMP Enabled on GPU)

| Compute | Model     | Optimizer | Epochs | Accuracy (%) | Time (ms)   | FLOPs (GFLOPs) |
|--------|-----------|-----------|--------|--------------|------------|----------------|
| CPU | ResNet-18 | SGD  | 3 | 86.93 | 1851212.34 | 0.96 |
| CPU | ResNet-18 | Adam | 3 | 88.08 | 2748829.62 | 0.96 |
| CPU | ResNet-50 | SGD  | 1 | 75.86 | 1395364.84 | 2.56 |
| CPU | ResNet-50 | Adam | 1 | 70.91 | 1878685.74 | 2.56 |
| GPU | ResNet-18 | SGD  | 5 | 88.34 | 202796.74  | 0.96 |
| GPU | ResNet-18 | Adam | 5 | 90.34 | 244300.73  | 0.96 |
| GPU | ResNet-50 | SGD  | 5 | 84.39 | 401587.10  | 2.56 |
| GPU | ResNet-50 | Adam | 5 | 81.62 | 499694.68  | 2.56 |

FLOPs were computed using THOP and represent training computation per batch (forward + backward) for batch size = 16.
