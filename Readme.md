# Transformer Translation Optimization (English → Hindi)

## 📌 Overview
This project focuses on optimizing a Transformer-based translation model using Ray Tune and Optuna. The goal was to improve efficiency and achieve comparable or better performance in fewer training epochs.

---

## ⚙️ Baseline Performance
- Epochs: 100  
- Loss: 0.098  
- BLEU Score: 0.68  

---

## 🚀 Hyperparameter Tuning
Used Ray Tune with Optuna to tune:
- Learning Rate
- Batch Size
- Number of Attention Heads
- Feedforward Dimension
- Dropout

---

## 🏆 Best Configuration
- Learning Rate: 0.0003  
- Batch Size: 32  
- Num Heads: 4  
- d_ff: 1024  
- Dropout: 0.27  

---

## 📊 Final Results
- Epochs: 10  
- Loss: 1.53  
- BLEU Score: **0.6956**  

✅ Achieved better BLEU score than baseline  


---

## ⚠️ Challenges
- GPU session disconnections in Google Colab  
- Limited computational resources  

---

## 📂 Files
- `rollno_ass_4_tuned_en_to_hi.ipynb` → Main notebook  
- `rollno_ass_4_report.pdf` → Report  
- `rollno_ass_4_best_model.pth` → Trained model  

---

## 📦 Model Weights
Due to GitHub file size limitations (>100MB), the trained model is not uploaded directly.

👉 You can access it here: **[Best Model.pth](https://drive.google.com/file/d/1ya5e7v1Ud_QBg6O1nQ2Yp3LEV6gAEpeh/view?usp=sharing)]**

---

## 🧠 Key Learnings
- Hyperparameter tuning significantly improves convergence speed  
- BLEU score is a better evaluation metric than loss for translation  
- Efficient configurations can outperform baseline in fewer epochs  
- Resource management is important in real-world ML workflows  

---

## 🎯 Conclusion
The optimized Transformer model successfully outperformed the baseline by achieving a higher BLEU score in significantly fewer epochs, demonstrating the effectiveness of Ray Tune and Optuna for hyperparameter optimization.

---

## 👨‍💻 Author
Kunal Mishra  
