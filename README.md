# 😊 Facial Emotion Classification — From Classic ML to Deep Learning

Classifies a face image into one of four emotions — **Angry, Happy, Sad, Neutral**.

This project documents a complete journey: starting from a **classic machine-learning
baseline** and iterating all the way to a **production-grade deep-learning model**,
improving test accuracy from **57% → 90.2%**. Built and run entirely on **Google Colab**.

---

## 📈 The journey (three notebooks, one story)

| Stage | Notebook | Approach | Dataset | Test Accuracy |
|-------|----------|----------|---------|:---:|
| **1. Classic ML baseline** | `Image_classifier.ipynb` | HOG features + SVM / Random Forest / KNN | FER-2013 | **57%** |
| **2. Deep learning** | `image_classifier_dl.ipynb` | CNN (transfer learning + custom CNN) | FER+ | **82.7%** |
| **3. Production model** | `Emotion_Classifier_Production.ipynb` | EfficientNetV2S transfer learning | RAF-DB | **90.2%** |

Each stage tackled the previous one's limitation: noisy features → cleaner features &
CNNs → cleaner data (RAF-DB) + a stronger model + professional preprocessing.

---

## 🎯 Final model (Stage 3) — results

**90.2% test accuracy** (with Test-Time Augmentation), **macro-F1 0.88** — at the research
frontier for RAF-DB.

| Emotion | Precision | Recall | F1 |
|---------|:---:|:---:|:---:|
| Angry | 0.90 | 0.83 | 0.87 |
| Happy | 0.96 | 0.93 | 0.95 |
| Sad | 0.86 | 0.87 | 0.86 |
| Neutral | 0.84 | 0.89 | 0.86 |

Balanced across all classes despite heavy class imbalance (handled with class weights).

---

## 🧠 Key techniques demonstrated

**Classic ML**
- HOG (Histogram of Oriented Gradients) feature extraction
- SVM, Random Forest, KNN — trained and compared
- StandardScaler, cross-validation, GridSearchCV

**Deep learning**
- Transfer learning (MobileNetV2, EfficientNetB0, **EfficientNetV2S**)
- Two-phase fine-tuning (freeze → unfreeze)
- Custom CNN trained from scratch (for FER)

**Professional preprocessing**
- **CLAHE** lighting normalization (LAB color space)
- Data augmentation (flip, rotation, zoom, translation, brightness, contrast)
- Efficient `tf.data` pipeline (cache, prefetch, batch)

**Engineering & evaluation**
- Class weights for imbalance
- Mixed-precision training (faster on GPU)
- Stratified train/validation/test split (no data leakage)
- Confusion matrix, precision/recall/F1, macro-F1
- Test-Time Augmentation (TTA)
- Saved model + reusable `predict()` inference function

---

## ▶️ How to run (Google Colab)

1. Open **[colab.research.google.com](https://colab.research.google.com)** → **File → Upload notebook** → pick a notebook.
2. For the deep-learning notebooks: **Runtime → Change runtime type → T4 GPU**.
3. Get a Kaggle token: **[kaggle.com/settings](https://kaggle.com/settings) → API → Create Legacy API Key** (downloads `kaggle.json`).
4. Run the cells top to bottom; upload `kaggle.json` when prompted — the dataset downloads automatically.

**Datasets (Kaggle):**
- FER-2013 — `msambare/fer2013` + FER+ labels (`microsoft/FERPlus`)
- RAF-DB — `shuvoalok/raf-db-dataset`

---

## 🗂️ Repository structure

```
.
├── README.md                              # this file
├── requirements.txt                       # libraries
├── Image_classifier.ipynb                 # Stage 1: classic ML (57%)
├── image_classifier_dl.ipynb              # Stage 2: deep learning on FER+ (82.7%)
├── Emotion_Classifier_Production.ipynb    # Stage 3: production on RAF-DB (90.2%)
```

---

## ⚠️ Honest limitations

- Real-world facial-emotion labels are **inherently ambiguous** (even humans agree only ~65–70% on some faces), so ~90% on RAF-DB is the realistic research ceiling — not a tuning failure.
- `angry` is the rarest class, so it's the hardest; class weighting helped, but more data would help more.
- The model reflects RAF-DB's distribution; very different lighting/angles/populations (domain shift) would reduce accuracy.
- Single-face classification only — not multi-face or video.

---

## 🚀 Possible next steps

- Ensemble multiple backbones (+~1%)
- Face alignment preprocessing
- More data (AffectNet) or more `angry` samples
- Grad-CAM explainability (visualize where the model looks)
- Export to TensorFlow Lite / an API for deployment

---

## 🛠️ Tech stack

Python · TensorFlow / Keras · scikit-learn · scikit-image · OpenCV · NumPy · pandas · matplotlib · seaborn · Google Colab

---

*A portfolio project showing both classic ML and modern deep learning, with honest, rigorous evaluation at every step.*
