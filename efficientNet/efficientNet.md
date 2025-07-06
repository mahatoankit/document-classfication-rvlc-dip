# 🔧 GitHub Copilot Prompt: EfficientNet-B3 Image Classifier with PCA and Regularization

Build a complete PyTorch training notebook using **EfficientNet-B3** for a **16-class document image classification task**.

---

## 📂 Dataset Details

- Dataset directory: `/kaggle/input/rvl-cdip-dataset/test`
- This contains **all 40,000 labeled images** for training, validation, and testing.
- Even though the folder is named `test`, it includes **images from all classes**. Do **not** expect `/train` or `/val` folders.
- Split manually:
  - 70% → training set
  - 15% → validation set
  - 15% → test set
- Each image’s **label is the folder name** it's contained in.

---

## ✅ Data Pipeline

- Use `ImageFolder` to read the dataset.
- Randomly split into `train`, `val`, and `test` using `torch.utils.data.random_split`.
- Apply **Albumentations** for image augmentations:

### Training transforms:
- Resize to 300×300
- Random Horizontal Flip
- Rotation (±15°)
- Blur, Gaussian Noise, or Motion Blur
- Cutout (8 holes)
- Normalize to ImageNet stats
- Convert to Tensor with `ToTensorV2()`

### Validation/Test transforms:
- Resize to 300×300
- Normalize
- Convert to Tensor

---

## ✅ Model: EfficientNet-B3

- Load **EfficientNet-B3** from `timm` with `pretrained=True`
- Replace final layer with `nn.Linear(in_features, 16)`
- Fine-tune the **entire model** (not just classifier)

---

## 🧠 Dimensionality Reduction (Post-Feature Extraction)

- Before classification, apply **PCA** or **t-SNE/UMAP** on extracted features from the penultimate layer
- Explore:
  - `sklearn.decomposition.PCA(n_components=128 or 256)`
  - Optional: Visualize 2D embeddings using PCA or t-SNE

---

## 🛠️ Training Setup

- **Optimizer**: AdamW
- **Learning Rate**: 3e-4
- **Scheduler**: CosineAnnealingLR
- **Epochs**: 30–50
- **Loss Function**: CrossEntropyLoss with optional **Label Smoothing**
- **Batch Size**: 32
- **Regularization**:
  - Dropout in classifier head (e.g., `nn.Dropout(0.3)`)
  - Weight decay in optimizer (`weight_decay=1e-4`)
  - Label Smoothing via custom cross-entropy function
  - Early stopping if val accuracy doesn't improve for 5 epochs

---

## 📊 Evaluation

- Print training/validation accuracy + loss after each epoch
- Save best model as `efficientnet_best.pth`
- Plot loss and accuracy graphs
- Generate:
  - Confusion Matrix
  - Classification Report
  - Accuracy per class

---

## 📈 (Optional) Ensemble Learning

- Train two models:
  - EfficientNet-B3
  - ResNet-101
- Average softmax outputs to ensemble predictions

---

## 🧪 Inference

- Load best model
- Run prediction on random test images from the 15% split
- Visualize prediction + true class label

---

**Deliver a clean, modular Jupyter Notebook** with all of the above, runnable end-to-end inside **Kaggle** or **Colab (with dataset uploaded)**.
