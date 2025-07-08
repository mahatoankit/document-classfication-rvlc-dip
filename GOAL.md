
## 🎯 Goal:

**Train a vision-only document classification model that is performant enough to rival or approach multimodal models**, with the intent of submitting to a conference (e.g., **NCCI** at KU).

---

## ✅ Recommended Dataset for Classification

Use **[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/)** as your primary benchmark:

* **400,000** grayscale scanned document images.
* **16 classes**: e.g., Letter, Memo, Form, Email, Invoice, etc.
* Standard benchmark for document classification.
* Widely used in multimodal models too, so comparisons are possible.

You could also augment or test with:

* **Tobacco3482** (for small-scale benchmarking)
* **DocBank** / **DocLayNet** (for additional layout-aware visual pretraining, not classification)

---

## 🔍 Challenges vs. Multimodal Models

Multimodal models (e.g., LayoutLMv3, TILT, DocFormer) use:

* **Text** (OCR)
* **Layout** (bounding boxes)
* **Visuals** (raw image)

These models **perform better** because they understand both **content** and **structure**, but they:

* Require **expensive pretraining**
* Depend on **OCR**, which is error-prone in low-quality scans or multilingual settings
* Need **more compute**, especially during training

If you can close the performance gap using **pure vision**, your paper would be **highly relevant**, especially in constrained environments.

---

## 🚀 Suggested Research Strategy

### 1. **Use Strong Vision Backbones**

Use modern image classification architectures that are competitive or SOTA:

* **ConvNeXtV2**
* **Swin Transformer V2**
* **Vision Transformers (ViT-H/ViT-L)**
* **EfficientNetV2**

Pretrain on **ImageNet-21k** or a large-scale document-specific corpus if available (e.g., synthetic document images).

### 2. **Document-Specific Pretraining (Optional but Useful)**

You can consider **self-supervised pretraining** on document datasets (DocLayNet, DocBank) using contrastive or masked image modeling objectives (e.g., MAE, DINOv2). This helps encode layout and visual patterns without OCR.

### 3. **Augmentation is Key**

Use document-specific augmentations:

* Blur, noise, distortions (scanner artifacts)
* CutMix, MixUp
* Grid masking or layout-aware masking
* Synthetic overlays (stamps, annotations, handwritten notes)

### 4. **Resolution Matters**

Train at **high resolutions** (e.g., 384×384, 512×512 or higher). Layout and structure cues become visible only at finer resolutions.

### 5. **Evaluation & Baselines**

Compare against:

* **LayoutLMv3** (Multimodal SOTA)
* **Donut** (OCR-free vision-to-sequence)
* **ViT on RVL-CDIP** (for vision-only baselines)
* **ResNet50** (common baseline on RVL-CDIP)

Make sure to test on **RVL-CDIP** test set (40k images) and report:

* **Accuracy**
* **Confusion matrix** (some classes are harder)
* **Per-class F1 scores**

### 6. **Efficiency Focus (Optional Section in Paper)**

Argue with:

* **Inference time**
* **Parameter count**
* **Training time**
* **No OCR requirement**

---

## 🧪 Optional Advanced Ideas (If Time Permits)

* **Patch-level token masking (MAE, BEiT pretraining)**
* **Multi-resolution input fusion**
* **Attention on visual tokens guided by layout heuristics (like attention to top-left areas)**
* **Contrastive learning with augmented views (SimCLR, MoCo)**

---

## 📚 Suggested Papers to Cite or Beat

* LayoutLMv3 (ICLR 2023)
* DocFormer (NeurIPS 2021)
* Donut (ECCV 2022)
* BiViT (arXiv 2023) – Vision-only baseline for RVL-CDIP
* SelfDoc (ICDAR 2021) – Vision + contrastive learning

---

## 📝 Paper Contribution Idea

> “We propose a high-performing vision-only model for document classification that matches or outperforms several multimodal baselines, while reducing computational and data dependencies (no OCR or layout annotations).”

This framing is **very conference-appropriate**, especially for practical AI applications in emerging markets (relevant to **NCCI**).

---

## 🔧 Tools

* Framework: **PyTorch** or **HuggingFace Transformers/Vision**
* Dataset prep: `torchvision.datasets.ImageFolder` or custom loaders for RVL-CDIP
* Pretrained weights: Use **timm** library (ConvNeXt, ViT, etc.)

---

## ✅ TL;DR Checklist

| Task                                       | Status |
| ------------------------------------------ | ------ |
| Use RVL-CDIP for classification            | ✅      |
| Choose strong vision model (ViT, ConvNeXt) | ✅      |
| Add document-specific augmentations        | ✅      |
| Train at high resolution                   | ✅      |
| Benchmark against multimodal models        | ✅      |
| Highlight efficiency + simplicity          | ✅      |
| Write paper for NCCI (KU)                  | 🔜     |

---
