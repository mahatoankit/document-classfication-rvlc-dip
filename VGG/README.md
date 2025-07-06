### 🔧 Prompt for VGG16 Transfer Learning on RVL-CDIP Dataset

> **Task**: Implement a complete transfer learning pipeline using VGG16 on the RVL-CDIP dataset located at `/kaggle/input/rvl-cdip-dataset/test`. The dataset folder contains **all 40,000 labeled images** grouped by class names (folder names).

---

### ✅ Requirements:

1. **Data Handling**

   * Load all images from `/kaggle/input/rvl-cdip-dataset/test`
   * Automatically extract labels from folder names
   * Perform a **stratified split** into:

     * 70% training
     * 15% validation
     * 15% test

2. **Data Analysis & Preprocessing**

   * Visualize class distribution
   * Plot sample images from each class
   * Check for corrupted files or anomalies
   * Apply **outlier detection/removal** (based on image size, pixel stats, etc.)
   * Perform **dimensionality reduction** using **PCA or t-SNE/UMAP** (use PCA before training to analyze the feature space)
   * Normalize and preprocess images (resize to 224x224 for VGG16)

3. **Model**

   * Use **VGG16** (pretrained on ImageNet)
   * Remove top layers and add:

     * GlobalAveragePooling
     * Dense + Dropout + BatchNorm (as regularization)
     * Final Dense layer with softmax activation
   * Use **L2 Regularization** and **Dropout** to reduce overfitting
   * Freeze base layers initially, then unfreeze top N layers for fine-tuning

4. **Training**

   * Compile with Adam optimizer, categorical crossentropy, and accuracy metrics
   * Use learning rate scheduler or ReduceLROnPlateau
   * Implement early stopping and model checkpointing
   * Track training/validation loss and accuracy

5. **Evaluation**

   * Evaluate final model on the test set
   * Plot confusion matrix and classification report
   * Visualize Grad-CAM for interpretability (optional)
