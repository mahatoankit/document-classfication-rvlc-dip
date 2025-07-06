# Document Classification using VGG16 Transfer Learning

A comprehensive implementation of document classification using VGG16 transfer learning on the RVL-CDIP dataset. This project demonstrates state-of-the-art techniques for classifying documents into 16 different categories.

## 📁 Project Structure

```
DocumentClassification/
├── requirements.txt              # Python dependencies
├── README.md                    # This file
├── efficientNet/
│   ├── EfficientNet-B3.ipynb   # EfficientNet implementation
│   └── efficientNet.md         # EfficientNet documentation
└── VGG/
    ├── vgg16-sota.ipynb         # Main VGG16 implementation
    └── INSTRUCTION.md           # VGG specific instructions
```

## 🎯 Project Overview

This project implements a robust document classification system using VGG16 transfer learning to classify documents from the RVL-CDIP dataset into 16 different categories. The implementation includes comprehensive data analysis, preprocessing, model training, and evaluation.

### Document Classes (16 categories)

- Advertisement
- Budget
- Email
- File Folder
- Form
- Handwritten
- Invoice
- Letter
- Memo
- News Article
- Presentation
- Questionnaire
- Resume
- Scientific Publication
- Scientific Report
- Specification

## 🚀 Features

- **Transfer Learning**: Utilizes pre-trained VGG16 model with ImageNet weights
- **Two-Phase Training**: Frozen base layers followed by fine-tuning
- **Comprehensive Analysis**: Includes PCA analysis, data visualization, and quality checks
- **Robust Preprocessing**: Image quality validation and standardized preprocessing
- **Performance Metrics**: Detailed evaluation with confusion matrices and classification reports
- **Model Persistence**: Saves trained models and preprocessing components
- **Prediction Pipeline**: Ready-to-use prediction function for new documents

## 📋 Requirements

### System Requirements

- Python 3.7+
- GPU support recommended (CUDA compatible)
- Minimum 8GB RAM (16GB recommended)
- 10GB+ free disk space

### Python Dependencies

All required packages are listed in `requirements.txt`. Key dependencies include:

- TensorFlow 2.10+
- Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- OpenCV, Pillow
- Jupyter Notebook

## 🔧 Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd DocumentClassification
   ```

2. **Create virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the RVL-CDIP dataset**
   - Download from [RVL-CDIP Dataset](https://www.cs.cmu.edu/~aharley/rvl-cdip/)
   - Extract to your preferred location
   - Update the `data_path` in the configuration cell

## 📊 Dataset

The RVL-CDIP dataset contains 400,000 grayscale images of documents categorized into 16 classes. Each image is approximately 224x224 pixels after preprocessing.

### Data Split

- **Training**: 70% of the dataset
- **Validation**: 15% of the dataset
- **Testing**: 15% of the dataset

## 🏗️ Model Architecture

The model uses VGG16 as the base architecture with the following modifications:

```
VGG16 Base (Pre-trained on ImageNet)
├── Global Average Pooling 2D
├── Batch Normalization
├── Dense (512 units, ReLU) + L2 Regularization
├── Dropout (0.5)
├── Batch Normalization
├── Dense (256 units, ReLU) + L2 Regularization
├── Dropout (0.5)
└── Dense (16 units, Softmax) # Output layer
```

## 🎓 Training Process

### Phase 1: Frozen Base Training

- **Epochs**: 20
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Strategy**: Train only the custom classification head

### Phase 2: Fine-tuning

- **Epochs**: 15
- **Learning Rate**: 0.0001
- **Strategy**: Unfreeze top 4 layers of VGG16 for fine-tuning

### Training Features

- **Early Stopping**: Prevents overfitting
- **Learning Rate Reduction**: Adaptive learning rate scheduling
- **Model Checkpointing**: Saves best models during training
- **TensorBoard Logging**: Comprehensive training monitoring

## 📈 Performance Metrics

The model is evaluated using multiple metrics:

- **Accuracy**: Overall classification accuracy
- **Top-3 Accuracy**: Accuracy when considering top 3 predictions
- **Precision, Recall, F1-Score**: Per-class and weighted averages
- **Confusion Matrix**: Detailed class-wise performance analysis

## 🔍 Usage

### Running the Notebook

1. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Navigate to `VGG/vgg16-sota.ipynb`

3. Update the dataset path in the configuration cell:

   ```python
   CONFIG = {
       'data_path': '/path/to/your/rvlcdip/dataset',
       # ... other configurations
   }
   ```

4. Run all cells sequentially

### Using the Trained Model

After training, you can use the model for predictions:

```python
# Load the trained model
model = tf.keras.models.load_model('/kaggle/working/vgg16_document_classifier.h5')

# Load label encoder
with open('/kaggle/working/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Predict a document class
predicted_class, confidence, top_3 = predict_document_class('path/to/document.jpg')
print(f"Predicted class: {predicted_class} (confidence: {confidence:.3f})")
```

## 📁 Output Files

The training process generates several output files in `/kaggle/working/`:

### Model Files

- `vgg16_document_classifier.h5`: Final trained model
- `best_vgg16_model.h5`: Best model from Phase 1
- `best_vgg16_finetuned.h5`: Best model from Phase 2
- `vgg16_architecture.json`: Model architecture
- `label_encoder.pkl`: Label encoder for class mapping

### Analysis Files

- `training_history.csv`: Complete training history
- `training_summary.csv`: Summary statistics
- `classification_report.txt`: Detailed performance report
- `per_class_accuracy.csv`: Per-class performance metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `comprehensive_analysis.png`: Complete training analysis

### Logs

- `training_log.csv`: Training metrics log
- `logs/`: TensorBoard logs directory

## 🔧 Configuration

Key configuration parameters in the notebook:

```python
CONFIG = {
    'data_path': '/path/to/dataset',      # Dataset location
    'img_size': (224, 224),               # Input image size
    'batch_size': 32,                     # Training batch size
    'train_ratio': 0.7,                   # Training split ratio
    'val_ratio': 0.15,                    # Validation split ratio
    'test_ratio': 0.15,                   # Test split ratio
    'random_state': 42                    # Random seed
}
```

## 🎯 Best Practices

1. **Data Quality**: The notebook includes comprehensive data quality checks
2. **Reproducibility**: Fixed random seeds for consistent results
3. **Memory Management**: Efficient data loading with generators
4. **Model Validation**: Stratified splits and cross-validation
5. **Monitoring**: TensorBoard integration for training visualization

## 🚨 Troubleshooting

### Common Issues

1. **GPU Memory Issues**

   - Reduce batch size
   - Enable memory growth for GPU
   - Use mixed precision training

2. **Dataset Path Issues**

   - Verify dataset structure matches expected format
   - Check file permissions
   - Ensure correct path separators for your OS

3. **Training Slow/Stalled**
   - Check GPU utilization
   - Verify data loading pipeline
   - Monitor system resources

### Memory Optimization

```python
# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

## 📝 Results Interpretation

### Model Performance

- **Test Accuracy**: Typically achieves 85-90% accuracy
- **Top-3 Accuracy**: Usually 95%+ for top-3 predictions
- **Best Classes**: Forms, invoices, and letters typically perform best
- **Challenging Classes**: Handwritten documents and specifications may be more difficult

### Analysis Tools

- **Confusion Matrix**: Identifies class-specific errors
- **PCA Visualization**: Shows feature space representation
- **Training Curves**: Monitors learning progress
- **Per-Class Metrics**: Detailed performance breakdown

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **RVL-CDIP Dataset**: Thanks to the creators of the RVL-CDIP dataset
- **VGG16 Architecture**: Based on the Visual Geometry Group's VGG16 model
- **TensorFlow/Keras**: For the deep learning framework
- **Open Source Community**: For the various libraries and tools used

## 📞 Support

For questions or issues:

1. Check the troubleshooting section
2. Review the notebook comments and documentation
3. Create an issue in the repository
4. Contact the maintainers

---

**Note**: This project is designed for educational and research purposes. For production use, consider additional optimizations, security measures, and thorough testing.
