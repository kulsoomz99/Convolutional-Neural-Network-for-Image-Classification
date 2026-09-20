# Convolutional Neural Networks for Image Classification: Custom Architectures & Transfer Learning

This repository contains the code and experiments for an **Image Processing and Computer Vision** module assignment. The project focuses on image classification using the **Oxford-IIIT Pet Dataset** through two primary approaches: designing a custom CNN from scratch with an ablation study, and applying transfer learning using a pre-trained ResNet-18.

## Dataset
- **Oxford-IIIT Pet Dataset**: A 37-category pet dataset containing thousands of images of various cat and dog breeds. 
- *Note: The notebook automatically clones the dataset repository and parses the annotations via a custom PyTorch `Dataset` class.*

---

## Project Components

### Part 1: Custom CNN Architecture & Ablation Study
A modular CNN was designed from scratch using fundamental PyTorch layers. The architecture incorporates:
- **DeepStem**: For initial spatial resolution reduction.
- **InceptionBlocks**: For multi-scale feature extraction.
- **Configurable Components**: Customizable activation functions, normalization layers, and dropout rates.

**Ablation Study**: 
To understand the impact of various design choices, an extensive ablation study was conducted. The following variations were tested against a baseline configuration:
- Activation Functions (ReLU vs. LeakyReLU)
- Network Depth (4 Stages vs. 2 Stages)
- Dropout Rates (Head vs. Feature Extractor)
- Stem Design (DeepStem vs. Classic 7x7 Kernel)
- Normalization (BatchNorm vs. Identity)
- Data Augmentation (Enabled vs. Disabled)

### Part 2: Transfer Learning with ResNet-18
To push the classification accuracy towards the ~90% target, a pre-trained ImageNet **ResNet-18** model was fine-tuned. Three distinct fine-tuning strategies were evaluated:
1. **Baseline Fine-Tuning**: Full network training with a simple single `Linear` classification head.
2. **Layer-Wise Fine-Tuning**: Freezing the backbone initially, then unfreezing it using **differential learning rates** (lower LR for the feature extractor, higher LR for the head).
3. **Enhanced Head Fine-Tuning**: Replacing the simple linear head with a deeper `Sequential` block (`Linear -> ReLU -> Linear`) combined with the differential learning rate strategy.

---

## Key Findings
- **Normalization is Critical**: Removing Batch Normalization caused the custom CNN to collapse (accuracy dropped to ~2.7%), proving its necessity for training deep networks from scratch.
- **Data Augmentation**: Disabling augmentation led to severe overfitting, highlighting its importance for generalization.
- **Dropout Placement**: Counter-intuitively, removing dropout from the feature extractor yielded the best custom model performance, suggesting the default regularization was too aggressive.
- **Transfer Learning Strategy**: The **Enhanced Head with differential learning rates** proved to be the most robust. It prevented the catastrophic forgetting and overfitting seen in the baseline approach, achieving the highest validation accuracy (~88.9%) with the most stable loss curves.

---

##  Requirements & Dependencies
The project is built using **PyTorch**. To install the required dependencies, run:

```bash
pip install torch torchvision torchmetrics tqdm matplotlib pandas torchsummary
