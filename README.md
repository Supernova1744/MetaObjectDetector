# MetaObjectDetector

A PyTorch-based framework for few-shot object detection (FSOD) using both baseline and MAML meta-learning approaches. This project leverages the [FSOD dataset](https://github.com/ucbdrive/fsod) and supports flexible episodic and standard training.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
  - [Baseline Training](#baseline-training)
  - [MAML Meta-Learning Training](#maml-meta-learning-training)
- [Results & Checkpoints](#results--checkpoints)
- [References](#references)

---

## Features
- **Few-Shot Object Detection**: Episodic and standard training modes.
- **Meta-Learning**: MAML (Model-Agnostic Meta-Learning) for fast adaptation.
- **Baseline**: Standard supervised Faster R-CNN training for comparison.
- **Flexible Dataset Loader**: Easily configure N-way, K-shot, and query splits.
- **Logging & Checkpointing**: Automatic CSV logging and best model saving.

---

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Supernova1744/MetaObjectDetector.git
   cd MetaObjectDetector
   ```

2. **Install dependencies:**
   - It is recommended to use a Python 3.9+ virtual environment.
   - Install all required packages:
     ```bash
     pip install -r requirements.txt
     ```
   - Ensure you have a working CUDA setup for GPU training (see [PyTorch Get Started](https://pytorch.org/get-started/locally/)).

---

## Dataset Preparation

This project uses the [FSOD dataset](https://github.com/ucbdrive/fsod).

1. **Download the FSOD dataset:**
   - Visit the [FSOD dataset GitHub page](https://github.com/ucbdrive/fsod) for download instructions.
   - Download the images and annotation files (e.g., `fsod_train.json`, `fsod_test.json`).

2. **Organize your data directory:**
   - Place images in a directory, e.g., `C:/Users/<YourUser>/Downloads/data/`
   - Place annotation files in `C:/Users/<YourUser>/Downloads/data/annotations/`
   - Example structure:
     ```
     data/
       images/
         ... (image files)
       annotations/
         fsod_train.json
         fsod_test.json
     ```

3. **Update paths in scripts if needed:**
   - The default scripts expect the above structure. If your paths differ, edit the `Image_dir` and `Annotation_path` variables in `train_baseline.py` and `train_maml.py`.

---

## Training

### Baseline Training

Trains a standard Faster R-CNN model on the FSOD dataset.

```bash
python train_baseline.py
```

- Training and validation losses, as well as mAP, are logged to a CSV file in the project root.
- The best model (by validation mAP) is saved in the `checkpoints/` directory.

### MAML Meta-Learning Training

Trains a meta-learned Faster R-CNN using MAML for few-shot adaptation.

```bash
python train_maml.py
```

- Episodic meta-training is performed.
- Support and query losses, as well as validation mAP, are logged to a CSV file.
- The best meta-learned model is saved in `checkpoints/`.

---

## Results & Checkpoints
- Training logs are saved as CSV files (e.g., `baseline_loss_log_*.csv`, `maml_log_*.csv`).
- Best model weights are saved in the `checkpoints/` directory.
- You can use these checkpoints for evaluation or further fine-tuning.

---

## References
- **FSOD Dataset:** [https://github.com/ucbdrive/fsod](https://github.com/ucbdrive/fsod)
- **MAML Paper:** [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
- **PyTorch:** [https://pytorch.org/](https://pytorch.org/)
- **Learn2Learn:** [https://github.com/learnables/learn2learn](https://github.com/learnables/learn2learn)

---

## Notes
- For best results, use a CUDA-enabled GPU.
- If you encounter issues with dataset loading or training, check your file paths and ensure all dependencies are installed.
- For custom N-way/K-shot settings, edit the parameters in the dataset instantiation in the training scripts.
