# Reinforcement Learning — Homework 9  
### Convolutional Neural Networks for Building Classification  
**Course:** CSCI 4160/6963, ECSE 4965/6965  
**Source:** :contentReference[oaicite:0]{index=0}

---

## 📌 Overview

This project involves training convolutional neural networks (CNNs) for two tasks:

1. **Warm-up:** Achieve **≥70% test accuracy** on CIFAR-10 using a small CNN (≤4 hidden layers; batch norm, dropout, and max pool excluded from the limit).  
2. **Main Task:** Train a larger CNN (≤5 million parameters) to classify **11 buildings on the RPI campus**, using ~500 images per class (image size: 252×189×3).

A hidden test set will be used for grading.  
The **top two teams** on the hidden test set receive **+10 bonus points**.

---

## 🖼 Dataset

The campus-building image dataset is stored on the CCI cluster:

- **Classes:** 87 Gym, Amos Eaton, EMPAC, Greene, JEC, Lally, Library, Ricketts, Sage, Troy Building, Voorhees  
- **Path:** `~/scratch-shared/all_images`

Images are split into:
- Training set  
- Validation set  
- Hidden test set (instructor-supplied only)

---

## 💻 Environment Setup (CCI Cluster)

All training must be performed on the **CCI NPL cluster**.

### Connect to Landing Pad
```bash
ssh RNL3user@blp01.ccni.rpi.edu
