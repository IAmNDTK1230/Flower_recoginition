# Flower Detection and Recognition using YOLO, ConvNeXt and Vision Transformer (ViT)

## Overview
This project presents an end-to-end flower recognition system combining **object detection** and **image classification**.

- **YOLO** is used to detect flowers in images and localize bounding boxes.
- **ConvNeXt** and **Vision Transformer (ViT)** are used to classify detected flowers into specific categories.

The project demonstrates a complete computer vision pipeline from raw images to final flower species prediction, suitable for real-world applications such as smart gardening and plant identification systems.

---

## Dataset
- **Oxford 102 Flower Dataset**
- 102 flower categories
- Images with high variation in background, scale, and lighting conditions
- **Open Images V7 Dataset**
- ~5000 images for label “flower”

The dataset is used for:
- Training and evaluating classification models (ConvNeXt, ViT)
- Detection data is prepared by converting images into YOLO format

---

## Models
### 1. Object Detection
- **YOLO**
  - Detects flower regions and outputs bounding boxes

### 2. Image Classification
- **ConvNeXt**
  - CNN-based architecture with strong feature extraction capability
- **Vision Transformer (ViT)**
  - Transformer-based model that captures global image context

---
## Fine-Tuning

We apply transfer learning to adapt pre-trained models to the flower recognition task.

### Stage 1 – Feature Freezing
- Freeze backbone layers  
- Train only the classification / detection head  
→ Stabilizes training and learns task-specific decision boundaries

### Stage 2 – Unfreezing
- Partially or fully unfreeze the backbone  
- Use a smaller learning rate for backbone layers  
→ Enables domain-specific feature adaptation

**Benefits:** Faster convergence and better performance on limited and imbalanced data.


## System Pipeline
1. Input image
2. Flower detection using YOLO
3. Crop detected flower regions
4. Flower classification using ConvNeXt or ViT
5. Final prediction output

---

##  Experimental Results
| Task            | Model      | Metric        | Result  |     Metric    | Result |     Metric      | Result |
|-----------------|------------|---------------|---------|---------------|--------|-----------------|--------|
| Detection       | YOLO       | mAP           | 0.771   |    x          |   x    |    x            |   x    |  
| Classification  | ConvNeXt   | Accuracy      | 0.9951  |F1-score macro | 0.9961 |F1-score weightz | 0.9963 |
| Classification  | ViT        | Accuracy      | 0.9902  |F1-score macro | 0.9976 |F1-score weightz | 0.9975 |    

> Results may vary depending on training configuration.

---

## Demo Results

### Input Image
| | | |
|---|---|---|
| ![](assets/demo1.jpg) | ![](assets/demo2.jpg) | ![](assets/demo3.jpg) |

### ConvNeXt
| | | |
|---|---|---|
| ![](assets/demo1_ConvNext.jpg) | ![](assets/demo2_ConvNext.jpg) | ![](assets/demo3_ConvNext.jpg) |

### ViT
| | | |
|---|---|---|
| ![](assets/demo1_vit.jpg) | ![](assets/demo2_vit.jpg) | ![](assets/demo3_Vit.jpg) |

---
## Log
ConvNeXt

[https://wandb.ai/tuankiet1302051-fpt-university/CNN-Flower-Classification/reports/ConvNext--VmlldzoxNTc5MTkxNg?accessToken=c6ab9qyj6l3dtgxaqs0nczbehk6tt0sp01lz1j6y76f4yyy10eybzeepuikrsuni
](https://api.wandb.ai/links/tuankiet1302051-fpt-university/cm30jqd5)


Vit

https://wandb.ai/blud-fpt-university/flower-classification-vit/reports/ViT-Model--VmlldzoxNTc5ODUxMg

