# Parking Space Occupancy Detection  
_CECS 456 – Fall 2025 Final Project_

This project uses deep learning to classify individual parking spaces as:

- **Free**
- **Not free**
- **Partially free**

The classification is performed using two models:

1. **A custom Convolutional Neural Network (CNN)** built from scratch  
2. **A ResNet18 Transfer Learning model** pre-trained on ImageNet  

The dataset consists of **cropped parking-space images** derived from polygon-based annotations provided in the original Parking Space Detection Dataset.

---

##  Project Features

- Automated **data preparation** from XML polygon annotations  
- Lightweight **custom CNN** achieving ~95% accuracy  
- **ResNet18 transfer learning** achieving ~98% accuracy  
- Full evaluation:
  - Classification report
  - Confusion matrix
  - Accuracy & loss curves  
- **Reproducible** experiments with fixed random seed  
- Easy to run in **Google Colab** (no Drive setup required)

---

## Repo Structure

```
Parking-Space-Detection/
│
├── data/
│   ├── crops/                     # Training/evaluation dataset (903 images)
│   │   ├── free_parking_space/
│   │   ├── not_free_parking_space/
│   │   └── partially_free_parking_space/
│   │
│   └── raw/                       # Original Kaggle dataset (optional)
│       ├── boxes/
│       ├── images/
│       ├── annotations.xml
│       └── parking.csv
│
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_custom_cnn.ipynb
│   ├── 03_resnet18_transfer.ipynb
│
├── README.md
└── requirements.txt
```

---

## Running the Project in Google Colab

### 1. Clone the repository

Open a new Colab notebook and run:
```
!git clone https://github.com/blessthef4llen/Parking-Space-Detection.git
%cd Parking-Space-Detection
```
### 2. Open any notebook

### 3. Run all cells from top to bottom 

---

## Models Overview

### Custom CNN (Model 1)
Light weight CNN with 3 convolutional layers and a fully connected head 
- Peak Accuracy: 97.78%
- Final Accuracy (Seeded Run): 95.56%

### ResNet18 Transfer Learning (Model 2)
ImageNet-pretrained ResNet18 with all its convolutional layers frozen
- Peak Accuracy: 98.33%
- Final Accuracy (Seeded Run): 98.33%

---

## Results Summary 

| Model       | Peak Accuracy | Final Accuracy |
|-------------|---------------|----------------|
| Custom CNN  | 97.78%        | 95.56%         |
| ResNet18 TL | 98.33%        | 98.33%         |

---

## Reproducability
All notebooks use deterministic seeding to stabilize results

```
import random, numpy as np, torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Data Preparation Notes
The original dataset provides:

- Large parking-lot images

- XML annotations containing polygon coordinates describing each parking space

In 01_data_prep.ipynb, polygons were converted into bounding boxes and cropped into individual parking-space images.

The final dataset contains 903 cropped parking-space samples, organized by label.

---

## Author

Name: Han Htoo Zin

CECS 456 – Fall 2025

California State University, Long Beach

---

## Acknowledgments 
- Dataset: Parking Space Detection Dataset (TrainingDataPro, Kaggle)
- Pretrained Model: ResNet18 from PyTorch
- Tools: Google Colab (T4 GPU)

END of README.md
