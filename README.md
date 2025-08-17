# 📹 Anomaly Detection in Surveillance Videos

A deep learning-based system for **real-time detection of abnormal and criminal activities** in video streams.  
Leverages Convolutional Neural Networks to classify frames into multiple crime and anomaly categories, enabling automated surveillance, security monitoring, and incident alerting.

***

## 🚀 Introduction

This project identifies suspicious events in video feeds by evaluating each frame with a trained model.  
Predicted categories include: **Abuse, Robbery, Explosion, Shooting, Shoplifting, Vandalism, RoadAccidents, and more**.

- Accepts live or recorded video input.
- Labels each frame with the predicted event type.
- Visualizes and logs results for further analysis.

***

## 🌟 Features

- **Multi-class anomaly detection:** Recognizes 14 activity categories plus “Normal.”
- **Frame-level classification:** CNN model predicts incident per frame for responsiveness.
- **Real-time prediction and display:** Overlays labels on live video.
- **Model evaluation & analytics:** Confusion matrix, accuracy/loss curves, ROC, Precision-Recall, per-class accuracy.
- **Easy retraining and testing:** Modular scripts for training and inference.

***

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib, Seaborn**
- **scikit-learn** (for model evaluation)

***

## ⚡ Workflow Overview

> ### 🧩 How It Works
>
> | **Stage**                | **Action**                                             |
> |--------------------------|--------------------------------------------------------|
> | Data Preparation         | Organize training/testing images into class folders     |
> | Model Training           | Train CNN to classify video frame images               |
> | Model Evaluation         | Analyze performance with detailed metrics/graphs       |
> | Real-Time/Batch Inference| Process video frames, predict and annotate events      |
> | Visualization            | Display and log predicted category per frame           |

***

## 📁 Project Structure

> ### 📁 Project Structure
>
> ```
> ├── training.py                    # Model training, evaluation, analytics
> ├── cmaera.py                      # Real-time or batch video classification
> ├── Anomaly_detection_model.h5      # Saved trained CNN model
> ├── [dataset folders]               # Organized training/test data
> ├── README.md                       # Documentation
> ```

***

## 📝 Setup Instructions

### Prerequisites

- Python 3.x
- Required libraries
  ```bash
  pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn
  ```

### Dataset Preparation

- **Organize your data:**  
  Place images for each activity category in their respective folders under `Train` and `Test` directories.  
  Example:  
  ```
  D:/major/Train/Abuse/
  D:/major/Train/Robbery/
  D:/major/Test/Fighting/
  ```
  Each folder should only contain relevant images.

***

## 💻 Usage

### 1. Train the Model

Edit `training.py` paths to point to your dataset, then run:
```bash
python training.py
```
- Trains the CNN on labeled images
- Saves the model as `Anomaly_detection_model.h5`
- Displays evaluation plots and metrics

### 2. Video Crime Detection

Edit `cmaera.py` to set your test video path. Run:
```bash
python cmaera.py
```
- Loads the trained model
- Processes every frame of the video, predicts and overlays detected activity

***

## 📊 Evaluation Metrics & Visuals

- Training/Validation Accuracy and Loss Curves
- Confusion Matrix (raw and normalized)
- Classification Report (precision, recall, F1 for all classes)
- Precision-Recall and ROC Curves
- Per-class accuracy bar graph

***

## 🎯 Supported Event Categories

- Abuse
- Arrest
- Arson
- Assault
- Burglary
- Explosion
- Fighting
- RoadAccidents
- Robbery
- Shooting
- Shoplifting
- Stealing
- Vandalism
- Normal

***

## 🌍 Applications

- Smart surveillance and incident detection
- Crime monitoring and analytics
- Automated security alarm systems
- Forensic video analysis

***

## 🛠️ Future Enhancements

- Expand dataset for higher accuracy and more event types
- Integrate with live CCTV camera feeds
- Alert system for high-risk activities
- Export results as logs or reports

***

## 📬 Contact

Questions, feedback, or contributions? Contact [swastikdasoff@gmail.com]

***
