# 👁️ AI Smart Surveillance & Object Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)
![DeepFace](https://img.shields.io/badge/DeepFace-Facenet-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

## 📌 Overview

This repository hosts an advanced **AI-powered Smart Monitoring System**. Initially starting as a basic object detection script, the project has evolved into a full-scale surveillance application that combines **YOLOv8** for real-time object/person detection and **DeepFace** for facial recognition.

It provides a sophisticated live-feed UI that tracks individuals, identifies known vs. unknown persons, links belongings to their owners, and flags when an item is left behind (abandoned object detection). It also includes a full pipeline for academic research using face preprocessing on the LFW dataset.

## 🚀 Key Features

* **Real-Time Object & Person Tracking**: Uses **YOLOv8** and internal trackers to track multiple people and objects simultaneously.
* **Facial Recognition (Known vs. Unknown)**: Integrates **DeepFace (Facenet)** to identify registered faces and explicitly highlights "Unknown" individuals.
* **Abandoned Object Detection**: Intelligently links personal items (e.g., backpacks, laptops, suitcases) to the person carrying them. Triggers a **LOST** alert if the person walks away and leaves the item behind.
* **Event Logging & Analytics**: Automatically logs events like `ENTRY`, `EXIT`, and `ITEM_LEFT_BEHIND` with timestamps and names. Includes data visualization scripts to generate insightful metrics.
* **Smart UI Dashboard**: An integrated OpenCV overlay showing rolling FPS, active tracking counts, recognized identities, and a live event log.
* **Dataset Preprocessing (Research Mode)**: Includes automated face extraction, normalization, and resizing utilities for the LFW dataset to train and evaluate recognition models.

## 🛠️ Technology Stack

* **Core Language**: Python 3.x
* **Computer Vision**: OpenCV (`cv2`), imutils
* **Object Detection & Tracking**: Ultralytics YOLO (`yolov8n.pt`)
* **Facial Recognition**: DeepFace (Facenet model, Cosine distance metric)
* **Data Processing & Analytics**: NumPy, Pandas, Matplotlib, Seaborn

## 📂 Project Structure

```text
Project/
│── real_time_object_detection.py  # Main entry script for the live surveillance feed
│── deep_learning_object_detection.py # Standard deep learning object detection code
│── face_preprocessing.py          # Preprocesses LFW dataset for facial recognition training
│── check_db.py                    # Script to verify and manage the face database
│── analyze_metrics.py             # Generates analytical graphs from system event logs
│── logger.py                      # Handles event logging (Entry, Exit, Left Item)
│── databases/                     # Cached representations for facial recognition
│── face_database/                 # Folder containing subfolders of known faces
│── dataset/                       # Output folder for preprocessed research datasets
│── logs/                          # System and preprocessing event logs and generated graphs
│── yolov8n.pt                     # Pre-trained YOLOv8 nano model
│── requirements.txt               # Python dependencies
│── RESEARCH_SETUP.md              # Research paper methodology and academic notes
```

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Harryhunjan/Object-Detection.git
   cd Object-Detection
   ```

2. **Install dependencies:**
   ```bash
   cd Project
   pip install -r requirements.txt
   pip install pandas matplotlib seaborn deepface ultralytics
   ```
   *(Ensure you have PyTorch installed with CUDA support if you intend to run inference on a GPU).*

3. **Prepare the Face Database:**
   * Add images of known individuals into the `Project/face_database/` directory. Create a subfolder for each person (e.g., `face_database/Harry/1.jpg`).
   * Run the DB check script to initialize embeddings:
     ```bash
     python check_db.py
     ```

## 🎯 Usage

### 1. Smart Surveillance Dashboard (Live)

To launch the real-time smart surveillance dashboard using your webcam:

```bash
cd Project
python real_time_object_detection.py
```

*Press **`q`** at any time to safely exit the video stream.*

### 2. Analytics & Graphs

After running the main script, the system will generate logs inside `logs/monitoring_events.csv`.
To visualize the tracked events (Entries, Exits, Abandoned Objects, Person Frequency):

```bash
python analyze_metrics.py
```
This will generate graph images in the `logs/` directory for reporting.

### 3. Face Preprocessing (For Research)

If you are using this project for an academic evaluation with the LFW Deep-Funneled dataset:



## 📊 Output

* Displays input with bounding boxes around detected objects
* Shows coordinates (x, y, width, height) of each detected object

## 📝 Future Enhancements

* Integration with **deep learning models (YOLO, SSD, Faster R-CNN)**
* Multi-object tracking
* Performance improvements for large-scale datasets
* GUI-based user interface

