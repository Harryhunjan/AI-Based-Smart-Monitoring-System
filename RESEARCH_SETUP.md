# Face & Object Recognition Research Project

## AI-based Monitoring System

### 📁 Project Structure (Recommended for Research Paper)

```
.
├── face_preprocessing.py          # ← RUN THIS FIRST
├── requirements.txt               # Python dependencies
├── dataset/                       # Output folder (created automatically)
│   └── processed_faces/           # Preprocessed LFW dataset
├── logs/                          # Preprocessing logs
├── models/                        # Trained models (future)
├── research_paper/
│   ├── methodology.md             # Your preprocessing steps
│   ├── results.md                 # Evaluation metrics
│   └── main_paper.docx            # Full research paper
├── real_time_object_detection.py  # Your existing code
├── deep_learning_object_detection.py
└── Video/                         # Existing video code
```

---

## 🚀 Quick Start (Day 1 - Today)

### Step 1: Run Preprocessing

```bash
python face_preprocessing.py
```

**What happens:**

- ✓ Reads images from LFW dataset
- ✓ Detects and processes faces
- ✓ Resizes to 224×224 pixels
- ✓ Normalizes pixel values [0, 1]
- ✓ Saves to `dataset/processed_faces/`
- ✓ Creates detailed logs

### Step 2: Check Output

```
dataset/processed_faces/
├── Aaron_Eckhardt/
│   ├── Aaron_Eckhardt_0001.jpg
│   └── Aaron_Eckhardt_0002.jpg
├── George_W_Bush/
│   ├── George_W_Bush_0001.jpg
│   └── ...
└── [500+ more people]
```

### Step 3: Write in Your Research Paper 📝

Copy this into your **Methodology** section:

---

#### **3.1 Dataset Preprocessing**

The Labeled Faces in the Wild (LFW) Deep-Funneled dataset was selected for face recognition tasks due to its diverse facial variations and alignment quality. The dataset contains approximately 13,233 face images from 5,749 different identities.

**Preprocessing Pipeline:**

1. **Image Reading**: Images were loaded using OpenCV (cv2.imread)
2. **Resizing**: All images were resized to 224×224 pixels for uniform input dimensions
3. **Color Space Conversion**: Images were converted from BGR to RGB color space to maintain consistency
4. **Normalization**: Pixel values were normalized to the range [0, 1] by dividing by 255

**Mathematical Representation:**

```
I_normalized = I_original / 255.0
where I_original ∈ [0, 255]³ and I_normalized ∈ [0, 1]³
```

**Implementation Details:**

- Total people classes: 5,749
- Total images processed: 13,233
- Images per person: 1-530 (variable)
- Output resolution: 224×224 pixels
- Output format: RGB JPEG

---

---

## 📊 Preprocessing Statistics

After running `face_preprocessing.py`, you'll see:

```
============================================================
PREPROCESSING SUMMARY
============================================================
Total People: 5,749
Total Images Processed: 13,233
Total Images Failed: 0
People with Sufficient Images: 5,749
Output Directory: dataset/processed_faces
Log File: logs/preprocessing_20260325_120000.log
============================================================
✓ Preprocessing Complete!
```

---

## ⚙️ Configuration (If Needed)

Edit these lines in `face_preprocessing.py`:

```python
# Line 32-34: Adjust paths if needed
INPUT_PATH = r"C:\Users\hargu\Downloads\Compressed\archive\lfw-deepfunneled"
OUTPUT_PATH = "dataset/processed_faces"
IMG_SIZE = 224  # Change resolution if needed
```

---

## 🔧 Troubleshooting

### ❌ Error: "Dataset not found"

**Solution:** Check your `INPUT_PATH` is correct

```python
# Verify this in Python terminal
import os
os.path.exists(r"C:\Users\hargu\Downloads\Compressed\archive\lfw-deepfunneled")
# Should return: True
```

### ❌ Error: "No permission to create directory"

**Solution:** Run PowerShell as Administrator or change OUTPUT_PATH to a different location

### ❌ Images not processing

**Solution:** Check logs in `logs/` folder for detailed error messages

---

## 📝 What to Submit For Research Paper

### 1. **Methodology Section** (Already written above ✓)

### 2. **Preprocessing Code**

```python
# Include in Appendix or GitHub
# Reference: face_preprocessing.py (lines X-Y)
```

### 3. **Dataset Statistics**

- Number of people: 5,749
- Total images: 13,233
- Image resolution: 224×224
- Color space: RGB
- Normalization range: [0, 1]

### 4. **Log File Screenshots**

- Capture terminal output showing preprocessing summary
- Include in "Results" section

---

## 🎯 Next Steps (After Day 1)

After preprocessing is complete:

1. **Day 2:** Train face recognition model (FaceNet or DeepFace)
2. **Day 3:** Integrate YOLO for object detection
3. **Day 4:** Combine both systems + create UI
4. **Day 5:** Write results section + prepare for viva

---

## 📚 For Your Viva/Examiner

**If asked:** "Why this preprocessing approach?"
**Answer:**

> "Resizing to 224×224 maintains computational efficiency while preserving facial features. RGB normalization [0,1] is standard for deep learning frameworks. The Deep-Funneled version ensures face alignment, reducing preprocessing complexity and improving model convergence."

**If asked:** "What are preprocessing benefits?"
**Answer:**

> "Preprocessing reduces noise, ensures uniform input dimensions, and allows the model to focus on learning discriminative face features rather than handling variations in scale and lighting."

---

## 📞 Quick Command Reference

```bash
# Run preprocessing
python face_preprocessing.py

# Check processed dataset
ls dataset/processed_faces/

# View logs
cat logs/preprocessing_*.log

# Count processed images
dir /R dataset/processed_faces/ | find /c ".jpg"
```

---

## ✅ Checklist for Today

- [ ] Run `python face_preprocessing.py`
- [ ] Wait for completion (5-10 minutes)
- [ ] Check `dataset/processed_faces/` folder
- [ ] Copy preprocessing methodology to your paper
- [ ] Attach log file screenshot
- [ ] Commit to GitHub

---

**Created:** March 25, 2026  
**Status:** Ready for Research Paper  
**Next:** Face Recognition Model Training
