# AI-Based Smart Monitoring System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![CustomTkinter](https://img.shields.io/badge/CustomTkinter-GUI-purple.svg)


A real-time AI surveillance system with a full desktop GUI dashboard. Detects and tracks people via webcam or video file, identifies known faces using ArcFace embeddings, links personal belongings to their owners, and fires alerts when an item is left unattended. All events are logged to a queryable CSV file with timestamps and can be visualized in an embedded analytics panel.

---

## Key features

- **Desktop GUI dashboard** — A modern dark-themed CustomTkinter application with tabbed views for Live Feed, Face Registration, and Analytics. Sidebar controls allow real-time adjustment of detection confidence, face similarity thresholds, and tracked object classes.
- **Dual input mode** — Supports both live webcam capture and pre-recorded video file playback (MP4, AVI, MKV, MOV). Video file output with overlays is automatically saved to the `output/` directory.
- **Person detection and tracking** — YOLOv8 nano detects people frame by frame. ByteTrack assigns each person a persistent ID across frames, even through brief occlusions.
- **Face recognition** — InsightFace (`buffalo_l` model) extracts 512-dimensional ArcFace embeddings. A person is matched against registered identities using cosine similarity. Recognition runs in a dedicated persistent background thread to avoid blocking the video loop.
- **In-app face registration** — Register new identities directly from the GUI, either by capturing a frame from the live stream or uploading an image file. Embedding caches are automatically invalidated and rebuilt.
- **Object ownership tracking** — Bags, laptops, bottles, phones, and suitcases are detected and linked to the nearest person using IoOA (Intersection over Object Area) with an Euclidean distance fallback.
- **Abandoned object detection** — If a tracked object stays stationary for 30 consecutive frames and its linked person moves away or leaves the frame, an `ITEM_LEFT_BEHIND` alert is logged and the bounding box turns red.
- **Event logging** — Every `ENTRY`, `EXIT`, `IDENTIFIED`, and `ITEM_LEFT_BEHIND` event is appended to `logs/monitoring_events.csv` with a timestamp, person name, and item class.
- **Embedded analytics** — The Analytics tab renders interactive Matplotlib charts (event type distribution bar chart, person tracking frequency pie chart) directly inside the dashboard without requiring a separate script.
- **Real-time metrics overlay** — A horizontal metrics bar displays live FPS, total people tracked, known identity count, and unknown/unverified count.
- **Face recognition warm-up indicator** — An indeterminate progress bar appears while InsightFace initializes on first recognition, disappearing once the first result returns.
- **Fullscreen mode** — Press `F11` to toggle fullscreen; `Escape` to exit.

---

## Technology stack

| Component | Technology | Details |
|---|---|---|
| GUI framework | CustomTkinter 5.x | Dark-themed desktop app with tabs, sliders, checkboxes |
| Object detection | YOLOv8 nano (`yolov8n.pt`) | Ultralytics single-stage detector |
| Multi-object tracking | ByteTrack | Embedded in YOLO via `persist=True` |
| Face recognition | InsightFace `buffalo_l` | ArcFace model, cosine similarity matching |
| Face extraction | Haar Cascade + fallback crop | Pre-filters person ROI before sending to InsightFace |
| Embedding cache | Pickle (`.pkl`) | Averaged embeddings per registered person |
| Frame capture | imutils `VideoStream` / OpenCV `VideoCapture` | Threaded webcam or file-based capture |
| Computer vision | OpenCV (`cv2`) | Drawing, display, frame ops, video writing |
| Event logging | Pandas + CSV | Append-mode, structured log |
| Analytics | Matplotlib + Seaborn (embedded in Tkinter) | In-app and standalone graph generation |
| Language | Python 3.8+ | |

---

## Project structure

```
.
├── smart_monitoring_gui.py         # Main GUI application (CustomTkinter dashboard)
├── real_time_object_detection.py   # Standalone CLI loop — detection, tracking, rendering
├── face_recognizer.py              # InsightFace / ArcFace recognition pipeline
├── logger.py                       # Event logger (CSV append)
├── config.py                       # Tunable face recognition parameters
├── analyze_metrics.py              # Standalone post-session analytics graphs
├── face_preprocessing.py           # LFW dataset preprocessing (research use)
├── check_db.py                     # Rebuild face embedding cache (CLI)
├── build_cache.py                  # Alternate cache builder
├── deep_learning_object_detection.py  # MobileNet SSD detection (legacy/reference)
├── MobileNetSSD_deploy.prototxt.txt   # MobileNet SSD architecture (legacy)
├── yolov8n.pt                      # YOLOv8 nano weights (git-ignored)
├── requirements.txt                # Python dependencies
├── RESEARCH_SETUP.md               # Academic evaluation methodology
├── databases/                      # Registered face images — one subfolder per person
│   ├── Hargun Hunjan/
│   │   ├── 1.jpg
│   │   └── 2.jpg
│   └── known_faces_cache.pkl       # Auto-generated ArcFace embedding cache
├── logs/                           # Event logs and generated graphs
│   └── monitoring_events.csv
├── output/                         # Processed video file outputs with overlays
├── images/                         # Reference screenshots and sample images
└── Video/                          # Sample input video files
```

---

## Configuration

### Face recognition parameters (`config.py`)

```python
FACE_MATCH_THRESHOLD = 0.55   # Cosine similarity threshold (0–1). Higher = stricter matching.
DETECTION_THRESHOLD  = 0.5    # Minimum InsightFace detection confidence to attempt recognition.
COOLDOWN_TIME        = 2.0    # Seconds before re-attempting recognition on the same track ID.
MODEL_NAME           = "buffalo_l"  # InsightFace model package (includes ArcFace + detection).
DEFAULT_DB_PATH      = "databases"  # Directory where registered face folders reside.
CACHE_FILE           = "databases/known_faces_cache.pkl"
```

### Runtime parameters (adjustable via GUI sidebar)

| Parameter | Default | GUI Control |
|---|---|---|
| YOLO confidence threshold | 0.25 | Slider (0.05–0.95) |
| Face similarity threshold | 0.45 | Slider (0.10–0.80) |
| Tracked object classes | All 7 | Per-class checkboxes |

### Hardcoded detection constants (in `smart_monitoring_gui.py` and `real_time_object_detection.py`)

```python
STATIONARY_THRESH_PX       = 20   # Max pixel movement for an object to be considered stationary.
STATIONARY_FRAMES_REQUIRED = 30   # Frames an object must be stationary before triggering alert.
DISAPPEAR_TIMEOUT          = 2.0  # Seconds before a missing person is logged as EXIT.
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Harryhunjan/AI-Based-Smart-Monitoring-System.git
cd AI-Based-Smart-Monitoring-System
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

Core dependencies:

| Package | Version |
|---|---|
| `ultralytics` | 8.4.39 |
| `insightface` | 1.0.1 |
| `onnxruntime` | 1.26.0 |
| `opencv-python` | 4.13.0.92 |
| `customtkinter` | 5.2.2 |
| `pillow` | 12.2.0 |
| `imutils` | 0.5.4 |
| `numpy` | 2.4.4 |
| `pandas` | 3.0.2 |
| `matplotlib` | 3.10.8 |
| `seaborn` | 0.13.2 |

For GPU acceleration, install PyTorch with CUDA support before running:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**4. Register known faces**

Add images of people you want to identify into `databases/`. Create one subfolder per person:

```
databases/
├── Harry/
│   ├── 1.jpg
│   └── 2.jpg
└── Alice/
    └── alice_front.jpg
```

Then build the embedding cache:

```bash
python check_db.py
```

This scans every image, extracts ArcFace embeddings, averages them per person, and saves the result to `databases/known_faces_cache.pkl`. Re-run this whenever you add or update registered faces via the CLI. Alternatively, use the **Register Face** tab in the GUI to add faces without touching the filesystem.

---

## Usage

### GUI dashboard (recommended)

```bash
python smart_monitoring_gui.py
```

This launches the full desktop application with three tabs:

1. **Live Feed** — Start/stop webcam or video file monitoring, view real-time detections with bounding boxes, metrics bar, and a scrolling event log console.
2. **Register Face** — Enter a person's name and either capture a frame from the live stream or upload an image file. The embedding cache is automatically rebuilt.
3. **Analytics** — Click "Refresh Analytics Charts" to render event distribution and person tracking frequency charts from the CSV log.

Use the sidebar to:
- Toggle between **Camera** and **Video File** input
- Adjust YOLO detection confidence and face similarity thresholds in real-time
- Enable/disable specific object classes to track

### Standalone CLI (legacy)

```bash
python real_time_object_detection.py
```

Opens a webcam feed with an OpenCV overlay dashboard. Press `q` to exit. Does not include GUI controls, face registration, or embedded analytics.

### Standalone analytics

```bash
python analyze_metrics.py
```

Reads `logs/monitoring_events.csv` and outputs graph images (event distribution, person frequency, lost items) to the `logs/` directory.

### LFW dataset preprocessing (research only)

```bash
python face_preprocessing.py
```

Preprocesses the LFW Deep-Funneled dataset for academic benchmarking. Not required for live system use. See `RESEARCH_SETUP.md` for methodology details.

---

## How face recognition works

1. When YOLOv8 detects a new person (new ByteTrack ID), their bounding box crop is extracted from the frame.
2. A Haar Cascade face detector attempts to locate a face within the person crop. If it fails, the upper 35% of the bounding box is used as a fallback.
3. The face ROI is resized to 224×224 and submitted to a persistent background `FaceRecognizer` thread via a task queue.
4. InsightFace (`buffalo_l`) runs face detection and alignment within the crop. The largest detected face is selected.
5. ArcFace generates a 512-dimensional embedding for that face.
6. The embedding is compared against all cached identity embeddings using cosine similarity.
7. If the best match score is above `FACE_MATCH_THRESHOLD` (default `0.55`), the person is identified by name. Otherwise they remain "Unknown".
8. A per-track cooldown (`COOLDOWN_TIME = 2.0s`) prevents redundant recognition calls on the same person.
9. GPU is used automatically if CUDA is available; falls back to CPU if not.

---

## Object ownership linking

Items are linked to people using a two-stage approach:

1. **IoOA (Intersection over Object Area)** — If the object's bounding box overlaps with a person's box by more than 30% of the object's area, they are linked.
2. **Euclidean proximity fallback** — If IoOA is too low, the closest person within 1.2× the bounding box dimension is linked as the owner.

When a linked object becomes stationary (moves less than 20 pixels over 30 frames) and the owner moves away or disappears, an `ITEM_LEFT_BEHIND` alert is triggered.

---

## Event log format

Events are appended to `logs/monitoring_events.csv`:

| Timestamp | Event | Person_Name | Item_Class | Confidence |
|---|---|---|---|---|
| 2025-06-01 10:23:11 | ENTRY | Unknown | None | |
| 2025-06-01 10:23:30 | IDENTIFIED | Harry | Backpack | |
| 2025-06-01 10:23:45 | ITEM_LEFT_BEHIND | Harry | backpack | |
| 2025-06-01 10:24:02 | EXIT | Harry | None | |

---

## Tracked object classes

The system detects the following COCO object classes alongside people:

| COCO ID | Class |
|---|---|
| 0 | person |
| 24 | backpack |
| 26 | handbag |
| 28 | suitcase |
| 39 | bottle |
| 63 | laptop |
| 67 | cell phone |

These can be individually toggled on/off via the sidebar checkboxes in the GUI.

---

## Application architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   SmartMonitoringApp (CTk)                   │
│  ┌──────────┐  ┌──────────────────────────────────────────┐ │
│  │ Sidebar  │  │  CTkTabview                              │ │
│  │          │  │  ┌──────────┬────────────┬────────────┐  │ │
│  │ Controls │  │  │Live Feed │Register    │ Analytics  │  │ │
│  │ Sliders  │  │  │          │Face        │            │  │ │
│  │ Classes  │  │  │ Video    │ Name entry │ Matplotlib │  │ │
│  │ Source   │  │  │ Display  │ Capture    │ Charts     │  │ │
│  │          │  │  │ Metrics  │ Upload     │            │  │ │
│  │          │  │  │ Event Log│ DB List    │            │  │ │
│  └──────────┘  │  └──────────┴────────────┴────────────┘  │ │
└─────────────────────────────────────────────────────────────┘

Background Threads:
  ├── ModelLoaderThread     → Loads YOLO + Logger on startup
  ├── FaceRecognizer        → Persistent InsightFace worker (task queue)
  └── VideoWorker           → Frame capture → YOLO → tracking → rendering
```

---

## Known limitations

- **Entry/exit logic is camera-angle dependent.** EXIT is triggered when a person disappears from frame for 2 seconds, not when they cross a physical door. A virtual trip-line approach is the correct fix and is planned.
- **ByteTrack ID reassignment can cause false entries.** Brief occlusions may cause the tracker to drop and reissue a new ID for the same person, logging a duplicate ENTRY.
- **Logger is not thread-safe.** The CSV is written synchronously. Concurrent reads or multi-camera setups will risk data corruption.
- **Single camera pipeline.** Detection, tracking, recognition, and rendering all run in a single VideoWorker thread. Effective throughput is ~5–12 FPS on CPU.
- **Haar Cascade pre-filter is approximate.** The frontal face cascade may miss profile faces; the 35% upper-body fallback crop compensates but is less precise.

---

## Roadmap

- [x] ~~CustomTkinter GUI dashboard with tabbed layout~~
- [x] ~~Video file input support with output recording~~
- [x] ~~In-app face registration (capture from stream + upload)~~
- [x] ~~Embedded Matplotlib analytics in the GUI~~
- [x] ~~Real-time slider controls for detection and recognition thresholds~~
- [x] ~~Per-class tracking toggles~~
- [x] ~~Face recognition warm-up progress indicator~~
- [x] ~~Fullscreen mode (F11)~~
- [ ] Virtual trip-line entry/exit detection (direction-aware line crossing)
- [ ] Thread-safe logging with a queue-based writer
- [ ] Wire `config.py` into the main detection loop (currently hardcoded in CLI mode)
- [ ] Multi-camera support
- [ ] ByteTrack ID reassignment deduplication
- [ ] Alert sound notifications for abandoned objects
- [ ] Export analytics reports to PDF
