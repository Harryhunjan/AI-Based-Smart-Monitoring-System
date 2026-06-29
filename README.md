# AI-Based Smart Monitoring System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

A real-time AI surveillance system that detects and tracks people via webcam, identifies known faces using ArcFace embeddings, links personal belongings to their owners, and fires alerts when an item is left unattended. All events are logged to a queryable CSV file with timestamps.

---

## What it does

- **Person detection and tracking** — YOLOv8 nano detects people frame by frame. ByteTrack assigns each person a persistent ID across frames, even through brief occlusions.
- **Face recognition** — InsightFace (`buffalo_l` model) extracts 512-dimensional ArcFace embeddings. A person is matched against registered identities using cosine similarity. Recognition runs in a background thread to avoid blocking the video loop.
- **Object ownership tracking** — Bags, laptops, bottles, phones, and suitcases are detected and linked to the nearest person using Euclidean distance between bounding box centers.
- **Abandoned object detection** — If a tracked object stays stationary for 30 consecutive frames and its linked person moves away or leaves the frame, a `ITEM_LEFT_BEHIND` alert is logged and the bounding box turns red.
- **Event logging** — Every `ENTRY`, `EXIT`, and `ITEM_LEFT_BEHIND` event is appended to `logs/monitoring_events.csv` with a timestamp, person name, and item class.
- **Live dashboard overlay** — OpenCV renders a stats panel (people count, known/unknown split, FPS, timestamp) and a rolling event log directly on the video feed.
- **Analytics** — Run `analyze_metrics.py` after a session to generate event frequency graphs and timelines from the CSV log.

---

## Technology stack

| Component | Technology | Details |
|---|---|---|
| Object detection | YOLOv8 nano (`yolov8n.pt`) | Ultralytics single-stage detector |
| Multi-object tracking | ByteTrack | Embedded in YOLO via `persist=True` |
| Face recognition | InsightFace `buffalo_l` | ArcFace model, cosine similarity matching |
| Embedding cache | Pickle (`.pkl`) | Averaged embeddings per registered person |
| Frame capture | imutils `VideoStream` | Threaded webcam capture |
| Computer vision | OpenCV (`cv2`) | Drawing, display, frame ops |
| Event logging | Pandas + CSV | Append-mode, structured log |
| Analytics | Matplotlib + Seaborn | Post-session graph generation |
| Language | Python 3.8+ | |

---

## Project structure

```
.
├── real_time_object_detection.py   # Main loop — detection, tracking, rendering
├── face_recognizer.py              # InsightFace / ArcFace recognition pipeline
├── logger.py                       # Event logger (CSV append)
├── config.py                       # All tunable parameters
├── analyze_metrics.py              # Post-session analytics graphs
├── face_preprocessing.py           # LFW dataset preprocessing (research use)
├── check_db.py                     # Rebuild face embedding cache
├── requirements.txt                # Python dependencies
├── RESEARCH_SETUP.md               # Academic evaluation methodology
├── face_database/                  # Registered faces — one subfolder per person
│   └── Harry/
│       ├── 1.jpg
│       └── 2.jpg
├── databases/                      # Auto-generated ArcFace embedding cache
│   └── known_faces_cache.pkl
├── logs/                           # Event logs and generated graphs
│   └── monitoring_events.csv
└── Video/                          # Sample video files
```

---

## Configuration

All tunable values are in `config.py`:

```python
FACE_MATCH_THRESHOLD = 0.55   # Cosine similarity threshold (0–1). Higher = stricter matching.
DETECTION_THRESHOLD  = 0.5    # Minimum InsightFace detection confidence to attempt recognition.
COOLDOWN_TIME        = 2.0    # Seconds before re-attempting recognition on the same track ID.
MODEL_NAME           = "buffalo_l"  # InsightFace model package (includes ArcFace + detection).
DEFAULT_DB_PATH      = "databases"  # Directory where the embedding cache is stored.
CACHE_FILE           = "databases/known_faces_cache.pkl"
```

Values hardcoded in `real_time_object_detection.py` (not yet wired to config):

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

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

Core dependencies: `ultralytics`, `insightface`, `opencv-python`, `imutils`, `numpy`, `pandas`, `matplotlib`, `seaborn`

For GPU acceleration, install PyTorch with CUDA support before running:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**3. Register known faces**

Add images of people you want to identify into `face_database/`. Create one subfolder per person:

```
face_database/
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

This scans every image, extracts ArcFace embeddings, averages them per person, and saves the result to `databases/known_faces_cache.pkl`. Re-run this whenever you add or update registered faces.

---

## Usage

**Run the live surveillance feed**

```bash
python real_time_object_detection.py
```

Press `q` to exit cleanly.

**Generate analytics from a recorded session**

```bash
python analyze_metrics.py
```

Reads `logs/monitoring_events.csv` and outputs graphs to the `logs/` directory.

**LFW dataset preprocessing (research only)**

```bash
python face_preprocessing.py
```

Preprocesses the LFW Deep-Funneled dataset for academic benchmarking. Not required for live system use.

---

## How face recognition works

1. When YOLOv8 detects a new person (new ByteTrack ID), their bounding box crop is passed to a background daemon thread.
2. InsightFace (`buffalo_l`) runs face detection and alignment within the crop. The largest detected face is selected.
3. ArcFace generates a 512-dimensional embedding for that face.
4. The embedding is compared against all cached identity embeddings using cosine similarity.
5. If the best match score is above `FACE_MATCH_THRESHOLD` (default `0.55`), the person is identified by name. Otherwise they remain "Unknown".
6. A per-track cooldown (`COOLDOWN_TIME = 2.0s`) prevents redundant recognition calls on the same person.
7. GPU is used automatically if CUDA is available; falls back to CPU if not.

---

## Event log format

Events are appended to `logs/monitoring_events.csv`:

| Timestamp | Event | Person_Name | Item_Class | Confidence |
|---|---|---|---|---|
| 2025-06-01 10:23:11 | ENTRY | Harry | None | |
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

---

## Known limitations

- **Entry/exit logic is camera-angle dependent.** EXIT is triggered when a person disappears from frame for 2 seconds, not when they cross a physical door. A virtual trip-line approach is the correct fix and is planned.
- **ByteTrack ID reassignment can cause false entries.** Brief occlusions may cause the tracker to drop and reissue a new ID for the same person, logging a duplicate ENTRY.
- **Logger is not thread-safe.** The CSV is written synchronously on the main thread. Concurrent reads or multi-camera setups will risk data corruption.
- **Single-threaded pipeline.** Detection, tracking, recognition, and rendering all block each other. Effective throughput is ~5–12 FPS on CPU.

---

## Roadmap

- [ ] Virtual trip-line entry/exit detection (direction-aware line crossing)
- [ ] Thread-safe logging with a queue-based writer
- [ ] Wire `config.py` into the main detection loop (currently hardcoded)
- [ ] Multi-camera support
- [ ] ByteTrack ID reassignment deduplication

