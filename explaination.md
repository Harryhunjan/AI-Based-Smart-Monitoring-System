# AI-Based Smart Monitoring System — Complete Project Explanation

> **Purpose of this document:** Provide a comprehensive, self-contained explanation of the entire project so that any AI assistant, developer, or reviewer can fully understand the system's architecture, data flow, algorithms, configuration, and file responsibilities without needing to read the source code line by line.

---

## 1. Project Overview

This project is a **real-time AI-powered surveillance and smart monitoring desktop application** built entirely in Python. It uses a webcam or pre-recorded video file as input, detects and tracks people and personal belongings in real time, recognizes known individuals by their faces, links personal objects (bags, laptops, phones, etc.) to their owners, and raises alerts when an item is left unattended. All events are logged to a CSV file and can be visualized through embedded analytics charts.

The system is designed as a **single-machine, single-camera desktop application** — not a web app or cloud service. It runs on Windows (primary target) and can also run on Linux/macOS with Python 3.8+.

### Core Capabilities

1. **Person Detection** — YOLOv8 nano detects people in every video frame.
2. **Multi-Object Tracking** — ByteTrack (embedded in Ultralytics YOLO) assigns persistent integer IDs to each detected person and object across frames, surviving brief occlusions.
3. **Face Recognition** — InsightFace's ArcFace model extracts 512-dimensional facial embeddings and matches them against a database of registered identities using cosine similarity.
4. **Object Ownership Linking** — Detected personal belongings are linked to the nearest person using bounding box overlap (IoOA) and Euclidean distance proximity.
5. **Abandoned Object Detection** — If a linked object stays stationary for 30+ frames and its owner moves away or leaves the frame, an `ITEM_LEFT_BEHIND` alert is triggered.
6. **Event Logging** — Every `ENTRY`, `EXIT`, `IDENTIFIED`, and `ITEM_LEFT_BEHIND` event is appended to a structured CSV log file.
7. **Analytics Dashboard** — Matplotlib charts (bar chart of event types, pie chart of person frequencies) are rendered directly inside the GUI.
8. **In-App Face Registration** — New identities can be registered via the GUI by capturing a live frame or uploading an image file.

---

## 2. Technology Stack

| Layer | Technology | Role |
|---|---|---|
| **Language** | Python 3.8+ | All code is Python |
| **GUI Framework** | CustomTkinter 5.x | Dark-themed desktop GUI with tabs, sliders, checkboxes |
| **Object Detection** | YOLOv8 nano (`yolov8n.pt`, Ultralytics) | Single-stage object detector; detects people + 6 object classes |
| **Multi-Object Tracking** | ByteTrack | Built into YOLO via `model.track(persist=True)` — assigns persistent IDs |
| **Face Recognition** | InsightFace `buffalo_l` (ArcFace) | 512-dim embedding extraction + cosine similarity matching |
| **Face Pre-Filter** | OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) | Quick frontal face localization within person bounding box |
| **Video Capture** | imutils `VideoStream` (webcam) / OpenCV `VideoCapture` (file) | Threaded frame acquisition |
| **Computer Vision** | OpenCV (`cv2`) | Drawing, color conversion, image I/O, resizing |
| **Embedding Cache** | Python `pickle` (`.pkl` file) | Serialized dict of `{person_name: averaged_embedding_vector}` |
| **Event Logging** | Pandas + CSV | Append-mode structured event log |
| **Analytics** | Matplotlib + Seaborn (embedded in Tkinter via `FigureCanvasTkAgg`) | In-app and standalone chart generation |
| **Image Processing** | Pillow (PIL) | Frame conversion for Tkinter display |
| **Numerical** | NumPy | Array operations, cosine similarity, embedding math |
| **ML Runtime** | ONNX Runtime | Backend for InsightFace model inference |

---

## 3. Project File Structure and File Responsibilities

```
Project Root/
│
├── smart_monitoring_gui.py          # PRIMARY ENTRY POINT — Full GUI application
├── real_time_object_detection.py    # SECONDARY ENTRY — Standalone CLI-only version (legacy)
├── face_recognizer.py               # Face recognition engine (InsightFace wrapper)
├── logger.py                        # Event logger (CSV append)
├── config.py                        # Tunable face recognition parameters
├── analyze_metrics.py               # Standalone post-session analytics script
├── face_preprocessing.py            # LFW dataset preprocessor (research/academic only)
├── check_db.py                      # CLI tool to inspect face embedding cache
├── build_cache.py                   # CLI tool to rebuild face embedding cache
├── requirements.txt                 # Python package dependencies
├── yolov8n.pt                       # YOLOv8 nano model weights (git-ignored)
├── .gitignore                       # Git ignore rules
│
├── databases/                       # Face registration database
│   ├── Hargun Hunjan/               # One subfolder per registered person
│   │   ├── 1.jpg                    # Face images of that person
│   │   └── 2.jpg
│   ├── Kashish Kharb/
│   ├── Prajjwal Kumar/
│   ├── Ramneet Singh/
│   ├── Simar Singh Nayyar/
│   ├── Smriti khanor/
│   ├── Vivan Sharma/
│   ├── Yashasvi/
│   └── known_faces_cache.pkl        # Auto-generated averaged ArcFace embeddings
│
├── logs/                            # Runtime event logs
│   └── monitoring_events.csv        # Append-mode structured event log
│
├── output/                          # Saved processed video files with overlays
├── images/                          # Reference screenshots
└── Video/                           # Sample input video files
```

### Detailed File Descriptions

#### `smart_monitoring_gui.py` (1572 lines) — Main Application

This is the **primary entry point** and the largest file. It contains:

- **`SmartMonitoringApp`** class (extends `ctk.CTk`) — The main GUI window with:
  - **Sidebar** (left): System status label, Start/Stop buttons, Camera/Video File toggle, YOLO confidence slider, face similarity threshold slider, per-class tracking checkboxes.
  - **Tabview** (right): Three tabs:
    1. **Live Feed** — Video display panel, face recognition warm-up progress bar, horizontal metrics bar (FPS, people count, known/unknown), scrolling event log console.
    2. **Register Face** — Name entry field, capture-from-stream button, upload-image button, list of registered identities with delete buttons.
    3. **Analytics** — Refresh button, embedded Matplotlib charts (event distribution bar chart, person frequency pie chart).

- **`ModelLoaderThread`** class — Background daemon thread that imports `ultralytics`, loads the YOLO model (`yolov8n.pt`), and instantiates the `EventLogger`. Runs on app startup to avoid freezing the GUI.

- **`FaceRecognizer`** class — Persistent daemon thread with a task queue. Imports `face_recognizer.py`, initializes InsightFace, and processes face matching requests sequentially. Communicates via `task_queue` (input) and `result_queue` (output).

- **`VideoWorker`** class — Background daemon thread that runs the main detection loop:
  1. Captures frames from webcam (`imutils.VideoStream`) or video file (`cv2.VideoCapture`).
  2. Runs `yolo_model.track()` with ByteTrack persistence.
  3. Processes detected persons (registers new tracks, submits face crops to `FaceRecognizer`).
  4. Processes detected objects (links to nearest person, checks stationarity).
  5. Detects abandoned objects.
  6. Draws bounding boxes, labels, and overlays onto the frame.
  7. Packages the annotated frame + stats into `frame_queue` for the GUI to display.
  8. Cleans up stale tracks (EXIT detection).

- **`migrate_database_folders()`** — Startup utility that flattens `databases/train/` and `databases/validation/` subdirectories into `databases/` (migration from an older folder structure).

- **`on_closing()`** — Graceful shutdown handler.

#### `face_recognizer.py` (250 lines) — Face Recognition Engine

This module wraps InsightFace's `FaceAnalysis` (`buffalo_l` model package, which includes both a face detector and the ArcFace embedding model):

- **`initialize_model()`** — Lazy-loads the InsightFace model. Attempts GPU (`CUDAExecutionProvider`) first, falls back to CPU. Thread-safe via a lock and double-check pattern.

- **`load_known_faces(force_rebuild=False)`** — Loads the embedding cache from `known_faces_cache.pkl`. If the cache is missing, corrupt, or `force_rebuild=True`, it:
  1. Scans every person subfolder in `databases/`.
  2. Reads each image file.
  3. Runs InsightFace face detection + ArcFace embedding extraction.
  4. Averages all embeddings per person to create a single robust reference vector.
  5. Saves the result to the pickle cache.

- **`cosine_similarity(emb1, emb2)`** — Computes cosine similarity between two 512-dim vectors.

- **`recognize_face(roi, track_id)`** — The core recognition function:
  1. Applies per-track cooldown rate-limiting (`COOLDOWN_TIME = 2.0s`).
  2. Validates the input crop dimensions.
  3. Runs InsightFace face detection on the crop.
  4. Selects the largest detected face.
  5. Checks face detection confidence against `DETECTION_THRESHOLD`.
  6. Extracts the ArcFace embedding.
  7. Compares against all known face embeddings using cosine similarity.
  8. If the best match exceeds `FACE_MATCH_THRESHOLD`, returns the matched name; otherwise "Unknown".

- **`try_recognize()`** — Async wrapper that spawns a daemon thread for recognition (used by the CLI version, not the GUI).

**Global State:**
- `app` — InsightFace FaceAnalysis model instance (singleton).
- `known_faces` — Dict of `{display_name: np.ndarray}` (512-dim embeddings).
- `last_attempt_time` — Dict of `{track_id: timestamp}` for cooldown.
- `last_match_result` — Dict of `{track_id: (name, similarity)}` for caching recent results.

#### `config.py` (10 lines) — Configuration Constants

```python
FACE_MATCH_THRESHOLD = 0.55   # Cosine similarity threshold for positive match
DEFAULT_DB_PATH = "databases"  # Root directory for registered face folders
MODEL_NAME = "buffalo_l"       # InsightFace model package name
DETECTION_THRESHOLD = 0.5     # Minimum face detection confidence
COOLDOWN_TIME = 2.0           # Seconds between recognition attempts per track ID
CACHE_FILE = "databases/known_faces_cache.pkl"  # Path to embedding cache
```

#### `logger.py` (32 lines) — Event Logger

- **`EventLogger`** class:
  - Creates `logs/` directory and `monitoring_events.csv` (with header) if they don't exist.
  - **`log_event(event_type, person_name, item_class, confidence)`** — Appends a timestamped row to the CSV in append mode using Pandas.
  - CSV columns: `Timestamp, Event, Person_Name, Item_Class, Confidence`.

#### `real_time_object_detection.py` (341 lines) — Standalone CLI Version

A legacy/alternative entry point that does the same detection + tracking + face recognition pipeline but renders output via a raw OpenCV window (`cv2.imshow`) instead of the CustomTkinter GUI. Features a transparent stats overlay panel and event log overlay. Does **not** support video file input, in-app face registration, or embedded analytics.

#### `analyze_metrics.py` (74 lines) — Standalone Analytics

Reads `logs/monitoring_events.csv` and generates three charts:
1. Bar chart of event type distribution.
2. Pie chart of person tracking frequency.
3. Bar chart of item types left behind.

Saves chart images to `logs/` and displays them via `plt.show()`.

#### `face_preprocessing.py` (210 lines) — LFW Dataset Preprocessor

Research-only utility for preprocessing the LFW-DeepFunneled face dataset. Reads raw images, resizes to 224×224, normalizes to [0,1], and saves to a structured output directory. Not required for normal system operation.

#### `check_db.py` (25 lines) — Cache Inspector

CLI tool that loads `known_faces_cache.pkl` and prints each registered person's name, embedding shape, and vector norm. Used for debugging.

#### `build_cache.py` (14 lines) — Cache Rebuilder

CLI tool that calls `face_recognizer.load_known_faces(force_rebuild=True)` to regenerate the embedding cache from scratch.

---  
## 4. System Architecture and Threading Model

The application uses **4 threads** in the GUI version:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAIN THREAD (Tkinter Event Loop)                     │
│                                                                              │
│  SmartMonitoringApp (CTk)                                                    │
│  ├── Renders GUI: sidebar, tabs, video display, metrics, event log           │
│  ├── poll_camera_queue()  → every 15ms, drains frame_queue, updates display  │
│  ├── poll_log_queue()     → every 100ms, drains log_queue, updates console   │
│  ├── User interactions: sliders, buttons, checkboxes → update VideoWorker    │
│  └── Face registration: capture/upload → save image → clear cache            │
│                                                                              │
│  Queues:                                                                     │
│  ├── frame_queue (maxsize=2)   VideoWorker → Main Thread                     │
│  ├── log_queue (unbounded)     VideoWorker → Main Thread                     │
│  ├── face_task_queue           VideoWorker → FaceRecognizer                  │
│  └── face_result_queue         FaceRecognizer → VideoWorker                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  THREAD 1: ModelLoaderThread (daemon, one-shot)   │
│  • Imports ultralytics, loads YOLO model           │
│  • Imports logger, creates EventLogger instance    │
│  • Fires callback to Main Thread when done         │
│  • Thread exits after loading completes            │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│  THREAD 2: FaceRecognizer (daemon, persistent)    │
│  • Imports face_recognizer module                  │
│  • Initializes InsightFace (buffalo_l model)       │
│  • Loads known face embeddings from cache          │
│  • Runs infinite loop:                             │
│    - Waits on face_task_queue (1s timeout)         │
│    - Receives (face_image, track_id, threshold)    │
│    - Calls face_recognizer.recognize_face()        │
│    - Puts (track_id, matched_path, distance)       │
│      onto face_result_queue                        │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│  THREAD 3: VideoWorker (daemon, persistent)       │
│  • Runs infinite outer loop (waits for running=T)  │
│  • When started:                                   │
│    - Opens webcam or video file                    │
│    - Runs detection loop per frame:                │
│      1. Check face_result_queue for completed IDs  │
│      2. Read frame from source                     │
│      3. Run YOLO tracking inference                │
│      4. Process persons (register, submit faces)   │
│      5. Process objects (link, stationarity check) │
│      6. Detect abandoned objects                   │
│      7. Draw overlays and bounding boxes           │
│      8. Package frame+stats → frame_queue          │
│      9. Package events → log_queue                 │
│     10. Clean up stale tracks                      │
│    - On stop: release capture, notify GUI          │
└──────────────────────────────────────────────────┘
```

### Startup Sequence

1. `migrate_database_folders()` runs to flatten legacy folder structure.
2. `SmartMonitoringApp()` constructor builds the entire GUI layout.
3. `ModelLoaderThread` starts and loads YOLO + Logger in the background.
4. Two polling loops (`poll_camera_queue` and `poll_log_queue`) begin via `self.after()`.
5. When `ModelLoaderThread` completes, its callback:
   - Enables the Start button.
   - Creates `face_task_queue` and `face_result_queue`.
   - Starts the `FaceRecognizer` thread (which imports InsightFace).
   - Starts the `VideoWorker` thread (but does NOT begin streaming).
6. User clicks "Start Camera Feed" → `VideoWorker.start_stream()` is called → detection loop begins.

### Data Flow (Per Frame)

```
Camera/File → VideoWorker reads frame
         ↓
    YOLO model.track(frame, persist=True)
         ↓
    Returns: bounding boxes, track IDs, class IDs, confidences
         ↓
    ┌─ For each person (class 0):
    │   • Register new track or update existing
    │   • Log ENTRY event for new persons
    │   • If face not yet identified AND not already queued:
    │   │   1. Extract person ROI from frame
    │   │   2. Try Haar Cascade face detection on ROI
    │   │   3. If no face found, use upper-35% fallback crop
    │   │   4. Resize face crop to 224×224
    │   │   5. Submit (face_image, track_id) to face_task_queue
    │   │      → FaceRecognizer thread picks it up
    │   │      → Runs InsightFace detection + ArcFace embedding
    │   │      → Compares against known_faces via cosine similarity
    │   │      → Puts (track_id, match_path, distance) in face_result_queue
    │   │      → VideoWorker reads result on next frame iteration
    │   │      → Updates person name from "Unknown" to matched name
    │   • Draw green box (known) or red box (unknown) + label
    │
    ├─ For each object (backpack, handbag, suitcase, bottle, laptop, phone):
    │   • Register new track or update existing
    │   • Track position history for stationarity detection
    │   • Link to nearest person via IoOA or distance proximity
    │   • If stationary + owner far/gone → ITEM_LEFT_BEHIND alert (red box)
    │   • Draw bounding box + owner label
    │
    ├─ Clean up stale tracks:
    │   • Person not seen for 2.0s → log EXIT, remove from tracked_persons
    │   • Object not seen for 4.0s → remove from tracked_objects
    │
    └─ Package frame + stats → frame_queue → Main Thread renders in GUI
```

---

## 5. Key Algorithms Explained

### 5.1 Face Recognition Pipeline

1. **Face Extraction from Person Bounding Box:**
   - YOLOv8 detects the person → provides a bounding box crop.
   - OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) attempts to localize a frontal face within that person crop. The largest detected face is selected.
   - **Fallback:** If Haar Cascade finds no face, the upper 35% height × 80% centered width of the person bounding box is used as a rough face crop.
   - The extracted face region is resized to 224×224 pixels.

2. **Face Embedding Extraction:**
   - InsightFace's `FaceAnalysis` (model package `buffalo_l`) runs its own internal face detector + alignment on the 224×224 crop.
   - The largest aligned face is selected.
   - If the face detection confidence (`det_score`) is below `DETECTION_THRESHOLD` (0.5), the crop is rejected.
   - ArcFace generates a 512-dimensional L2-normalized embedding vector.

3. **Identity Matching:**
   - The probe embedding is compared against every registered identity's averaged embedding using **cosine similarity**.
   - The identity with the highest cosine similarity is selected.
   - If `best_similarity >= FACE_MATCH_THRESHOLD` (default 0.55), the person is identified.
   - Otherwise, they remain "Unknown".

4. **Cooldown Rate-Limiting:**
   - Each `track_id` has a cooldown timer (`COOLDOWN_TIME = 2.0s`).
   - Recognition is not re-attempted on the same track until the cooldown expires.
   - This prevents redundant GPU/CPU work on the same person every frame.

5. **Embedding Cache (`known_faces_cache.pkl`):**
   - On first load (or rebuild), every image in every `databases/<PersonName>/` subfolder is processed.
   - All embedding vectors for a person are averaged into a single reference vector.
   - The dict `{display_name: averaged_embedding}` is serialized to a pickle file.
   - Subsequent loads read directly from the pickle cache (fast).

### 5.2 Object Ownership Linking

Objects are linked to people using a two-stage heuristic:

1. **IoOA (Intersection over Object Area):**
   - For each object, compute the intersection area with every person's bounding box.
   - `IoOA = intersection_area / object_area`
   - If any person has `IoOA > 0.3` (30% of the object's area overlaps with the person), link that object to that person.

2. **Euclidean Distance Fallback:**
   - If no person achieves IoOA > 0.3, find the closest person by Euclidean distance between bounding box centers.
   - If the distance is less than `1.2 × max(object_width, person_width)`, link the object to that person.
   - Otherwise, the object remains unlinked (unless it's stationary, in which case the previous link is preserved).

### 5.3 Abandoned Object Detection

1. **Stationarity Check:**
   - Each tracked object maintains a position history (center coordinates) of the last 30 frames.
   - If the maximum displacement across all 30 positions is less than 20 pixels (`STATIONARY_THRESH_PX`), the object is marked as **stationary**.

2. **Owner Separation Check:**
   - If an object is stationary AND has a linked owner:
     - If the owner has not been seen for 2+ seconds (`DISAPPEAR_TIMEOUT`), → **abandoned**.
     - If the owner is visible but the distance from owner's center to the object exceeds `(owner_width + 200px)`, → **abandoned**.

3. **Alert:**
   - The bounding box turns red.
   - An `ITEM_LEFT_BEHIND` event is logged (once per stationary period, via `lost_alert_logged` flag).
   - If the object starts moving again, the `lost_alert_logged` flag resets.

### 5.4 Entry / Exit Detection

- **ENTRY:** Logged when a new ByteTrack ID appears (first time a person is detected).
- **EXIT:** Logged when a tracked person has not been seen in any frame for `DISAPPEAR_TIMEOUT` (2.0 seconds). The person's record is then removed from the tracking dictionary.
- **Limitation:** This is purely disappearance-based, not direction-aware. A person walking behind an obstacle briefly may trigger a false EXIT + re-ENTRY. A virtual trip-line approach is planned but not yet implemented.

---

## 6. COCO Object Classes Tracked

The system tracks these 7 COCO classes (configured via `TARGET_CLASSES`):

| COCO ID | Class Name | Purpose |
|---|---|---|
| 0 | person | Primary subject for tracking and face recognition |
| 24 | backpack | Personal belonging — linked to owner |
| 26 | handbag | Personal belonging — linked to owner |
| 28 | suitcase | Personal belonging — linked to owner |
| 39 | bottle | Personal belonging — linked to owner |
| 63 | laptop | Personal belonging — linked to owner |
| 67 | cell phone | Personal belonging — linked to owner |

Each class can be individually toggled on/off via the sidebar checkboxes in the GUI.

---

## 7. Event Log Format

Events are stored in `logs/monitoring_events.csv` with append-mode writes:

| Column | Type | Description |
|---|---|---|
| Timestamp | string | `YYYY-MM-DD HH:MM:SS` |
| Event | string | One of: `ENTRY`, `EXIT`, `IDENTIFIED`, `ITEM_LEFT_BEHIND` |
| Person_Name | string | Recognized name or `"Unknown"` |
| Item_Class | string | Object class name (e.g., `backpack`) or `"None"` |
| Confidence | string | Usually empty (legacy column) |

Example rows:
```
2025-06-01 10:23:11,ENTRY,Unknown,None,
2025-06-01 10:23:30,IDENTIFIED,Hargun Hunjan,Backpack,
2025-06-01 10:23:45,ITEM_LEFT_BEHIND,Hargun Hunjan,backpack,
2025-06-01 10:24:02,EXIT,Hargun Hunjan,None,
```

---

## 8. GUI Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   SmartMonitoringApp (CTk)                       │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐ │
│  │   SIDEBAR (280px) │  │  CTkTabview                          │ │
│  │                    │  │                                      │ │
│  │  Title: "SMART     │  │  ┌──────────┬─────────────┬────────┐│ │
│  │   MONITOR"         │  │  │Live Feed │Register Face│Analytic││ │
│  │                    │  │  │          │             │s       ││ │
│  │  Status: [label]   │  │  │ Video    │ Name Entry  │ Charts ││ │
│  │                    │  │  │ Display  │ Capture Btn │        ││ │
│  │  [Start] button    │  │  │          │ Upload Btn  │        ││ │
│  │  [Stop] button     │  │  │ Warm-up  │ Reg. List   │        ││ │
│  │                    │  │  │ Bar      │ (scrollable)│        ││ │
│  │  Source: Camera/   │  │  │          │             │        ││ │
│  │    Video File      │  │  │ Metrics  │             │        ││ │
│  │  [Browse Video]    │  │  │ Bar      │             │        ││ │
│  │                    │  │  │          │             │        ││ │
│  │  YOLO Conf: [====] │  │  │ Event    │             │        ││ │
│  │  Face Sim:  [====] │  │  │ Log      │             │        ││ │
│  │                    │  │  │ Console  │             │        ││ │
│  │  ☑ Person          │  │  └──────────┴─────────────┴────────┘│ │
│  │  ☑ Backpack        │  │                                      │ │
│  │  ☑ Handbag         │  └──────────────────────────────────────┘ │
│  │  ☑ Suitcase        │                                          │
│  │  ☑ Bottle          │                                          │
│  │  ☑ Laptop          │                                          │
│  │  ☑ Cell Phone      │                                          │
│  │                    │                                          │
│  │  Powered by YOLOv8 │                                          │
│  │  & InsightFace     │                                          │
│  └──────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
```

### GUI Polling Mechanism

- **`poll_camera_queue()`** — Runs every 15ms via `self.after()`. Drains `frame_queue`, takes only the latest frame (discarding older ones to prevent lag), resizes the PIL image to fit the display label while maintaining aspect ratio, and converts it to a `CTkImage` for rendering. Also updates the metrics bar labels and controls the face recognition warm-up progress bar visibility.

- **`poll_log_queue()`** — Runs every 100ms via `self.after()`. Drains `log_queue` and appends each event message to the scrolling `CTkTextbox` console, prefixed with `[ALERT]` (for unknown/warning events) or `[LOG]` (for normal events).

### Face Recognition Warm-Up Indicator

- When the first face recognition task is submitted in a session, `face_rec_warming_up = True`.
- This causes an indeterminate `CTkProgressBar` to appear below the video display with the label "Face Recognition Initializing...".
- Once the first result comes back from the `FaceRecognizer` thread, the progress bar is hidden.
- This provides UX feedback during the ~2-5 second InsightFace cold-start.

---

## 9. Configuration and Tunable Parameters

### Static Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `FACE_MATCH_THRESHOLD` | 0.55 | Minimum cosine similarity for a positive face match. Higher = stricter. |
| `DEFAULT_DB_PATH` | `"databases"` | Directory containing person subfolders with face images. |
| `MODEL_NAME` | `"buffalo_l"` | InsightFace model package. Includes face detection + ArcFace. |
| `DETECTION_THRESHOLD` | 0.5 | Minimum InsightFace face detection confidence to proceed with recognition. |
| `COOLDOWN_TIME` | 2.0 | Seconds between recognition attempts on the same track ID. |
| `CACHE_FILE` | `"databases/known_faces_cache.pkl"` | Path to the serialized embedding cache. |

### Runtime Parameters (GUI Sidebar)

| Parameter | Default | Range | GUI Control |
|---|---|---|---|
| YOLO confidence threshold | 0.25 | 0.05 – 0.95 | Slider (18 steps) |
| Face similarity threshold | 0.45 | 0.10 – 0.80 | Slider (14 steps) |
| Tracked object classes | All 7 enabled | Per-class on/off | Checkboxes |

**Note:** The GUI uses `dist_threshold = 0.45` as the face similarity threshold, which is separate from `config.FACE_MATCH_THRESHOLD = 0.55`. The `dist_threshold` in the GUI controls whether the GUI accepts a match result (comparing `1.0 - similarity` against the threshold), while `FACE_MATCH_THRESHOLD` in `face_recognizer.py` controls whether the recognition engine itself returns a match. Both must be satisfied for a face to be displayed as identified.

### Hardcoded Constants (in detection loops)

| Constant | Value | Description |
|---|---|---|
| `STATIONARY_THRESH_PX` | 20 px | Max pixel movement for an object to be considered stationary. |
| `STATIONARY_FRAMES_REQUIRED` | 30 frames | Number of consecutive frames an object must be stationary. |
| `DISAPPEAR_TIMEOUT` | 2.0 s | Seconds before a missing person is logged as EXIT. |
| `imgsz` | 320 px | YOLO inference resolution (smaller = faster, less accurate). |

---

## 10. Face Registration Workflow

### Via GUI (Capture from Stream)

1. User enters a name (e.g., "John Doe") in the Register Face tab.
2. User clicks "Capture Face (From Stream)" while the camera is active.
3. The system captures the current raw frame.
4. Haar Cascade attempts face detection on the full frame.
5. If a face is found, the largest face is cropped with 15% padding and resized to 224×224.
6. If no face is found, the raw full frame is saved (with a warning).
7. The image is saved to `databases/john_doe/<name>_face.jpg`.
8. The pickle cache is deleted and the in-memory `known_faces` dict is cleared.
9. On the next recognition attempt, `face_recognizer.py` will rebuild the cache.

### Via GUI (Upload Image)

1. User enters a name and clicks "Upload Image File".
2. A file dialog opens (filters: JPG, JPEG, PNG, BMP).
3. The selected image is read, face-cropped (if possible), and saved to the person's folder.
4. Cache is invalidated as above.

### Via CLI

1. Manually add images to `databases/<PersonName>/`.
2. Run `python build_cache.py` to rebuild the embedding cache.
3. Or run `python check_db.py` to inspect the current cache contents.

### Deleting a Registered Face

1. In the Register Face tab, click "Remove" next to a person's name.
2. The system stops the camera stream (to avoid file locks).
3. The entire person folder is deleted via `shutil.rmtree()`.
4. Cache is cleared and the face list is refreshed.
5. Camera stream is restarted if it was active.

---

## 11. Video File Processing

When the user selects "Video File" mode:

1. The sidebar shows a "Browse Video..." button.
2. User selects a video file (MP4, AVI, MKV, MOV, WMV, FLV).
3. On start, `VideoWorker` opens the file with `cv2.VideoCapture`.
4. An output video writer (`cv2.VideoWriter`) is created at `output/<basename>_output_<timestamp>.mp4` with the same FPS and resolution as the input.
5. Every processed frame (with drawn bounding boxes and overlays) is written to the output video.
6. When the video ends, the output file is saved and the GUI shows "Video Ended".

---

## 12. Analytics System

### Embedded Analytics (GUI Tab)

- User clicks "Refresh Analytics Charts" in the Analytics tab.
- The system reads `logs/monitoring_events.csv` using Pandas.
- Two Matplotlib subplots are rendered:
  1. **Bar Chart:** Count of each event type (ENTRY, EXIT, IDENTIFIED, ITEM_LEFT_BEHIND).
  2. **Pie Chart:** Frequency distribution of tracked person names.
- Charts use dark theme styling (`facecolor="#1a1a24"`) to match the GUI.
- Charts are embedded directly in the GUI via `FigureCanvasTkAgg`.

### Standalone Analytics (`analyze_metrics.py`)

- Generates three charts and saves them as PNG files in `logs/`:
  1. `event_distribution_chart.png`
  2. `person_frequency_pie.png`
  3. `items_left_behind_chart.png` (only if ITEM_LEFT_BEHIND events exist)

---

## 13. Dependencies and Installation

### Required Python Packages (`requirements.txt`)

```
numpy==2.4.4
imutils==0.5.4
opencv-python==4.13.0.92
ultralytics==8.4.39
insightface==1.0.1
onnxruntime==1.26.0
pandas==3.0.2
matplotlib==3.10.8
seaborn==0.13.2
customtkinter==5.2.2
pillow==12.2.0
```

### Optional: GPU Acceleration

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Required External Files

- `yolov8n.pt` — YOLOv8 nano model weights. Auto-downloaded by Ultralytics on first run.
- InsightFace `buffalo_l` model — Auto-downloaded by InsightFace on first initialization.

---

## 14. Inter-Module Communication Map

```
smart_monitoring_gui.py
  ├── imports: cv2, numpy, imutils, PIL, customtkinter, matplotlib, pandas, seaborn
  ├── imports at runtime (in threads): ultralytics.YOLO, logger.EventLogger, face_recognizer
  ├── uses: config.py (indirectly via face_recognizer)
  │
  ├── ModelLoaderThread
  │     └── loads: ultralytics.YOLO("yolov8n.pt") → global yolo_model
  │     └── loads: logger.EventLogger() → global logger_instance
  │
  ├── FaceRecognizer thread
  │     └── imports: face_recognizer module
  │     └── calls: face_recognizer.initialize_model()
  │     └── calls: face_recognizer.load_known_faces()
  │     └── calls: face_recognizer.recognize_face(roi, track_id)
  │
  └── VideoWorker thread
        └── uses: global yolo_model (YOLO inference)
        └── uses: global logger_instance (CSV logging)
        └── submits tasks to: FaceRecognizer via face_task_queue
        └── reads results from: FaceRecognizer via face_result_queue

face_recognizer.py
  ├── imports: insightface.app.FaceAnalysis
  ├── imports: config (for thresholds, paths, model name)
  ├── reads: databases/<PersonName>/*.jpg (face images)
  ├── reads/writes: databases/known_faces_cache.pkl

logger.py
  ├── imports: pandas
  ├── writes: logs/monitoring_events.csv

config.py
  ├── no imports beyond os
  ├── consumed by: face_recognizer.py

analyze_metrics.py
  ├── reads: logs/monitoring_events.csv
  ├── writes: logs/*.png (chart images)

real_time_object_detection.py (standalone alternative)
  ├── imports: face_recognizer, ultralytics.YOLO, logger.EventLogger
  ├── duplicates much of VideoWorker logic in a single-threaded OpenCV loop
```

---

## 15. Known Limitations

1. **Entry/Exit is camera-disappearance-based**, not direction-aware. A person occluded briefly may trigger false EXIT + re-ENTRY.
2. **ByteTrack ID reassignment** can cause duplicate ENTRY events for the same physical person after brief occlusions.
3. **Logger is not thread-safe.** CSV writes are synchronous via Pandas. Concurrent multi-camera writes would risk data corruption.
4. **Single camera pipeline.** Detection, tracking, recognition, and rendering all run in one `VideoWorker` thread. Typical throughput is ~5–12 FPS on CPU.
5. **Haar Cascade pre-filter may miss profile faces.** The upper-body fallback crop compensates but is less precise for InsightFace.
6. **`config.py` is not wired into the CLI version.** `real_time_object_detection.py` has some values hardcoded separately.
7. **No authentication or multi-user support.** The app is designed for local single-operator use.

---

## 16. Planned Future Improvements (Roadmap)

- [ ] Virtual trip-line entry/exit detection (direction-aware line crossing)
- [ ] Thread-safe logging with a queue-based writer
- [ ] Wire `config.py` into all detection loops
- [ ] Multi-camera support
- [ ] ByteTrack ID reassignment deduplication
- [ ] Alert sound notifications for abandoned objects
- [ ] Export analytics reports to PDF

---

## 17. How to Run

### GUI Dashboard (Recommended)
```bash
python smart_monitoring_gui.py
```

### Standalone CLI (Legacy)
```bash
python real_time_object_detection.py
```

### Rebuild Face Cache (CLI)
```bash
python build_cache.py
```

### Inspect Face Cache (CLI)
```bash
python check_db.py
```

### Generate Analytics Charts (CLI)
```bash
python analyze_metrics.py
```

---

## 18. Glossary

| Term | Definition |
|---|---|
| **ArcFace** | A face recognition loss function / model architecture that produces highly discriminative 512-dimensional facial embeddings. Used via InsightFace's `buffalo_l` package. |
| **ByteTrack** | A multi-object tracking algorithm that associates detections across frames using motion and appearance cues. Embedded in Ultralytics YOLO via `persist=True`. |
| **Cosine Similarity** | A measure of similarity between two vectors, computed as `dot(A,B) / (‖A‖ × ‖B‖)`. Range: -1 to 1. Higher = more similar. |
| **COCO** | Common Objects in Context — a large-scale object detection dataset. YOLOv8 is trained on COCO and uses COCO class IDs. |
| **Embedding** | A fixed-length numerical vector representation of a face. Two face embeddings from the same person will have high cosine similarity. |
| **IoOA** | Intersection over Object Area — ratio of the intersection area between an object box and a person box to the object's own area. Used for ownership linking. |
| **InsightFace** | An open-source face recognition library that provides face detection, alignment, and embedding extraction models. |
| **Track ID** | A persistent integer identifier assigned by ByteTrack to a detected entity across video frames. |
| **YOLOv8 Nano** | The smallest and fastest variant of the YOLOv8 family of single-stage object detectors. Model file: `yolov8n.pt`. |

---

*This document was generated to provide complete project context for AI assistants and developers. Last updated: July 2026.*
