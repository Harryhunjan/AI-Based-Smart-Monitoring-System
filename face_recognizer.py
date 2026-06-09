import os
import time
import pickle
import threading
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional, Callable
import config

# Global state for models and embeddings
app = None
known_faces: Dict[str, np.ndarray] = {}
lock = threading.Lock()

# Cooldown and result tracking to optimize frame-by-frame performance
last_attempt_time: Dict[int, float] = {}
last_match_result: Dict[int, Tuple[str, float]] = {}

def initialize_model() -> None:
    """
    Lazy-loads the InsightFace FaceAnalysis model.
    Tries GPU initialization first, falling back automatically to CPU if GPU is unavailable.
    """
    global app
    if app is not None:
        return

    with lock:
        # Double check to prevent race conditions during concurrent initialization
        if app is not None:
            return

        from insightface.app import FaceAnalysis

        print(f"[FACE] Initializing FaceAnalysis model (name='{config.MODEL_NAME}')...")
        try:
            # Attempt loading with GPU providers
            app = FaceAnalysis(name=config.MODEL_NAME, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0)
            print("[FACE] InsightFace FaceAnalysis model initialized successfully on GPU (ctx_id=0)")
        except Exception as e:
            print(f"[FACE] GPU initialization failed: {e}. Falling back to CPU...")
            try:
                app = FaceAnalysis(name=config.MODEL_NAME, providers=['CPUExecutionProvider'])
                app.prepare(ctx_id=-1)
                print("[FACE] InsightFace FaceAnalysis model initialized successfully on CPU (ctx_id=-1)")
            except Exception as cpu_error:
                print(f"[ERROR] Critical failure: Could not initialize InsightFace on CPU: {cpu_error}")
                raise

def load_known_faces(force_rebuild: bool = False) -> Dict[str, np.ndarray]:
    """
    Loads registered face embeddings from the cache file.
    If the cache is missing or force_rebuild is True, scans databases/, computes
    ArcFace embeddings, averages them per person, and saves the new cache.
    """
    global known_faces

    # Return in-memory embeddings if already loaded and no rebuild requested
    if known_faces and not force_rebuild:
        return known_faces

    # Attempt to load from the pickle file cache
    if os.path.exists(config.CACHE_FILE) and not force_rebuild:
        try:
            with open(config.CACHE_FILE, "rb") as f:
                loaded = pickle.load(f)
            
            # Type and size validation
            if isinstance(loaded, dict) and all(isinstance(k, str) and isinstance(v, np.ndarray) for k, v in loaded.items()):
                with lock:
                    known_faces = loaded
                print(f"[FACE] Successfully loaded {len(known_faces)} registered identities from cache file: {config.CACHE_FILE}")
                return known_faces
            else:
                print("[FACE] Cache file format was invalid. Rebuilding...")
        except Exception as e:
            print(f"[FACE] Failed to read embeddings cache: {e}. Rebuilding...")

    # Rebuild the embeddings cache
    print("[FACE] Building new face embeddings cache. This may take a few minutes...")
    initialize_model()

    new_known_faces: Dict[str, np.ndarray] = {}
    db_path = config.DEFAULT_DB_PATH
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    # Scan directories in database path
    for person_dir_name in os.listdir(db_path):
        person_path = os.path.join(db_path, person_dir_name)
        # Skip files, train/validation structural roots
        if not os.path.isdir(person_path) or person_dir_name in ["train", "validation"]:
            continue

        display_name = person_dir_name.replace("_", " ").title()
        embeddings: List[np.ndarray] = []

        for img_file in os.listdir(person_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            img_path = os.path.join(person_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Run face detection and alignment
                faces = app.get(img)
                if faces:
                    # Choose the largest detected face as the subject
                    largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    embeddings.append(largest_face.embedding)
            except Exception as e:
                print(f"[FACE] Error extracting embedding from {img_path}: {e}")

        if embeddings:
            # Average embeddings per person for robust registration
            avg_embedding = np.mean(embeddings, axis=0)
            new_known_faces[display_name] = avg_embedding
            print(f"[FACE] Registered {display_name}")

    # Write new cache file
    if new_known_faces:
        try:
            with open(config.CACHE_FILE, "wb") as f:
                pickle.dump(new_known_faces, f)
            print(f"[FACE] Saved face embeddings cache to {config.CACHE_FILE} ({len(new_known_faces)} registered)")
        except Exception as e:
            print(f"[FACE] Failed to write embeddings cache file: {e}")

    with lock:
        known_faces = new_known_faces

    return known_faces

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two feature vector embeddings.
    """
    dot_product = np.dot(emb1, emb2)
    norm_a = np.linalg.norm(emb1)
    norm_b = np.linalg.norm(emb2)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def recognize_face(roi: np.ndarray, track_id: int) -> Tuple[str, float]:
    """
    Performs face detection, extraction, and similarity comparison for a given person crop.
    
    Args:
        roi: BGR frame crop of the tracked person (or face candidate).
        track_id: Integer tracking ID to apply recognition cooldown rates.
        
    Returns:
        A tuple of (recognized_name, similarity_score).
    """
    current_time = time.time()

    # 1. Cooldown Rate-Limiting Check
    if track_id in last_attempt_time:
        if current_time - last_attempt_time[track_id] < config.COOLDOWN_TIME:
            if track_id in last_match_result:
                return last_match_result[track_id]
            return "Unknown", 0.0

    last_attempt_time[track_id] = current_time

    # 2. Face Crop Validation
    if roi is None or not isinstance(roi, np.ndarray) or roi.size == 0:
        return "Unknown", 0.0

    h, w, _ = roi.shape
    if h < 20 or w < 20:
        # Ignore crops too small to detect/extract faces reliably
        return "Unknown", 0.0

    initialize_model()
    load_known_faces()

    if not known_faces:
        # No identities registered in the system
        return "Unknown", 0.0

    try:
        # Detect faces within the person ROI
        faces = app.get(roi)
        if not faces:
            # No face found in this crop
            res = ("Unknown", 0.0)
            last_match_result[track_id] = res
            return res

        # Sort to find the largest face box (assumed to be our target person)
        largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        
        # Check face detection quality confidence
        if largest_face.det_score < config.DETECTION_THRESHOLD:
            res = ("Unknown", float(largest_face.det_score))
            last_match_result[track_id] = res
            print(f"[FACE] Unknown Person ({largest_face.det_score:.2f})")
            return res

        current_embedding = largest_face.embedding

        # 3. Match via Cosine Similarity against known faces
        best_name = "Unknown"
        best_sim = 0.0

        for name, known_emb in known_faces.items():
            sim = cosine_similarity(current_embedding, known_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        # Compare against configured match threshold
        if best_sim >= config.FACE_MATCH_THRESHOLD:
            # Split display name to output first name or folder basename as logged
            logged_name = best_name.split()[0]
            print(f"[FACE] Recognized {logged_name} ({best_sim:.2f})")
            res = (best_name, best_sim)
        else:
            print(f"[FACE] Unknown Person ({best_sim:.2f})")
            res = ("Unknown", best_sim)

        last_match_result[track_id] = res
        return res

    except Exception as e:
        print(f"[FACE] Error performing recognition on crop: {e}")
        return "Unknown", 0.0

def try_recognize(track_id: int, roi: np.ndarray, callback: Callable[[int, str, float], None]) -> None:
    """
    Asynchronous wrapper API. Launches recognition in a background daemon thread
    and invokes callback(track_id, name, confidence) upon completion.
    """
    def run_async() -> None:
        name, similarity = recognize_face(roi, track_id)
        if callback:
            try:
                callback(track_id, name, similarity)
            except Exception as e:
                print(f"[ERROR] Callback error in try_recognize: {e}")

    t = threading.Thread(target=run_async, daemon=True)
    t.start()
