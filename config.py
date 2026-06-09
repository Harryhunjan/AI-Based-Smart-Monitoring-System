import os

# Face Recognition Configuration Settings
FACE_MATCH_THRESHOLD = 0.55     # Cosine similarity threshold (higher is more strict/less false positives)
DEFAULT_DB_PATH = "databases"   # Directory where registered face folders reside
MODEL_NAME = "buffalo_l"        # InsightFace model package (buffalo_l includes detection & ArcFace)
DETECTION_THRESHOLD = 0.5      # Detection confidence score threshold for validating crops
COOLDOWN_TIME = 2.0            # Time in seconds before re-attempting recognition on a track_id
CACHE_FILE = os.path.join(DEFAULT_DB_PATH, "known_faces_cache.pkl")  # Path to pickle file cache
