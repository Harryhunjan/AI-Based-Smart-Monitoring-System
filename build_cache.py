import os
import face_recognizer

print("[INFO] Rebuilding Face Recognition database cache for InsightFace (ArcFace)...")

# Force rebuild of InsightFace database embeddings cache
try:
    known = face_recognizer.load_known_faces(force_rebuild=True)
    print(f"[SUCCESS] Database cache rebuilt successfully! Registered {len(known)} identities.")
    for name in known:
        print(f"  - {name}")
except Exception as e:
    print(f"[ERROR] Failed to rebuild face recognition cache: {e}")
