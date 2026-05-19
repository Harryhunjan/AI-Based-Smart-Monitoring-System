from deepface import DeepFace
import os
import shutil

print("[INFO] Building Face Recognition database cache...")

db_path = "databases"

# First, clean up any old/corrupted pkl files
for file in os.listdir(db_path):
    if file.endswith(".pkl"):
        try:
            os.remove(os.path.join(db_path, file))
            print(f"[INFO] Removed old cache file: {file}")
        except Exception as e:
            pass

print("[INFO] Starting DeepFace represent to build new cache. This may take a few minutes depending on the number of images...")

# Running a dummy find to force DeepFace to generate the embeddings pkl cache.
# We use a dummy image (just taking one from the database)
dummy_img = None
for root, dirs, files in os.walk(db_path):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            dummy_img = os.path.join(root, f)
            break
    if dummy_img:
        break

if dummy_img:
    print(f"[INFO] Using {dummy_img} as dummy to trigger cache build...")
    try:
        DeepFace.find(img_path=dummy_img, db_path=db_path, model_name="Facenet", distance_metric="cosine", enforce_detection=False, detector_backend='opencv')
        print("[SUCCESS] Database cache built successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to build cache: {e}")
else:
    print("[ERROR] No images found in databases directory to build cache.")
