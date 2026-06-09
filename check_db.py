import pickle
import os
import numpy as np
import config

pkl_path = config.CACHE_FILE

print(f"[INFO] Checking face recognition database cache file: {pkl_path}")

if not os.path.exists(pkl_path):
    print("[ERROR] Cache file does not exist. Run 'python build_cache.py' to generate it first.")
else:
    try:
        with open(pkl_path, "rb") as f:
            known_faces = pickle.load(f)
            
        print(f"\nTotal registered face identities: {len(known_faces)}")
        print("Registered names and embedding vectors in database:")
        for name, emb in sorted(known_faces.items()):
            norm_val = np.linalg.norm(emb)
            print(f"  * {name:<25} | Embedding shape: {emb.shape} | Vector norm: {norm_val:.4f}")
            
    except Exception as e:
        print(f"[ERROR] Failed to read database cache: {e}")
