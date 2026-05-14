import pickle
import os
from collections import Counter

pkl_path = r"databases\ds_model_facenet_detector_opencv_aligned_normalization_base_expand_0.pkl"
try:
    with open(pkl_path, "rb") as f:
        representations = pickle.load(f)
    
    names = []
    for item in representations:
        path = item["identity"]
        parent_dir = os.path.basename(os.path.dirname(path))
        if parent_dir not in ["train", "validation", "databases"]:
            names.append(parent_dir.replace("_", " "))
        else:
            names.append(os.path.splitext(os.path.basename(path))[0].replace("_", " "))
            
    counts = Counter(names)
    print("Total valid face representations extracted:", len(representations))
    print("Representations per person in database:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
except Exception as e:
    print(f"Error: {repr(e)}")
