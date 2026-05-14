import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

from deepface import DeepFace
print("Testing DeepFace DB rebuild with OpenCV...")
try:
    img_path = r"databases\train\Hargun Hunjan\WhatsApp_Image_2026-04-20_at_12_19_54.jpg"
    dfs = DeepFace.find(img_path=img_path, db_path="databases", model_name="Facenet", distance_metric="cosine", enforce_detection=True, detector_backend='opencv', silent=False)
    print("Success. Found matches:", len(dfs[0]) if dfs else 0)
except Exception as e:
    print(f"Exception: {e}")
