# Run the script using: python real_time_object_detection.py

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import cv2
import threading
import os
import math
from deepface import DeepFace
from ultralytics import YOLO
from logger import EventLogger

# Initialize logger
logger = EventLogger()

# Initialize YOLO model
print("[INFO] loading YOLO model...")
model = YOLO("yolov8n.pt")

# DeepFace globals
is_recognizing = False
last_face_match_confidence = "N/A"
last_face_match_name = "None"

# Tracking dictionaries
# tracked_persons: id -> {"name": "Unknown", "entry_log": bool, "last_seen": timestamp, "face_identified": bool, "bbox": (x1, y1, x2, y2), "conf": float}
tracked_persons = {}

# tracked_objects: id -> {"cls_name": name, "linked_person": id, "positions": [(x,y)], "stationary": bool, "lost_alert_logged": bool, "last_seen": timestamp, "bbox": (x1, y1, x2, y2), "conf": float}
tracked_objects = {}

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def recognize_face_task(face_image, track_id):
    global is_recognizing, last_face_match_confidence, last_face_match_name
    try:
        temp_img_path = f"temp_roi_{track_id}.jpg"
        cv2.imwrite(temp_img_path, face_image)
        # We pass the person ROI. DeepFace will use its detector (default is opencv) to find the face within.
        dfs = DeepFace.find(img_path=temp_img_path, db_path="face_database", enforce_detection=False, detector_backend='opencv', silent=True)
        if len(dfs) > 0 and not dfs[0].empty:
            matched_path = dfs[0].iloc[0]['identity']
            recognized_name = os.path.splitext(os.path.basename(matched_path))[0].replace("_", " ")
            
            # Extract distance/confidence metric from DeepFace if available
            distance_metric = dfs[0].iloc[0].get('VGG-Face_cosine', dfs[0].iloc[0].get('distance', 0.0))
            last_face_match_confidence = f"{max(0, 100 - (distance_metric * 100)):.2f}%"
            last_face_match_name = recognized_name

            if track_id in tracked_persons:
                tracked_persons[track_id]["name"] = recognized_name
                tracked_persons[track_id]["face_identified"] = True
    except Exception as e:
        pass
    finally:
        is_recognizing = False
        try:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
        except:
            pass

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

fps = FPS().start()

STATIONARY_THRESH_PX = 20
STATIONARY_FRAMES_REQUIRED = 30
DISAPPEAR_TIMEOUT = 2.0  # seconds until exit

# COCO Classes of interest: 0: person, 24: backpack, 26: handbag, 28: suitcase, 39: bottle, 63: laptop, 67: cell phone
TARGET_CLASSES = [0, 24, 26, 28, 39, 63, 67]

# Variables to calculate rolling FPS
frame_count = 0
start_time = time.time()
current_fps = 0

while True:
    frame = vs.read()
    if frame is None:
        continue
    
    # Optional: Resize for speed if necessary, but yolov8 auto-scales
    # frame = imutils.resize(frame, width=800)
    
    # Run YOLO tracking inference
    # persist=True handles the ID tracking internally via bot-sort/bytetrack
    results = model.track(frame, persist=True, classes=TARGET_CLASSES, verbose=False)
    
    current_time = time.time()
    current_person_ids = []
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        clss = results[0].boxes.cls.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        # Phase 1: Process Persons
        for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
            if cls == 0:  # Person
                current_person_ids.append(track_id)
                (startX, startY, endX, endY) = box.astype("int")
                
                # Register new person
                if track_id not in tracked_persons:
                    tracked_persons[track_id] = {
                        "name": "Unknown",
                        "entry_log": False,
                        "last_seen": current_time,
                        "face_identified": False,
                        "bbox": (startX, startY, endX, endY)
                    }
                else:
                    tracked_persons[track_id]["last_seen"] = current_time
                    tracked_persons[track_id]["bbox"] = (startX, startY, endX, endY)
                
                # Log Entry
                if not tracked_persons[track_id]["entry_log"]:
                    logger.log_event("ENTRY", person_name=tracked_persons[track_id]["name"])
                    tracked_persons[track_id]["entry_log"] = True
                
                # Try to identify face
                if not tracked_persons[track_id]["face_identified"] and not is_recognizing:
                    person_roi = frame[startY:endY, startX:endX]
                    if person_roi.shape[0] > 20 and person_roi.shape[1] > 20:
                        is_recognizing = True
                        t = threading.Thread(target=recognize_face_task, args=(person_roi.copy(), track_id))
                        t.daemon = True
                        t.start()
                        
                # Draw Box
                name = tracked_persons[track_id]["name"]
                label = f"Person ID:{track_id} - {name}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        # Phase 2: Process Objects
        for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
            if cls != 0: # Object (Backpack, Laptop, etc.)
                (startX, startY, endX, endY) = box.astype("int")
                cx = int((startX + endX) / 2)
                cy = int((startY + endY) / 2)
                cls_name = model.names[cls]
                
                if track_id not in tracked_objects:
                    tracked_objects[track_id] = {
                        "cls_name": cls_name,
                        "linked_person": None,
                        "positions": [],
                        "stationary": False,
                        "lost_alert_logged": False,
                        "last_seen": current_time,
                        "bbox": (startX, startY, endX, endY)
                    }
                else:
                    tracked_objects[track_id]["last_seen"] = current_time
                    tracked_objects[track_id]["bbox"] = (startX, startY, endX, endY)
                
                tr_obj = tracked_objects[track_id]
                tr_obj["positions"].append((cx, cy))
                if len(tr_obj["positions"]) > STATIONARY_FRAMES_REQUIRED:
                    tr_obj["positions"].pop(0)
                    
                # Check Stationarity
                if len(tr_obj["positions"]) == STATIONARY_FRAMES_REQUIRED:
                    max_dist = 0
                    for p in tr_obj["positions"]:
                        d = get_distance(p, (cx, cy))
                        if d > max_dist: max_dist = d
                    
                    if max_dist < STATIONARY_THRESH_PX:
                        tr_obj["stationary"] = True
                    else:
                        tr_obj["stationary"] = False
                        tr_obj["lost_alert_logged"] = False
                
                # Linking object to person
                if not tr_obj["stationary"]:
                    closest_person = None
                    min_pd = float('inf')
                    for pid in current_person_ids:
                        px1, py1, px2, py2 = tracked_persons[pid]["bbox"]
                        pcx = int((px1 + px2) / 2)
                        pcy = int((py1 + py2) / 2)
                        pd = get_distance((cx, cy), (pcx, pcy))
                        if pd < min_pd:
                            min_pd = pd
                            closest_person = pid
                    
                    # If person is close, link them
                    if closest_person is not None:
                        px1, py1, px2, py2 = tracked_persons[closest_person]["bbox"]
                        if min_pd < max((endX-startX), (px2-px1)) * 1.5:
                            tr_obj["linked_person"] = closest_person
                        
                # Lost Object Logic
                color = (255, 0, 0) # Default Blue for objects
                if tr_obj["linked_person"] is not None:
                    p_name = tracked_persons[tr_obj["linked_person"]].get("name", "Unknown")
                    status_text = f"{cls_name.title()} (Linked to: {p_name})"
                else:
                    status_text = f"ID:{track_id} {cls_name.title()}"
                    
                # If object is stationary and its linked person is far or missing
                if tr_obj["stationary"] and tr_obj["linked_person"] is not None:
                    pid = tr_obj["linked_person"]
                    person_is_far = True
                    
                    if pid in tracked_persons and (current_time - tracked_persons[pid]["last_seen"]) < DISAPPEAR_TIMEOUT:
                        px1, py1, px2, py2 = tracked_persons[pid]["bbox"]
                        pcx = int((px1 + px2) / 2)
                        pcy = int((py1 + py2) / 2)
                        # We consider the person far if the distance exceeds a certain threshold
                        if get_distance((pcx, pcy), (cx, cy)) < (px2 - px1) + 200:
                            person_is_far = False
                            
                    if person_is_far:
                        color = (0, 0, 255) # Red for lost
                        p_name = "Unknown"
                        if pid in tracked_persons:
                            p_name = tracked_persons[pid]["name"]
                            
                        status_text = f"! LOST {cls_name} (Owner: {p_name}) !"
                        if not tr_obj["lost_alert_logged"]:
                            logger.log_event("ITEM_LEFT_BEHIND", person_name=p_name, item_class=cls_name)
                            tr_obj["lost_alert_logged"] = True
                            
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, status_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Phase 3: Cleanup and Exit Detection
    for pid in list(tracked_persons.keys()):
        if current_time - tracked_persons[pid]["last_seen"] > DISAPPEAR_TIMEOUT:
            name = tracked_persons[pid]["name"]
            logger.log_event("EXIT", person_name=name)
            del tracked_persons[pid]
            
    for oid in list(tracked_objects.keys()):
        if current_time - tracked_objects[oid]["last_seen"] > DISAPPEAR_TIMEOUT * 2: # Keep objects a bit longer
            del tracked_objects[oid]

    # Show Frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()