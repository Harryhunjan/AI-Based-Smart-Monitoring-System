# Run the script using: python real_time_object_detection.py

from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import numpy as np
import time
import cv2
import threading
import os
import math
from datetime import datetime
import face_recognizer
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
# tracked_persons: id -> {"name": "Unknown", "prev_name": "Unknown", "entry_log": bool, "last_seen": timestamp, "face_identified": bool, "bbox": (x1, y1, x2, y2), "conf": float}
tracked_persons = {}

# tracked_objects: id -> {"cls_name": name, "linked_person": id, "positions": [(x,y)], "stationary": bool, "lost_alert_logged": bool, "last_seen": timestamp, "bbox": (x1, y1, x2, y2), "conf": float}
tracked_objects = {}

MAX_UI_EVENTS = 5
ui_events = []

def add_ui_event(event_text, is_unknown=False):
    current_time_str = datetime.now().strftime("%H:%M:%S")
    color = (0, 0, 255) if is_unknown else (255, 255, 255)
    ui_events.append({'time': current_time_str, 'text': event_text, 'color': color})
    if len(ui_events) > MAX_UI_EVENTS:
        ui_events.pop(0)

def draw_transparent_rect(img, top_left, bottom_right, color, alpha):
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def recognize_face_task(face_image, track_id):
    global is_recognizing, last_face_match_confidence, last_face_match_name
    try:
        face_recognizer.initialize_model()
        face_recognizer.load_known_faces()
        
        name, similarity = face_recognizer.recognize_face(face_image, track_id)
        
        if name != "Unknown":
            last_face_match_confidence = f"{similarity * 100:.2f}%"
            last_face_match_name = name
            
            if track_id in tracked_persons:
                tracked_persons[track_id]["name"] = name
                tracked_persons[track_id]["face_identified"] = True
        else:
            last_face_match_confidence = f"{similarity * 100:.2f}%"
            last_face_match_name = "Unknown"
    except Exception as e:
        print(f"[ERROR] Error inside recognize_face_task: {e}")
    finally:
        is_recognizing = False

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
        
    # Increase the actual size of the frame
    frame = imutils.resize(frame, width=1000)
    
    frame_count += 1
    if time.time() - start_time >= 1.0:
        current_fps = frame_count / (time.time() - start_time)
        frame_count = 0
        start_time = time.time()
    
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
                # Clip to frame dimensions
                h_frame, w_frame, _ = frame.shape
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w_frame, endX)
                endY = min(h_frame, endY)
                
                # Register new person
                if track_id not in tracked_persons:
                    tracked_persons[track_id] = {
                        "name": "Unknown",
                        "prev_name": "Unknown",
                        "entry_log": False,
                        "last_seen": current_time,
                        "face_identified": False,
                        "bbox": (startX, startY, endX, endY)
                    }
                    add_ui_event(f"ID: {track_id} Unknown", is_unknown=True)
                else:
                    tracked_persons[track_id]["last_seen"] = current_time
                    tracked_persons[track_id]["bbox"] = (startX, startY, endX, endY)
                    
                    current_name = tracked_persons[track_id]["name"]
                    prev_name = tracked_persons[track_id].get("prev_name", "Unknown")
                    if current_name != prev_name and current_name != "Unknown":
                        add_ui_event(f"ID: {track_id} {current_name} (Known)", is_unknown=False)
                        tracked_persons[track_id]["prev_name"] = current_name
                
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
                label = f"ID: {track_id} | {name}"
                color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                # Draw filled rectangle for label
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (startX, max(0, startY - text_h - 10)), (startX + text_w + 10, startY), color, -1)
                cv2.putText(frame, label, (startX + 5, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
        # Phase 2: Process Objects
        for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
            if cls != 0: # Object (Backpack, Laptop, etc.)
                (startX, startY, endX, endY) = box.astype("int")
                # Clip to frame dimensions
                h_frame, w_frame, _ = frame.shape
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w_frame, endX)
                endY = min(h_frame, endY)
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
                if tr_obj["linked_person"] is not None and tr_obj["linked_person"] in tracked_persons:
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

    # Draw Top-Left Stats Panel
    h_frame, w_frame, _ = frame.shape
    panel_x1, panel_y1 = 10, 10
    panel_x2, panel_y2 = 280, 140
    draw_transparent_rect(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), 0.6)

    total_people = len(tracked_persons)
    known_count = sum(1 for p in tracked_persons.values() if p["name"] != "Unknown")
    unknown_count = sum(1 for p in tracked_persons.values() if p["name"] == "Unknown")

    current_date_time = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, current_date_time, (panel_x1 + 10, panel_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"People Tracked: {total_people}", (panel_x1 + 10, panel_y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(frame, f"Known: {known_count} | ", (panel_x1 + 10, panel_y1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    (w, h), _ = cv2.getTextSize(f"Known: {known_count} | ", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(frame, f"Unknown: {unknown_count}", (panel_x1 + 10 + w, panel_y1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (panel_x1 + 10, panel_y1 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Draw Bottom-Left Event Log
    log_x1, log_y1 = 10, h_frame - 160
    log_x2, log_y2 = 320, h_frame - 10
    draw_transparent_rect(frame, (log_x1, log_y1), (log_x2, log_y2), (0, 0, 0), 0.6)

    cv2.putText(frame, "Event Log", (log_x1 + 10, log_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    y_offset = log_y1 + 50
    for event in ui_events:
        cv2.putText(frame, f"{event['time']}  {event['text']}", (log_x1 + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, event['color'], 1)
        y_offset += 20

    # Show Frame
    cv2.imshow("Smart Monitoring - Live Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()