import os
import sys
import time
import queue
import math
import shutil
import threading
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import cv2
import numpy as np
import imutils
from PIL import Image, ImageTk
import customtkinter as ctk

# Matplotlib styling for embedding in Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns

# Config path settings
DEFAULT_DB_PATH = "databases"
DEFAULT_LOG_FILE = "logs/monitoring_events.csv"

# Global references for AI models (loaded in background thread)
yolo_model = None
face_recognizer = None
EventLogger = None
logger_instance = None

# Set theme and look
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

CLASS_MAPPING = {
    "Person": 0,
    "Backpack": 24,
    "Handbag": 26,
    "Suitcase": 28,
    "Bottle": 39,
    "Laptop": 63,
    "Cell Phone": 67
}

class ModelLoaderThread(threading.Thread):
    """Background thread to load large ML packages and models without freezing the GUI"""
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.daemon = True

    def run(self):
        global yolo_model, EventLogger, logger_instance
        try:
            print("[INFO] Background thread: importing ultralytics...")
            from ultralytics import YOLO
            from logger import EventLogger as el_import
            
            EventLogger = el_import
            logger_instance = EventLogger()
            
            print("[INFO] Background thread: loading YOLOv8 model...")
            yolo_model = YOLO("yolov8n.pt")
            
            print("[INFO] Background thread: AI models loaded successfully!")
            self.callback(True, "System Ready")
        except Exception as e:
            print(f"[ERROR] Background thread: model loading failed: {e}")
            self.callback(False, f"Load Error: {str(e)}")

class FaceRecognizer(threading.Thread):
    """Single, persistent background thread that executes InsightFace (ArcFace) matches sequentially"""
    def __init__(self, task_queue, result_queue):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.daemon = True
        self.running = True

    def run(self):
        print("[INFO] FaceRecognizer thread started. Importing face_recognizer...")
        try:
            import face_recognizer
            face_recognizer.initialize_model()
            face_recognizer.load_known_faces()
            print("[INFO] FaceRecognizer: InsightFace models loaded successfully and ready.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize FaceRecognizer inside recognizer thread: {e}")
            return
            
        while self.running:
            try:
                # Retrieve face matching task
                task = self.task_queue.get(timeout=1.0)
                face_image, track_id, dist_threshold = task
                
                try:
                    import face_recognizer
                    # Call InsightFace recognition
                    name, similarity = face_recognizer.recognize_face(face_image, track_id)
                    
                    if name != "Unknown":
                        # Create a dummy matched path structure to preserve GUI's expectations
                        folder_name = name.lower().replace(" ", "_")
                        matched_path = os.path.join(DEFAULT_DB_PATH, folder_name, "dummy.jpg")
                        distance_metric = 1.0 - similarity
                        print(f"[DEBUG] FaceRecognizer: Track {track_id} matched face: {matched_path} (dist: {distance_metric:.4f})")
                        self.result_queue.put((track_id, matched_path, distance_metric))
                    else:
                        distance_metric = 1.0 - similarity
                        print(f"[DEBUG] FaceRecognizer: Track {track_id} has no match in DB. (best sim: {similarity:.4f})")
                        self.result_queue.put((track_id, None, distance_metric))
                except Exception as e:
                    print(f"[ERROR] FaceRecognizer thread task error: {e}")
                    self.result_queue.put((track_id, None, 1.0))
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                continue

class VideoWorker(threading.Thread):
    """Background thread that captures camera frames, runs YOLO + InsightFace, and enqueues processed frames"""
    def __init__(self, frame_queue, log_queue, face_task_queue, face_result_queue, status_callback):
        super().__init__()
        self.frame_queue = frame_queue
        self.log_queue = log_queue
        self.face_task_queue = face_task_queue
        self.face_result_queue = face_result_queue
        self.status_callback = status_callback
        self.running = False
        self.daemon = True
        
        # Configuration parameters (can be adjusted on-the-fly)
        self.conf_threshold = 0.25
        self.dist_threshold = 0.45
        self.target_classes = [0, 24, 26, 28, 39, 63, 67]
        
        self.active_face_tasks = set()
        self.tracked_persons = {}
        self.tracked_objects = {}
        
        # UI stats variables
        self.current_fps = 0.0

        # Load Haar Cascade for face extraction from person bounding boxes
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            print(f"[ERROR] Failed to load Haar Cascade: {e}")
            self.face_cascade = None

    def calculate_iooa(self, box_obj, box_person):
        ox1, oy1, ox2, oy2 = box_obj
        px1, py1, px2, py2 = box_person
        
        # Calculate intersection coordinates
        ix1 = max(ox1, px1)
        iy1 = max(oy1, py1)
        ix2 = min(ox2, px2)
        iy2 = min(oy2, py2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
            
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        obj_area = (ox2 - ox1) * (oy2 - oy1)
        
        return inter_area / float(obj_area)

    def get_photo_count(self, name):
        if not name or name == "Unknown":
            return 0
        try:
            if os.path.exists(DEFAULT_DB_PATH):
                for item in os.listdir(DEFAULT_DB_PATH):
                    full_path = os.path.join(DEFAULT_DB_PATH, item)
                    if os.path.isdir(full_path) and item.lower().replace("_", " ") == name.lower():
                        return len([f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        except Exception as e:
            print(f"[ERROR] Failed to get photo count: {e}")
        return 0

    def start_stream(self):
        self.running = True
        self.active_face_tasks.clear()
        self.tracked_persons.clear()
        self.tracked_objects.clear()

    def stop_stream(self):
        self.running = False

    def log_event(self, event_type, person_name="Unknown", item_class="None", is_unknown=False):
        # Log to CSV
        if logger_instance:
            logger_instance.log_event(event_type, person_name=person_name, item_class=item_class)
        # Log to GUI
        timestamp = datetime.now().strftime("%H:%M:%S")
        if event_type == "IDENTIFIED":
            log_text = f"[{timestamp}] {event_type} | Person: {person_name}"
            if item_class != "None":
                log_text += f" | Carrying: {item_class}"
        else:
            log_text = f"[{timestamp}] {event_type} | Person: {person_name}"
            if item_class != "None":
                log_text += f" | Item: {item_class}"
        self.log_queue.put((log_text, is_unknown))

    def run(self):
        from imutils.video import VideoStream
        
        STATIONARY_THRESH_PX = 20
        STATIONARY_FRAMES_REQUIRED = 30
        DISAPPEAR_TIMEOUT = 2.0
        
        while True:
            if not self.running:
                time.sleep(0.1)
                continue
                
            print("[INFO] Starting VideoStream capture...")
            self.status_callback("Connecting Camera...")
            vs = VideoStream(src=0).start()
            time.sleep(1.5)
            self.status_callback("Monitoring Active")
            
            frame_count = 0
            start_time = time.time()
            
            while self.running:
                # 0. Check for completed face recognition tasks
                try:
                    while True:
                        res_track_id, matched_path, distance_metric = self.face_result_queue.get_nowait()
                        if res_track_id in self.active_face_tasks:
                            self.active_face_tasks.remove(res_track_id)
                            
                        if matched_path is not None and distance_metric <= self.dist_threshold:
                            parent_dir = os.path.basename(os.path.dirname(matched_path))
                            if parent_dir and parent_dir not in ["train", "validation", "databases"]:
                                recognized_name = parent_dir.replace("_", " ")
                            else:
                                recognized_name = os.path.splitext(os.path.basename(matched_path))[0].replace("_", " ")
                                
                            if res_track_id in self.tracked_persons:
                                self.tracked_persons[res_track_id]["name"] = recognized_name
                                self.tracked_persons[res_track_id]["photo_count"] = self.get_photo_count(recognized_name)
                                self.tracked_persons[res_track_id]["face_identified"] = True
                                # Find linked objects for this person to include in identification log
                                temp_linked_objs = []
                                for oid, obj in self.tracked_objects.items():
                                    if obj["linked_person"] == res_track_id:
                                        temp_linked_objs.append(obj["cls_name"].title())
                                carrying_str = ", ".join(sorted(list(set(temp_linked_objs)))) if temp_linked_objs else "None"
                                self.log_event("IDENTIFIED", person_name=recognized_name, item_class=carrying_str, is_unknown=False)
                except queue.Empty:
                    pass

                frame = vs.read()
                if frame is None:
                    continue
                
                h_frame, w_frame, _ = frame.shape
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    self.current_fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()
                
                # YOLO Tracking inference
                results = yolo_model.track(
                    frame, 
                    persist=True, 
                    classes=self.target_classes, 
                    conf=self.conf_threshold,
                    imgsz=320,
                    verbose=False
                )
                
                current_time = time.time()
                current_person_ids = []
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    clss = results[0].boxes.cls.int().cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    # 1. Process Persons
                    for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
                        if cls == 0:  # Person
                            current_person_ids.append(track_id)
                            startX, startY, endX, endY = box.astype("int")
                            startX, startY = max(0, startX), max(0, startY)
                            endX, endY = min(w_frame, endX), min(h_frame, endY)
                            
                            # Register track
                            if track_id not in self.tracked_persons:
                                self.tracked_persons[track_id] = {
                                    "name": "Unknown",
                                    "prev_name": "Unknown",
                                    "entry_log": False,
                                    "last_seen": current_time,
                                    "face_identified": False,
                                    "bbox": (startX, startY, endX, endY)
                                }
                                self.log_event("ENTRY", person_name="Unknown", is_unknown=True)
                            else:
                                self.tracked_persons[track_id]["last_seen"] = current_time
                                self.tracked_persons[track_id]["bbox"] = (startX, startY, endX, endY)
                            
                            # Logging Entry Event
                            if not self.tracked_persons[track_id]["entry_log"]:
                                self.tracked_persons[track_id]["entry_log"] = True
                                
                            # Face Identification task submission - extract face ROI using Haar Cascade or Fallback
                            if not self.tracked_persons[track_id]["face_identified"] and track_id not in self.active_face_tasks:
                                person_roi = frame[startY:endY, startX:endX]
                                if person_roi.shape[0] > 20 and person_roi.shape[1] > 20:
                                    face_roi = None
                                    if self.face_cascade is not None:
                                        gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                                        faces = self.face_cascade.detectMultiScale(gray_roi, 1.1, 3, minSize=(30, 30))
                                        if len(faces) > 0:
                                            fx, fy, fw, fh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                                            face_roi = person_roi[fy:fy+fh, fx:fx+fw]
                                            
                                    # Fallback: crop the upper 35% of the person's box
                                    if face_roi is None:
                                        h_person = endY - startY
                                        w_person = endX - startX
                                        face_h = int(h_person * 0.35)
                                        face_w = int(w_person * 0.8)
                                        start_fx = max(0, int((w_person - face_w) / 2))
                                        face_roi = person_roi[0:face_h, start_fx:start_fx+face_w]
                                        
                                    if face_roi.shape[0] > 10 and face_roi.shape[1] > 10:
                                        face_roi_resized = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_CUBIC)
                                        self.active_face_tasks.add(track_id)
                                        self.face_task_queue.put((face_roi_resized, track_id, self.dist_threshold))
                            
                            # Find linked objects for this person (from self.tracked_objects)
                            linked_objs = []
                            for oid, obj in self.tracked_objects.items():
                                if obj["linked_person"] == track_id:
                                    linked_objs.append(obj["cls_name"].title())
                            
                            # Render Frame Box with Photo Count and Carrying list
                            name = self.tracked_persons[track_id]["name"]
                            photo_count = self.tracked_persons[track_id].get("photo_count", 0)
                            if name != "Unknown":
                                name_display = f"{name} ({photo_count} photos)"
                            else:
                                name_display = name
                                
                            if linked_objs:
                                objs_str = ", ".join(sorted(list(set(linked_objs))))
                                label = f"ID: {track_id} | {name_display} | Carrying: {objs_str}"
                            else:
                                label = f"ID: {track_id} | {name_display}"
                                
                            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                            
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(frame, (startX, max(0, startY - text_h - 8)), (startX + text_w + 10, startY), color, -1)
                            cv2.putText(frame, label, (startX + 5, startY - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                    # 2. Process Objects
                    for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
                        if cls != 0:  # Not Person
                            startX, startY, endX, endY = box.astype("int")
                            startX, startY = max(0, startX), max(0, startY)
                            endX, endY = min(w_frame, endX), min(h_frame, endY)
                            cx, cy = int((startX + endX) / 2), int((startY + endY) / 2)
                            cls_name = yolo_model.names[cls]
                            
                            if track_id not in self.tracked_objects:
                                self.tracked_objects[track_id] = {
                                    "cls_name": cls_name,
                                    "linked_person": None,
                                    "positions": [],
                                    "stationary": False,
                                    "lost_alert_logged": False,
                                    "last_seen": current_time,
                                    "bbox": (startX, startY, endX, endY)
                                }
                            else:
                                self.tracked_objects[track_id]["last_seen"] = current_time
                                self.tracked_objects[track_id]["bbox"] = (startX, startY, endX, endY)
                                
                            tr_obj = self.tracked_objects[track_id]
                            tr_obj["positions"].append((cx, cy))
                            if len(tr_obj["positions"]) > STATIONARY_FRAMES_REQUIRED:
                                tr_obj["positions"].pop(0)
                                
                            # Evaluate stationarity
                            if len(tr_obj["positions"]) == STATIONARY_FRAMES_REQUIRED:
                                max_dist = 0
                                for p in tr_obj["positions"]:
                                    dist = math.sqrt((p[0] - cx)**2 + (p[1] - cy)**2)
                                    if dist > max_dist:
                                        max_dist = dist
                                if max_dist < STATIONARY_THRESH_PX:
                                    tr_obj["stationary"] = True
                                else:
                                    tr_obj["stationary"] = False
                                    tr_obj["lost_alert_logged"] = False
                                    
                            # Link item to owner (using overlap IoOA with proximity fallback)
                            # Only update linkage if the object is NOT currently flagged as stationary-and-abandoned
                            is_abandoned = False
                            if tr_obj["stationary"] and tr_obj["linked_person"] is not None:
                                pid = tr_obj["linked_person"]
                                if pid not in self.tracked_persons or (current_time - self.tracked_persons[pid]["last_seen"]) >= DISAPPEAR_TIMEOUT:
                                    is_abandoned = True
                                else:
                                    px1, py1, px2, py2 = self.tracked_persons[pid]["bbox"]
                                    pcx, pcy = int((px1 + px2) / 2), int((py1 + py2) / 2)
                                    if math.sqrt((pcx - cx)**2 + (pcy - cy)**2) >= (px2 - px1) + 200:
                                        is_abandoned = True
                                        
                            if not is_abandoned:
                                best_pid = None
                                max_iooa = 0.0
                                for pid in current_person_ids:
                                    p_box = self.tracked_persons[pid]["bbox"]
                                    iooa = self.calculate_iooa((startX, startY, endX, endY), p_box)
                                    if iooa > max_iooa:
                                        max_iooa = iooa
                                        best_pid = pid
                                        
                                if max_iooa > 0.3:
                                    tr_obj["linked_person"] = best_pid
                                else:
                                    closest_person = None
                                    min_pd = float('inf')
                                    for pid in current_person_ids:
                                        px1, py1, px2, py2 = self.tracked_persons[pid]["bbox"]
                                        pcx, pcy = int((px1 + px2) / 2), int((py1 + py2) / 2)
                                        pd = math.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                                        if pd < min_pd:
                                            min_pd = pd
                                            closest_person = pid
                                            
                                    if closest_person is not None:
                                        px1, py1, px2, py2 = self.tracked_persons[closest_person]["bbox"]
                                        max_dim = max((endX - startX), (px2 - px1))
                                        if min_pd < max_dim * 1.2:
                                            tr_obj["linked_person"] = closest_person
                                        else:
                                            if not tr_obj["stationary"]:
                                                tr_obj["linked_person"] = None
                                    else:
                                        if not tr_obj["stationary"]:
                                            tr_obj["linked_person"] = None
                                        
                            color = (255, 100, 0)
                            p_name = "None"
                            if tr_obj["linked_person"] is not None and tr_obj["linked_person"] in self.tracked_persons:
                                p_name = self.tracked_persons[tr_obj["linked_person"]].get("name", "Unknown")
                                status_text = f"{cls_name.title()} (Owner: {p_name})"
                            else:
                                status_text = f"ID: {track_id} {cls_name.title()}"
                                
                            # Lost Object Trigger
                            if tr_obj["stationary"] and tr_obj["linked_person"] is not None:
                                pid = tr_obj["linked_person"]
                                person_is_far = True
                                
                                if pid in self.tracked_persons and (current_time - self.tracked_persons[pid]["last_seen"]) < DISAPPEAR_TIMEOUT:
                                    px1, py1, px2, py2 = self.tracked_persons[pid]["bbox"]
                                    pcx, pcy = int((px1 + px2) / 2), int((py1 + py2) / 2)
                                    if math.sqrt((pcx - cx)**2 + (pcy - cy)**2) < (px2 - px1) + 200:
                                        person_is_far = False
                                        
                                if person_is_far:
                                    color = (0, 0, 255)
                                    status_text = f"! LOST {cls_name.upper()} (Owner: {p_name}) !"
                                    if not tr_obj["lost_alert_logged"]:
                                        self.log_event("ITEM_LEFT_BEHIND", person_name=p_name, item_class=cls_name, is_unknown=True)
                                        tr_obj["lost_alert_logged"] = True
                                        
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                            cv2.putText(frame, status_text, (startX, startY - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # 3. Clean up tracks
                for pid in list(self.tracked_persons.keys()):
                    if current_time - self.tracked_persons[pid]["last_seen"] > DISAPPEAR_TIMEOUT:
                        name = self.tracked_persons[pid]["name"]
                        self.log_event("EXIT", person_name=name, is_unknown=False)
                        del self.tracked_persons[pid]
                        
                for oid in list(self.tracked_objects.keys()):
                    if current_time - self.tracked_objects[oid]["last_seen"] > DISAPPEAR_TIMEOUT * 2:
                        del self.tracked_objects[oid]
                
                # Package frame and stats to pass to GUI
                total_people = len(self.tracked_persons)
                known_count = sum(1 for p in self.tracked_persons.values() if p["name"] != "Unknown")
                unknown_count = total_people - known_count
                
                stats = {
                    "fps": self.current_fps,
                    "total_people": total_people,
                    "known": known_count,
                    "unknown": unknown_count,
                    "raw_frame": frame.copy() # reference for screen captures
                }
                
                # Convert frame back to RGB PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Put in queue (keeping max size of 1 to prevent GUI lag)
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put((pil_img, stats))
                
            print("[INFO] VideoStream stopping...")
            vs.stop()
            self.status_callback("System Idle")

class SmartMonitoringApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("AI-Powered Smart Monitoring Dashboard")
        self.geometry("1300x850")
        self.minsize(1100, 750)
        
        # Internal state
        self.frame_queue = queue.Queue(maxsize=2)
        self.log_queue = queue.Queue()
        self.latest_raw_frame = None
        
        # Create Layout Grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Sidebar Frame (Left)
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(10, weight=1)
        
        # Sidebar title
        self.title_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="SMART MONITOR", 
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#3498db"
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.subtitle_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="AI Security Guard System", 
            font=ctk.CTkFont(family="Segoe UI", size=12, slant="italic"),
            text_color="#95a5a6"
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # Status Label
        self.status_header = ctk.CTkLabel(self.sidebar_frame, text="System Status:", font=ctk.CTkFont(size=12, weight="bold"))
        self.status_header.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.status_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Loading System...", 
            text_color="#f1c40f",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.status_label.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="w")
        
        # Stream buttons
        self.btn_start = ctk.CTkButton(
            self.sidebar_frame, 
            text="Start Camera Feed", 
            fg_color="#2ecc71", 
            hover_color="#27ae60",
            font=ctk.CTkFont(weight="bold"),
            command=self.start_camera_stream
        )
        self.btn_start.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        self.btn_start.configure(state="disabled") # Disabled until models load
        
        self.btn_stop = ctk.CTkButton(
            self.sidebar_frame, 
            text="Stop Camera Feed", 
            fg_color="#e74c3c", 
            hover_color="#c0392b",
            font=ctk.CTkFont(weight="bold"),
            command=self.stop_camera_stream
        )
        self.btn_stop.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        self.btn_stop.configure(state="disabled")
        
        # YOLO Sliders Label
        self.sliders_label = ctk.CTkLabel(self.sidebar_frame, text="AI Sensitivity Settings", font=ctk.CTkFont(size=14, weight="bold"))
        self.sliders_label.grid(row=6, column=0, padx=20, pady=(20, 5), sticky="w")
        
        # YOLO Conf Slider
        self.conf_lbl = ctk.CTkLabel(self.sidebar_frame, text="YOLO BBox Confidence: 0.25", font=ctk.CTkFont(size=11))
        self.conf_lbl.grid(row=7, column=0, padx=20, pady=(5, 0), sticky="w")
        self.conf_slider = ctk.CTkSlider(self.sidebar_frame, from_=0.05, to=0.95, number_of_steps=18, command=self.on_conf_change)
        self.conf_slider.grid(row=8, column=0, padx=20, pady=(0, 15), sticky="ew")
        self.conf_slider.set(0.25)
        
        # Face Similarity threshold slider
        self.dist_lbl = ctk.CTkLabel(self.sidebar_frame, text="Face Similarity Thresh: 0.45", font=ctk.CTkFont(size=11))
        self.dist_lbl.grid(row=9, column=0, padx=20, pady=(5, 0), sticky="w")
        self.dist_slider = ctk.CTkSlider(self.sidebar_frame, from_=0.1, to=0.8, number_of_steps=14, command=self.on_dist_change)
        self.dist_slider.grid(row=10, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.dist_slider.set(0.45)
        
        # Detection Target checkboxes
        self.targets_lbl = ctk.CTkLabel(self.sidebar_frame, text="Classes to Track", font=ctk.CTkFont(size=13, weight="bold"))
        self.targets_lbl.grid(row=11, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.checkbox_vars = {}
        row_idx = 12
        for cls_name in CLASS_MAPPING.keys():
            var = tk.BooleanVar(value=True)
            chk = ctk.CTkCheckBox(
                self.sidebar_frame, 
                text=cls_name, 
                variable=var, 
                font=ctk.CTkFont(size=11),
                command=self.update_target_classes
            )
            chk.grid(row=row_idx, column=0, padx=30, pady=2, sticky="w")
            self.checkbox_vars[cls_name] = var
            row_idx += 1
            
        # Branding at bottom of sidebar
        self.brand_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Powered by YOLOv8 & InsightFace (ArcFace)", 
            font=ctk.CTkFont(size=10), 
            text_color="#7f8c8d"
        )
        self.brand_label.grid(row=20, column=0, padx=20, pady=20)
        
        # Main Display Frame (Right Side)
        self.main_container = ctk.CTkFrame(self, corner_radius=10)
        self.main_container.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Tabs system
        self.tabview = ctk.CTkTabview(self.main_container)
        self.tabview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create tabs
        self.tab_feed = self.tabview.add("Live Feed")
        self.tab_register = self.tabview.add("Register Face")
        self.tab_analytics = self.tabview.add("Analytics")
        
        self.setup_feed_tab()
        self.setup_register_tab()
        self.setup_analytics_tab()

        # Key bindings for Fullscreen presentation (F11)
        self.is_fullscreen = False
        self.bind("<F11>", self.toggle_fullscreen)
        self.bind("<Escape>", self.exit_fullscreen)
        
        # Start model loading in background
        ModelLoaderThread(self.on_models_loaded).start()
        
        # Start polling loops
        self.poll_camera_queue()
        self.poll_log_queue()

    def toggle_fullscreen(self, event=None):
        """Toggles fullscreen state of the window"""
        self.is_fullscreen = not self.is_fullscreen
        self.attributes("-fullscreen", self.is_fullscreen)
        return "break"

    def exit_fullscreen(self, event=None):
        """Exits fullscreen state of the window"""
        self.is_fullscreen = False
        self.attributes("-fullscreen", False)
        return "break"

    # --- TABS SETUP ---
    def setup_feed_tab(self):
        self.tab_feed.grid_rowconfigure(0, weight=1)
        self.tab_feed.grid_rowconfigure(1, weight=0)
        self.tab_feed.grid_columnconfigure(0, weight=1)
        
        # Video Display Panel
        self.video_frame = ctk.CTkFrame(self.tab_feed, fg_color="#111116", corner_radius=8)
        self.video_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)
        
        self.video_label = ctk.CTkLabel(
            self.video_frame, 
            text="Camera Stream Off\nClick 'Start Camera Feed' to begin.",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#7f8c8d"
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")
        
        # Horizontal Metrics Bar
        self.metrics_bar = ctk.CTkFrame(self.tab_feed, height=60, corner_radius=8)
        self.metrics_bar.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.metrics_bar.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        self.metric_fps = ctk.CTkLabel(self.metrics_bar, text="FPS: 0.0", font=ctk.CTkFont(size=13, weight="bold"))
        self.metric_fps.grid(row=0, column=0, padx=10, pady=10)
        
        self.metric_total = ctk.CTkLabel(self.metrics_bar, text="People Tracked: 0", font=ctk.CTkFont(size=13, weight="bold"))
        self.metric_total.grid(row=0, column=1, padx=10, pady=10)
        
        self.metric_known = ctk.CTkLabel(self.metrics_bar, text="Known Identity: 0", font=ctk.CTkFont(size=13, weight="bold"), text_color="#2ecc71")
        self.metric_known.grid(row=0, column=2, padx=10, pady=10)
        
        self.metric_unknown = ctk.CTkLabel(self.metrics_bar, text="Unknown/Unverified: 0", font=ctk.CTkFont(size=13, weight="bold"), text_color="#e74c3c")
        self.metric_unknown.grid(row=0, column=3, padx=10, pady=10)
        
        # Bottom Console Log Panel
        self.log_header = ctk.CTkLabel(self.tab_feed, text="Live Events & Threat Activity Log", font=ctk.CTkFont(size=13, weight="bold"))
        self.log_header.grid(row=2, column=0, padx=5, pady=(10, 2), sticky="w")
        
        self.log_textbox = ctk.CTkTextbox(self.tab_feed, height=130, font=ctk.CTkFont(family="Consolas", size=11))
        self.log_textbox.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        self.log_textbox.insert("end", "[INFO] Application started. Initializing dashboard...\n")
        self.log_textbox.configure(state="disabled")

    def setup_register_tab(self):
        # Configure columns: 0 (left form) and 1 (right list)
        self.tab_register.grid_columnconfigure(0, weight=3)
        self.tab_register.grid_columnconfigure(1, weight=2)
        self.tab_register.grid_rowconfigure(2, weight=1) # Row containing forms and list
        
        # Heading (spans across both columns)
        reg_title = ctk.CTkLabel(
            self.tab_register, 
            text="Add Person to Face Database", 
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#3498db"
        )
        reg_title.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 5), sticky="w")
        
        reg_desc = ctk.CTkLabel(
            self.tab_register, 
            text="Adding a photo creates a folder under 'databases/' so the Face Recognition system can verify their identity.",
            font=ctk.CTkFont(size=12),
            text_color="#bdc3c7",
            justify="left"
        )
        reg_desc.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 15), sticky="w")
        
        # Left Side: Form Container Frame
        left_container = ctk.CTkFrame(self.tab_register, fg_color="transparent")
        left_container.grid(row=2, column=0, padx=(20, 10), pady=10, sticky="nsew")
        left_container.grid_columnconfigure(0, weight=1)
        left_container.grid_rowconfigure(2, weight=1)
        
        form_frame = ctk.CTkFrame(left_container, corner_radius=8)
        form_frame.grid(row=0, column=0, pady=10, sticky="ew")
        form_frame.grid_columnconfigure(1, weight=1)
        
        # Name Entry
        lbl_name = ctk.CTkLabel(form_frame, text="Person Name:", font=ctk.CTkFont(weight="bold"))
        lbl_name.grid(row=0, column=0, padx=20, pady=20, sticky="w")
        
        self.entry_name = ctk.CTkEntry(form_frame, placeholder_text="e.g. John Doe")
        self.entry_name.grid(row=0, column=1, padx=20, pady=20, sticky="ew")
        
        # Buttons frame
        btns_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        btns_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="ew")
        btns_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.btn_capture = ctk.CTkButton(
            btns_frame, 
            text="Capture Face (From Stream)", 
            fg_color="#3498db",
            hover_color="#2980b9",
            command=self.register_face_from_stream
        )
        self.btn_capture.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.btn_browse = ctk.CTkButton(
            btns_frame, 
            text="Upload Image File", 
            fg_color="#9b59b6",
            hover_color="#8e44ad",
            command=self.register_face_from_file
        )
        self.btn_browse.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        self.reg_status_label = ctk.CTkLabel(
            left_container, 
            text="", 
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.reg_status_label.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        # Right Side: Registered Database List Frame
        right_container = ctk.CTkFrame(self.tab_register, corner_radius=8)
        right_container.grid(row=2, column=1, padx=(10, 20), pady=10, sticky="nsew")
        right_container.grid_columnconfigure(0, weight=1)
        right_container.grid_rowconfigure(1, weight=1)
        
        lbl_list_title = ctk.CTkLabel(
            right_container, 
            text="Registered Identities", 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#3498db"
        )
        lbl_list_title.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="w")
        
        self.registered_list_frame = ctk.CTkScrollableFrame(right_container, fg_color="#1a1a24")
        self.registered_list_frame.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
        
        # Initial load of registered faces
        self.refresh_registered_list()

    def setup_analytics_tab(self):
        self.tab_analytics.grid_columnconfigure(0, weight=1)
        self.tab_analytics.grid_rowconfigure(1, weight=1)
        
        # Buttons Bar
        ctrl_bar = ctk.CTkFrame(self.tab_analytics, height=50, corner_radius=8)
        ctrl_bar.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        ctrl_bar.grid_columnconfigure(0, weight=1)
        
        btn_refresh = ctk.CTkButton(
            ctrl_bar, 
            text="Refresh Analytics Charts", 
            fg_color="#3498db", 
            hover_color="#2980b9",
            font=ctk.CTkFont(weight="bold"),
            command=self.refresh_analytics_plots
        )
        btn_refresh.grid(row=0, column=0, padx=20, pady=10, sticky="e")
        
        # Analytics Plotting Frame
        self.analytics_frame = ctk.CTkFrame(self.tab_analytics, fg_color="#1a1a24", corner_radius=8)
        self.analytics_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.analytics_frame.grid_rowconfigure(0, weight=1)
        self.analytics_frame.grid_columnconfigure(0, weight=1)
        
        self.analytics_canvas = None
        self.show_empty_analytics_msg("No metrics loaded. Click 'Refresh Analytics Charts' to render metrics plots.")

    def show_empty_analytics_msg(self, message):
        if self.analytics_canvas:
            self.analytics_canvas.get_tk_widget().destroy()
            self.analytics_canvas = None
            
        # Draw placeholder label
        self.placeholder_lbl = ctk.CTkLabel(
            self.analytics_frame, 
            text=message, 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#7f8c8d"
        )
        self.placeholder_lbl.grid(row=0, column=0, padx=20, pady=20)

    # --- CONTROLS ACTION HANDLERS ---
    def on_models_loaded(self, success, status_message):
        """Callback run by ModelLoaderThread when AI models load"""
        if success:
            self.status_label.configure(text=status_message, text_color="#2ecc71")
            self.btn_start.configure(state="normal")
            
            # Start FaceRecognizer worker thread
            self.face_task_queue = queue.Queue()
            self.face_result_queue = queue.Queue()
            
            self.face_recognizer = FaceRecognizer(self.face_task_queue, self.face_result_queue)
            self.face_recognizer.start()
            
            # Start background Video Worker (but do not start stream yet)
            self.video_worker = VideoWorker(
                self.frame_queue, 
                self.log_queue, 
                self.face_task_queue, 
                self.face_result_queue, 
                self.update_status
            )
            self.video_worker.start()
            
            self.log_internal_event("[SYSTEM] YOLOv8 and InsightFace face recognition models loaded successfully.")
        else:
            self.status_label.configure(text=status_message, text_color="#e74c3c")
            self.log_internal_event(f"[ERROR] Failed to load models: {status_message}")

    def update_status(self, text):
        """Callback passed to VideoWorker to print status changes in the sidebar"""
        self.status_label.configure(text=text)
        if text == "Monitoring Active":
            self.status_label.configure(text_color="#2ecc71")
        elif text == "Connecting Camera...":
            self.status_label.configure(text_color="#f1c40f")
        else:
            self.status_label.configure(text_color="#7f8c8d")

    def start_camera_stream(self):
        if hasattr(self, 'video_worker'):
            self.video_worker.start_stream()
            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            self.log_internal_event("[SYSTEM] Starting video feed monitoring...")

    def stop_camera_stream(self):
        if hasattr(self, 'video_worker'):
            self.video_worker.stop_stream()
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.video_label.configure(image=None, text="Camera Stream Off\nClick 'Start Camera Feed' to begin.")
            self.log_internal_event("[SYSTEM] Camera feed monitoring stopped.")
            
            # Reset metrics bar
            self.metric_fps.configure(text="FPS: 0.0")
            self.metric_total.configure(text="People Tracked: 0")
            self.metric_known.configure(text="Known Identity: 0")
            self.metric_unknown.configure(text="Unknown/Unverified: 0")

    def on_conf_change(self, val):
        self.conf_lbl.configure(text=f"YOLO BBox Confidence: {val:.2f}")
        if hasattr(self, 'video_worker'):
            self.video_worker.conf_threshold = val

    def on_dist_change(self, val):
        self.dist_lbl.configure(text=f"Face Similarity Thresh: {val:.2f}")
        if hasattr(self, 'video_worker'):
            self.video_worker.dist_threshold = val

    def update_target_classes(self):
        if hasattr(self, 'video_worker'):
            active_ids = [CLASS_MAPPING[name] for name, var in self.checkbox_vars.items() if var.get()]
            self.video_worker.target_classes = active_ids

    # --- LOG UTILITIES ---
    def log_internal_event(self, message):
        """Helper to append messages directly into GUI logs console"""
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", f"{message}\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    # --- QUEUE POLLING LOOPS ---
    def poll_camera_queue(self):
        """Checks the video worker queue for new processed frames and updates the UI"""
        latest_item = None
        try:
            # Drain the queue to only render the latest frame and prevent backlog lag
            while True:
                latest_item = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
            
        if latest_item is not None:
            pil_img, stats = latest_item
            
            # Update statistics indicators
            self.metric_fps.configure(text=f"FPS: {stats['fps']:.1f}")
            self.metric_total.configure(text=f"People Tracked: {stats['total_people']}")
            self.metric_known.configure(text=f"Known Identity: {stats['known']}")
            self.metric_unknown.configure(text=f"Unknown/Unverified: {stats['unknown']}")
            
            # Keep reference to raw image and update display
            self.latest_raw_frame = stats["raw_frame"]
            
            # Resize image to fit label display dimensions while maintaining aspect ratio
            w_lbl = self.video_frame.winfo_width()
            h_lbl = self.video_frame.winfo_height()
            if w_lbl > 10 and h_lbl > 10:
                w_orig, h_orig = pil_img.size
                aspect = w_orig / h_orig
                if w_lbl / h_lbl > aspect:
                    new_h = h_lbl
                    new_w = int(h_lbl * aspect)
                else:
                    new_w = w_lbl
                    new_h = int(w_lbl / aspect)
                
                pil_img_resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
                ctk_img = ctk.CTkImage(light_image=pil_img_resized, dark_image=pil_img_resized, size=(new_w, new_h))
                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img  # Prevent garbage collection on the PhotoImage
            
        # Run poll again in 15ms (~60fps target check)
        self.after(15, self.poll_camera_queue)

    def poll_log_queue(self):
        """Checks for new event log outputs from the worker and appends them with colors"""
        try:
            while True:
                log_text, is_unknown = self.log_queue.get_nowait()
                
                self.log_textbox.configure(state="normal")
                # Highlight unknown target matches with warning markers
                if is_unknown:
                    self.log_textbox.insert("end", f"[ALERT] {log_text}\n")
                else:
                    self.log_textbox.insert("end", f"[LOG] {log_text}\n")
                
                self.log_textbox.see("end")
                self.log_textbox.configure(state="disabled")
        except queue.Empty:
            pass
            
        self.after(100, self.poll_log_queue)

    # --- FACE REGISTRATION LOGIC ---
    def register_face_from_stream(self):
        # 1. Input Check
        name = self.entry_name.get().strip()
        if not name:
            self.show_registration_status("Error: Please enter a name first.", "#e74c3c")
            return
            
        # 2. Camera Active Check
        if not hasattr(self, 'video_worker') or not self.video_worker.running or self.latest_raw_frame is None:
            self.show_registration_status("Error: Camera feed must be running to capture frame.", "#e74c3c")
            return
            
        # 3. Capture face using Haar cascade detector on the raw frame
        frame = self.latest_raw_frame.copy()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Format filename
        folder_name = name.lower().replace(" ", "_")
        target_dir = os.path.join(DEFAULT_DB_PATH, folder_name)
        
        if len(faces) == 0:
            # Revert to saving the raw full image frame if no face is automatically detected
            self.show_registration_status("Warning: Face not detected. Stand closer, look directly at camera.", "#f1c40f")
            os.makedirs(target_dir, exist_ok=True)
            filename = os.path.join(target_dir, f"{folder_name}_raw.jpg")
            cv2.imwrite(filename, frame)
            self.finalize_registration(name, target_dir)
        else:
            # Crop the largest face
            largest_face = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            x, y, w, h = largest_face
            # Expand bounding box slightly for context
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)
            
            face_roi = frame[y1:y2, x1:x2]
            face_roi_resized = cv2.resize(face_roi, (224, 224))
            
            os.makedirs(target_dir, exist_ok=True)
            filename = os.path.join(target_dir, f"{folder_name}_face.jpg")
            cv2.imwrite(filename, face_roi_resized)
            self.finalize_registration(name, target_dir)

    def register_face_from_file(self):
        # 1. Input Check
        name = self.entry_name.get().strip()
        if not name:
            self.show_registration_status("Error: Please enter a name first.", "#e74c3c")
            return
            
        # 2. Open file selector dialog
        file_path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return
            
        # Format directories
        folder_name = name.lower().replace(" ", "_")
        target_dir = os.path.join(DEFAULT_DB_PATH, folder_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # Read selected file
        img = cv2.imread(file_path)
        if img is None:
            self.show_registration_status("Error: Failed to read chosen image.", "#e74c3c")
            return
            
        # Try face extraction on uploaded image to make sure it contains a face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        filename = os.path.join(target_dir, f"{folder_name}_uploaded.jpg")
        if len(faces) > 0:
            # Crop to face
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img.shape[1], x + w + pad_x)
            y2 = min(img.shape[0], y + h + pad_y)
            face_roi = img[y1:y2, x1:x2]
            img_to_save = cv2.resize(face_roi, (224, 224))
        else:
            img_to_save = img
            
        cv2.imwrite(filename, img_to_save)
        self.finalize_registration(name, target_dir)

    def finalize_registration(self, name, target_dir):
        # Clean pickle representations cache files so InsightFace updates immediately
        self.clear_database_cache()
        
        # Clear name field
        self.entry_name.delete(0, 'end')
        
        self.show_registration_status(
            f"Success! '{name}' registered in {target_dir}.\nEmbeddings cache has been cleared; new faces will index on next monitoring scan.",
            "#2ecc71"
        )
        self.log_internal_event(f"[SYSTEM] Registered new face for: {name}")
        self.refresh_registered_list()

    def clear_database_cache(self):
        if os.path.exists(DEFAULT_DB_PATH):
            for file in os.listdir(DEFAULT_DB_PATH):
                if file.endswith(".pkl"):
                    try:
                        os.remove(os.path.join(DEFAULT_DB_PATH, file))
                        print(f"[INFO] Deleted pickle cache: {file}")
                    except Exception as e:
                        print(f"[ERROR] Failed to delete cache: {e}")
        try:
            import face_recognizer
            face_recognizer.known_faces.clear()
            print("[INFO] Cleared in-memory face recognition cache.")
        except Exception as e:
            print(f"[ERROR] Failed to clear in-memory cache: {e}")

    def show_registration_status(self, text, color):
        self.reg_status_label.configure(text=text, text_color=color)

    def refresh_registered_list(self):
        """Reloads the list of registered faces in the sidebar of the registration tab"""
        # Clear previous widgets
        for widget in self.registered_list_frame.winfo_children():
            widget.destroy()
            
        # Scan databases directory directly (ignoring train/validation structural roots)
        names = []
        if os.path.exists(DEFAULT_DB_PATH):
            for item in os.listdir(DEFAULT_DB_PATH):
                full_path = os.path.join(DEFAULT_DB_PATH, item)
                if os.path.isdir(full_path) and item not in ["train", "validation"]:
                    # check if folder has at least one image file
                    try:
                        has_img = any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) for f in os.listdir(full_path))
                        if has_img:
                            display_name = item.replace("_", " ").title()
                            names.append((display_name, full_path))
                    except Exception as e:
                        pass
                        
        if not names:
            lbl = ctk.CTkLabel(self.registered_list_frame, text="No registered faces found.", font=ctk.CTkFont(size=12, slant="italic"))
            lbl.pack(pady=20)
            return
            
        for display_name, folder_path in sorted(names):
            try:
                num_photos = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            except:
                num_photos = 0
                
            row = ctk.CTkFrame(self.registered_list_frame, fg_color="transparent")
            row.pack(fill="x", padx=5, pady=4)
            
            lbl_bullet = ctk.CTkLabel(row, text="\U0001f464", font=ctk.CTkFont(size=14))
            lbl_bullet.pack(side="left", padx=(5, 5))
            
            display_text = f"{display_name} ({num_photos} photos)"
            lbl_name = ctk.CTkLabel(row, text=display_text, font=ctk.CTkFont(size=12, weight="bold"))
            lbl_name.pack(side="left")
            
            btn_delete = ctk.CTkButton(
                row, 
                text="Remove", 
                width=55, 
                height=20, 
                fg_color="#e74c3c", 
                hover_color="#c0392b", 
                font=ctk.CTkFont(size=10, weight="bold"),
                command=lambda p=folder_path, n=display_name: self.delete_registered_face(p, n)
            )
            btn_delete.pack(side="right", padx=5)

    def delete_registered_face(self, folder_path, display_name):
        """Removes a registered face folder and clears the database cache"""
        if os.path.exists(folder_path):
            try:
                # Stop camera stream first if deleting to avoid file-lock or race conditions
                stream_was_active = False
                if hasattr(self, 'video_worker') and self.video_worker.running:
                    stream_was_active = True
                    self.stop_camera_stream()
                    self.log_internal_event("[SYSTEM] Stopping camera feed to safely update database...")
                    time.sleep(1.0)
                
                shutil.rmtree(folder_path)
                self.clear_database_cache()
                self.log_internal_event(f"[SYSTEM] Deleted face registration for: {display_name}")
                self.refresh_registered_list()
                self.show_registration_status(f"Removed '{display_name}' from database.", "#2ecc71")
                
                # Relaunch stream if it was active
                if stream_was_active:
                    self.start_camera_stream()
            except Exception as e:
                self.show_registration_status(f"Error deleting face: {str(e)}", "#e74c3c")

    # --- ANALYTICS Tab LOGIC ---
    def refresh_analytics_plots(self):
        if not os.path.exists(DEFAULT_LOG_FILE):
            self.show_empty_analytics_msg("Log file 'logs/monitoring_events.csv' not found.\nStart the monitoring stream to capture and log threat metrics first.")
            return
            
        try:
            # Read metrics logs file
            df = pd.read_csv(DEFAULT_LOG_FILE, header=None, names=["Timestamp", "Event", "Person_Name", "Item_Class", "Confidence"])
            
            if df.empty:
                self.show_empty_analytics_msg("No events in logs file.")
                return
                
            # Create a combined matplotlib figure
            fig = Figure(figsize=(10, 5), dpi=100, facecolor="#1a1a24")
            
            # Subplot 1: Distribution of Event Types
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_facecolor("#1a1a24")
            event_counts = df['Event'].value_counts()
            
            # Using basic matplotlib bar chart for clean styling
            colors = ["#3498db", "#2ecc71", "#e74c3c", "#f1c40f"]
            bars = ax1.bar(event_counts.index, event_counts.values, color=colors[:len(event_counts)])
            ax1.set_title("System Event Types Count", color="white", fontsize=11, fontweight="bold")
            ax1.set_ylabel("Occurrences", color="white", fontsize=9)
            ax1.tick_params(colors="white", labelsize=8)
            ax1.spines['bottom'].set_color('#7f8c8d')
            ax1.spines['top'].set_color('none')
            ax1.spines['right'].set_color('none')
            ax1.spines['left'].set_color('#7f8c8d')
            
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', color="white", fontsize=8)
            
            # Subplot 2: Identified Persons frequencies
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.set_facecolor("#1a1a24")
            person_counts = df['Person_Name'].value_counts()
            
            wedges, texts, autotexts = ax2.pie(
                person_counts.values, 
                labels=person_counts.index, 
                autopct='%1.1f%%', 
                startangle=140, 
                textprops=dict(color="white", fontsize=7)
            )
            ax2.set_title("Tracking Identity Frequency", color="white", fontsize=11, fontweight="bold")
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(7)
                
            fig.tight_layout()
            
            # Clear placeholder
            if hasattr(self, 'placeholder_lbl'):
                self.placeholder_lbl.destroy()
                
            # Embed Matplotlib Canvas in CustomTkinter analytics frame
            if self.analytics_canvas:
                self.analytics_canvas.get_tk_widget().destroy()
                
            self.analytics_canvas = FigureCanvasTkAgg(fig, master=self.analytics_frame)
            canvas_widget = self.analytics_canvas.get_tk_widget()
            canvas_widget.config(bg="#1a1a24")
            canvas_widget.grid(row=0, column=0, sticky="nsew")
            self.analytics_canvas.draw()
            
            self.log_internal_event("[SYSTEM] Refreshed analytics graphs from event logs.")
        except Exception as e:
            self.show_empty_analytics_msg(f"Error parsing log file: {str(e)}")

# Database Migration to flat structure
def migrate_database_folders():
    """Migrates any subfolders under databases/train/ and databases/validation/ directly to databases/"""
    db_path = DEFAULT_DB_PATH
    if not os.path.exists(db_path):
        return
        
    # Delete any pickle cache files on startup to ensure InsightFace rebuilds representations
    for file in os.listdir(db_path):
        if file.endswith(".pkl"):
            try:
                os.remove(os.path.join(db_path, file))
                print(f"[INFO] Deleted old pickle cache: {file}")
            except Exception as e:
                print(f"[ERROR] Failed to delete cache {file}: {e}")
                
    for sub in ["train", "validation"]:
        sub_path = os.path.join(db_path, sub)
        if os.path.exists(sub_path) and os.path.isdir(sub_path):
            for item in os.listdir(sub_path):
                item_path = os.path.join(sub_path, item)
                if os.path.isdir(item_path):
                    # Destination is databases/item
                    dest_path = os.path.join(db_path, item)
                    os.makedirs(dest_path, exist_ok=True)
                    
                    # Move all files
                    try:
                        for file in os.listdir(item_path):
                            src_file = os.path.join(item_path, file)
                            dest_file = os.path.join(dest_path, file)
                            if os.path.isfile(src_file):
                                if os.path.exists(dest_file):
                                    base, ext = os.path.splitext(file)
                                    dest_file = os.path.join(dest_path, f"{base}_alt{ext}")
                                shutil.move(src_file, dest_file)
                        shutil.rmtree(item_path)
                    except Exception as e:
                        print(f"[ERROR] Failed migrating folder {item}: {e}")
            # Try to delete the subfolder (train/validation)
            try:
                os.rmdir(sub_path)
                print(f"[INFO] Removed structural folder: {sub_path}")
            except Exception as e:
                pass

# Close routine
def on_closing(app):
    print("[INFO] Shutting down application...")
    if hasattr(app, 'video_worker'):
        app.video_worker.stop_stream()
    app.destroy()
    sys.exit(0)

if __name__ == "__main__":
    migrate_database_folders()
    app = SmartMonitoringApp()
    app.protocol("WM_DELETE_WINDOW", lambda: on_closing(app))
    app.mainloop()
