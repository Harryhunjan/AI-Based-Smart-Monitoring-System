import argparse
import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import face_recognizer
from logger import EventLogger


TV_WINDOW = "AI Smart Monitoring - Live Output"
DASHBOARD_WINDOW = "AI Smart Monitoring - Dashboard"

TARGET_CLASSES = [
    0,   # person
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    24,  # backpack
    26,  # handbag
    28,  # suitcase
    39,  # bottle
    63,  # laptop
    67,  # cell phone
]

STATIONARY_THRESH_PX = 20
STATIONARY_FRAMES_REQUIRED = 30
DISAPPEAR_TIMEOUT = 2.0
MAX_RECENT_ALERTS = 8


@dataclass
class Detection:
    box: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int]


@dataclass
class PersonTrack:
    name: str = "Unknown"
    score: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    last_seen: float = 0.0
    entry_logged: bool = False
    identified_logged: bool = False


@dataclass
class ObjectTrack:
    class_name: str
    bbox: Tuple[int, int, int, int]
    linked_person: Optional[int] = None
    positions: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=STATIONARY_FRAMES_REQUIRED))
    stationary: bool = False
    lost_alert_logged: bool = False
    last_seen: float = 0.0


@dataclass
class PipelineStats:
    source_type: str
    source_label: str
    connection_status: str = "CONNECTING"
    fps: float = 0.0
    total_people: int = 0
    known_people: int = 0
    unknown_people: int = 0
    object_counts: Counter = field(default_factory=Counter)
    recent_alerts: Deque[str] = field(default_factory=lambda: deque(maxlen=MAX_RECENT_ALERTS))
    frame_index: int = 0
    face_enabled: bool = True
    last_error: str = ""


class CameraSource:
    """Camera-source layer: converts CLI source options into an OpenCV capture."""

    def __init__(
        self,
        source_type: str,
        url: Optional[str] = None,
        camera_index: int = 0,
        file_path: Optional[str] = None,
        retry_delay: float = 2.0,
    ):
        self.source_type = source_type
        self.url = url
        self.camera_index = camera_index
        self.file_path = file_path
        self.retry_delay = retry_delay
        self.capture: Optional[cv2.VideoCapture] = None
        self.source = self._build_source()
        self.source_label = self._build_label()
        self.reconnectable = self.source_type in {"phone", "rtsp", "usb"}

    def _build_source(self):
        if self.source_type == "phone":
            if not self.url:
                raise ValueError("--url is required when --source-type phone is used.")
            return normalize_phone_stream_url(self.url)
        if self.source_type == "rtsp":
            if not self.url:
                raise ValueError("--url is required when --source-type rtsp is used.")
            return self.url.strip()
        if self.source_type == "usb":
            return int(self.camera_index)
        if self.source_type == "file":
            if not self.file_path:
                raise ValueError("--file-path is required when --source-type file is used.")
            path = Path(self.file_path)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {path}")
            return str(path)
        raise ValueError(f"Unsupported source type: {self.source_type}")

    def _build_label(self) -> str:
        if self.source_type == "usb":
            return f"USB camera index {self.camera_index}"
        return str(self.source)

    def open(self) -> cv2.VideoCapture:
        while True:
            self.release()
            self.capture = cv2.VideoCapture(self.source)
            if self.capture.isOpened():
                return self.capture
            self.release()
            if not self.reconnectable:
                raise RuntimeError(f"Could not open video source: {self.source_label}")
            print(f"[WARN] Could not open {self.source_label}. Retrying in {self.retry_delay:.0f}s...")
            time.sleep(self.retry_delay)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.capture is None or not self.capture.isOpened():
            self.open()
        assert self.capture is not None
        return self.capture.read()

    def reconnect(self) -> bool:
        if not self.reconnectable:
            return False
        self.open()
        return True

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class FaceWorker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.busy_ids = set()
        self.results: Dict[int, Tuple[str, float, float]] = {}
        self.lock = threading.Lock()

    def recognize_async(self, person_crop: np.ndarray, track_id: int) -> None:
        if not self.enabled or track_id is None:
            return
        if person_crop is None or person_crop.size == 0:
            return
        with self.lock:
            if track_id in self.busy_ids:
                return
            self.busy_ids.add(track_id)

        thread = threading.Thread(
            target=self._recognize,
            args=(person_crop.copy(), int(track_id)),
            daemon=True,
        )
        thread.start()

    def _recognize(self, person_crop: np.ndarray, track_id: int) -> None:
        try:
            face_recognizer.initialize_model()
            face_recognizer.load_known_faces()
            name, score = face_recognizer.recognize_face(person_crop, track_id)
            with self.lock:
                self.results[track_id] = (name, score, time.time())
        except Exception as exc:
            print(f"[FACE] Recognition skipped for track {track_id}: {exc}")
        finally:
            with self.lock:
                self.busy_ids.discard(track_id)

    def get_result(self, track_id: int) -> Tuple[str, float]:
        with self.lock:
            result = self.results.get(int(track_id))
        if not result:
            return "Unknown", 0.0
        return result[0], result[1]


class AIProcessor:
    """AI processing layer: YOLO tracking, optional face recognition, and alert state."""

    def __init__(self, model_path: str, conf: float, detect_every: int, face_enabled: bool):
        print("[INFO] Loading YOLO model...")
        self.model = YOLO(model_path)
        self.conf = conf
        self.detect_every = max(1, detect_every)
        self.face_worker = FaceWorker(enabled=face_enabled)
        self.people: Dict[int, PersonTrack] = {}
        self.objects: Dict[int, ObjectTrack] = {}
        self.last_results = None
        self.event_logger = EventLogger()

    def process(self, frame: np.ndarray, stats: PipelineStats) -> Tuple[np.ndarray, Optional[str]]:
        stats.frame_index += 1
        should_detect = stats.frame_index % self.detect_every == 0 or self.last_results is None
        if should_detect:
            self.last_results = self.model.track(
                frame,
                persist=True,
                classes=TARGET_CLASSES,
                conf=self.conf,
                verbose=False,
            )

        detections = self._extract_detections(frame)
        now = time.time()
        alert_banner = self._update_tracks(frame, detections, now, stats)
        self._cleanup_tracks(now)
        self._update_stats(stats)

        output_frame = frame.copy()
        self._draw_clean_output(output_frame, detections, alert_banner)
        return output_frame, alert_banner

    def _extract_detections(self, frame: np.ndarray) -> List[Detection]:
        if not self.last_results or self.last_results[0].boxes is None:
            return []

        boxes_obj = self.last_results[0].boxes
        if boxes_obj.xyxy is None or boxes_obj.cls is None:
            return []

        boxes = boxes_obj.xyxy.cpu().numpy()
        classes = boxes_obj.cls.int().cpu().numpy()
        confs = boxes_obj.conf.cpu().numpy()
        ids = boxes_obj.id.int().cpu().numpy() if boxes_obj.id is not None else [None] * len(boxes)

        h, w = frame.shape[:2]
        detections: List[Detection] = []
        for box, class_id, confidence, track_id in zip(boxes, classes, confs, ids):
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    class_id=int(class_id),
                    class_name=self.model.names[int(class_id)],
                    confidence=float(confidence),
                    track_id=int(track_id) if track_id is not None else None,
                )
            )
        return detections

    def _update_tracks(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        now: float,
        stats: PipelineStats,
    ) -> Optional[str]:
        current_person_ids = []
        alert_banner = None

        for detection in detections:
            if detection.class_name != "person" or detection.track_id is None:
                continue

            current_person_ids.append(detection.track_id)
            person = self.people.setdefault(detection.track_id, PersonTrack(last_seen=now))
            person.bbox = detection.box
            person.last_seen = now

            if not person.entry_logged:
                self.event_logger.log_event("ENTRY", person_name=person.name, confidence=f"{detection.confidence:.2f}")
                person.entry_logged = True

            x1, y1, x2, y2 = detection.box
            self.face_worker.recognize_async(frame[y1:y2, x1:x2], detection.track_id)
            name, score = self.face_worker.get_result(detection.track_id)
            if name != person.name or score != person.score:
                person.name = name
                person.score = score
            if name != "Unknown" and not person.identified_logged:
                self.event_logger.log_event("IDENTIFIED", person_name=name, confidence=f"{score:.2f}")
                self._add_alert(stats, f"{name} identified")
                person.identified_logged = True

        for detection in detections:
            if detection.class_name == "person" or detection.track_id is None:
                continue

            obj = self.objects.setdefault(
                detection.track_id,
                ObjectTrack(class_name=detection.class_name, bbox=detection.box, last_seen=now),
            )
            obj.class_name = detection.class_name
            obj.bbox = detection.box
            obj.last_seen = now

            cx, cy = box_center(detection.box)
            obj.positions.append((cx, cy))
            if len(obj.positions) == STATIONARY_FRAMES_REQUIRED:
                obj.stationary = max(distance((cx, cy), p) for p in obj.positions) < STATIONARY_THRESH_PX
                if not obj.stationary:
                    obj.lost_alert_logged = False

            if not obj.stationary:
                obj.linked_person = self._find_nearest_person(obj.bbox, current_person_ids)

            if obj.stationary and obj.linked_person is not None and self._owner_is_far_or_missing(obj, now):
                owner_name = self.people.get(obj.linked_person, PersonTrack()).name
                alert_banner = f"ALERT: {obj.class_name.title()} left behind by {owner_name}"
                if not obj.lost_alert_logged:
                    self.event_logger.log_event("ITEM_LEFT_BEHIND", person_name=owner_name, item_class=obj.class_name)
                    self._add_alert(stats, alert_banner)
                    obj.lost_alert_logged = True

        return alert_banner

    def _find_nearest_person(self, obj_box: Tuple[int, int, int, int], current_person_ids: List[int]) -> Optional[int]:
        if not current_person_ids:
            return None

        ox, oy = box_center(obj_box)
        closest_person = None
        closest_distance = float("inf")
        for person_id in current_person_ids:
            person = self.people.get(person_id)
            if person is None:
                continue
            px, py = box_center(person.bbox)
            d = distance((ox, oy), (px, py))
            if d < closest_distance:
                closest_distance = d
                closest_person = person_id

        if closest_person is None:
            return None
        person_box = self.people[closest_person].bbox
        obj_width = obj_box[2] - obj_box[0]
        person_width = person_box[2] - person_box[0]
        if closest_distance < max(obj_width, person_width) * 1.5:
            return closest_person
        return None

    def _owner_is_far_or_missing(self, obj: ObjectTrack, now: float) -> bool:
        person = self.people.get(obj.linked_person) if obj.linked_person is not None else None
        if person is None or now - person.last_seen > DISAPPEAR_TIMEOUT:
            return True
        px, py = box_center(person.bbox)
        ox, oy = box_center(obj.bbox)
        person_width = person.bbox[2] - person.bbox[0]
        return distance((px, py), (ox, oy)) > person_width + 200

    def _cleanup_tracks(self, now: float) -> None:
        for person_id in list(self.people.keys()):
            person = self.people[person_id]
            if now - person.last_seen > DISAPPEAR_TIMEOUT:
                self.event_logger.log_event("EXIT", person_name=person.name)
                del self.people[person_id]

        for object_id in list(self.objects.keys()):
            if now - self.objects[object_id].last_seen > DISAPPEAR_TIMEOUT * 2:
                del self.objects[object_id]

    def _update_stats(self, stats: PipelineStats) -> None:
        stats.total_people = len(self.people)
        stats.known_people = sum(1 for person in self.people.values() if person.name != "Unknown")
        stats.unknown_people = stats.total_people - stats.known_people
        stats.object_counts = Counter(obj.class_name for obj in self.objects.values())

    def _draw_clean_output(self, frame: np.ndarray, detections: List[Detection], alert_banner: Optional[str]) -> None:
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            if detection.class_name == "person":
                name, score = self._person_label(detection.track_id)
                label = name if name == "Unknown" else f"{name} {score * 100:.0f}%"
                color = (0, 180, 0) if name != "Unknown" else (0, 0, 255)
            else:
                label = f"{detection.class_name} {detection.confidence:.2f}"
                color = (255, 130, 0)
                obj = self.objects.get(detection.track_id)
                if obj and obj.stationary and obj.lost_alert_logged:
                    color = (0, 0, 255)
                    label = f"LEFT BEHIND: {detection.class_name}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, label, x1, y1, color)

        if alert_banner:
            draw_alert_banner(frame, alert_banner)

    def _person_label(self, track_id: Optional[int]) -> Tuple[str, float]:
        if track_id is None:
            return "Unknown", 0.0
        person = self.people.get(track_id)
        if person is None:
            return "Unknown", 0.0
        return person.name, person.score

    def _add_alert(self, stats: PipelineStats, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        stats.recent_alerts.appendleft(f"{stamp}  {text}")
        print(f"[ALERT] {stamp} | {text}")


class DisplayOutput:
    """Display output layer: clean TV feed plus optional laptop dashboard window."""

    def __init__(self, fullscreen: bool, show_dashboard: bool, tv_x: Optional[int], tv_y: Optional[int]):
        self.fullscreen = fullscreen
        self.show_dashboard = show_dashboard
        cv2.namedWindow(TV_WINDOW, cv2.WINDOW_NORMAL)
        if tv_x is not None and tv_y is not None:
            cv2.moveWindow(TV_WINDOW, tv_x, tv_y)
        if self.fullscreen:
            cv2.setWindowProperty(TV_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        if self.show_dashboard:
            cv2.namedWindow(DASHBOARD_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(DASHBOARD_WINDOW, 520, 720)

    def show(self, output_frame: np.ndarray, stats: PipelineStats) -> int:
        cv2.imshow(TV_WINDOW, output_frame)
        if self.show_dashboard:
            cv2.imshow(DASHBOARD_WINDOW, render_dashboard(stats))
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        cv2.destroyAllWindows()


def normalize_phone_stream_url(raw_url: str):
    url = raw_url.strip()
    if not url:
        raise ValueError("Phone camera URL cannot be empty.")
    if url.isdigit():
        return int(url)
    if url.startswith("http://") or url.startswith("https://"):
        url = url.rstrip("/")
        if not url.endswith("/video"):
            url = f"{url}/video"
    return url


def box_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    return int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)


def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    if not width:
        return frame
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / float(w)
    return cv2.resize(frame, (width, int(h * scale)))


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - height - baseline - 8)
    cv2.rectangle(frame, (x, top), (x + width + 10, y), color, -1)
    cv2.putText(frame, text, (x + 5, y - 6), font, scale, (255, 255, 255), thickness)


def draw_alert_banner(frame: np.ndarray, text: str) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 48), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def put_dashboard_line(
    canvas: np.ndarray,
    text: str,
    y: int,
    color: Tuple[int, int, int] = (230, 230, 230),
    scale: float = 0.58,
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_dashboard(stats: PipelineStats) -> np.ndarray:
    canvas = np.zeros((720, 520, 3), dtype=np.uint8)
    canvas[:] = (24, 26, 30)
    cv2.rectangle(canvas, (0, 0), (520, 76), (40, 45, 52), -1)
    put_dashboard_line(canvas, "AI Smart Monitoring Dashboard", 34, (255, 255, 255), 0.72, 2)
    put_dashboard_line(canvas, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 62, (185, 195, 205), 0.5)

    status_color = (80, 220, 120) if stats.connection_status == "CONNECTED" else (0, 170, 255)
    if stats.connection_status == "ERROR":
        status_color = (0, 0, 255)

    y = 112
    put_dashboard_line(canvas, f"Connection: {stats.connection_status}", y, status_color, 0.64, 2)
    y += 30
    put_dashboard_line(canvas, f"Source type: {stats.source_type}", y)
    y += 26
    put_dashboard_line(canvas, f"Source: {stats.source_label[:58]}", y)
    y += 26
    put_dashboard_line(canvas, f"FPS: {stats.fps:.1f}   Frame: {stats.frame_index}", y)
    y += 26
    put_dashboard_line(canvas, f"Face recognition: {'ON' if stats.face_enabled else 'OFF'}", y)

    y += 48
    put_dashboard_line(canvas, "People", y, (255, 255, 255), 0.64, 2)
    y += 30
    put_dashboard_line(canvas, f"Total: {stats.total_people}", y)
    y += 26
    put_dashboard_line(canvas, f"Known: {stats.known_people}", y, (90, 220, 130))
    y += 26
    put_dashboard_line(canvas, f"Unknown: {stats.unknown_people}", y, (80, 120, 255))

    y += 48
    put_dashboard_line(canvas, "Objects", y, (255, 255, 255), 0.64, 2)
    y += 30
    if stats.object_counts:
        for class_name, count in stats.object_counts.most_common(6):
            put_dashboard_line(canvas, f"{class_name}: {count}", y)
            y += 24
    else:
        put_dashboard_line(canvas, "No tracked objects", y, (165, 170, 178))
        y += 24

    y = 560
    put_dashboard_line(canvas, "Recent Alerts", y, (255, 255, 255), 0.64, 2)
    y += 30
    if stats.recent_alerts:
        for alert in list(stats.recent_alerts)[:5]:
            put_dashboard_line(canvas, alert[:58], y, (80, 180, 255), 0.5)
            y += 24
    else:
        put_dashboard_line(canvas, "No alerts yet", y, (165, 170, 178), 0.5)

    put_dashboard_line(canvas, "Controls: q/Esc exit | f fullscreen", 700, (150, 160, 170), 0.48)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Source-independent live AI monitoring pipeline with clean TV output and laptop dashboard."
    )
    parser.add_argument(
        "--source-type",
        choices=["phone", "rtsp", "usb", "file"],
        required=True,
        help="Input source type: phone HTTP stream, RTSP camera, USB webcam, or video file.",
    )
    parser.add_argument("--url", help="HTTP or RTSP stream URL.")
    parser.add_argument("--camera-index", type=int, default=0, help="USB webcam index.")
    parser.add_argument("--file-path", help="Path to a video file for testing.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path.")
    parser.add_argument("--width", type=int, default=960, help="Processing/display width. Use 0 for native width.")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO detection confidence threshold.")
    parser.add_argument("--detect-every", type=int, default=2, help="Run YOLO every N frames.")
    parser.add_argument("--no-face", action="store_true", help="Disable face recognition.")
    parser.add_argument("--fullscreen", action="store_true", help="Open the clean output window in fullscreen.")
    parser.add_argument("--no-dashboard-window", action="store_true", help="Use terminal logs only for laptop status.")
    parser.add_argument("--tv-x", type=int, help="Optional x-position for the TV output window.")
    parser.add_argument("--tv-y", type=int, help="Optional y-position for the TV output window.")
    parser.add_argument("--log-every", type=float, default=2.0, help="Seconds between terminal status logs.")
    return parser.parse_args()


def print_status(stats: PipelineStats) -> None:
    objects = ", ".join(f"{name}:{count}" for name, count in stats.object_counts.most_common()) or "none"
    print(
        "[STATUS] "
        f"{stats.connection_status} | FPS {stats.fps:.1f} | "
        f"people total={stats.total_people} known={stats.known_people} unknown={stats.unknown_people} | "
        f"objects {objects}"
    )


def main() -> None:
    args = parse_args()
    source = CameraSource(
        source_type=args.source_type,
        url=args.url,
        camera_index=args.camera_index,
        file_path=args.file_path,
    )
    stats = PipelineStats(
        source_type=args.source_type,
        source_label=source.source_label,
        face_enabled=not args.no_face,
    )

    processor = AIProcessor(
        model_path=args.model,
        conf=args.conf,
        detect_every=args.detect_every,
        face_enabled=not args.no_face,
    )
    display = DisplayOutput(
        fullscreen=args.fullscreen,
        show_dashboard=not args.no_dashboard_window,
        tv_x=args.tv_x,
        tv_y=args.tv_y,
    )

    print(f"[INFO] Opening {args.source_type} source: {source.source_label}")
    source.open()
    stats.connection_status = "CONNECTED"
    print("[INFO] Pipeline ready. Press q or Esc to exit.")

    fps_frames = 0
    fps_started_at = time.time()
    last_status_at = 0.0

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                if source.reconnectable:
                    stats.connection_status = "RECONNECTING"
                    print("[WARN] Frame unavailable. Reconnecting...")
                    source.reconnect()
                    stats.connection_status = "CONNECTED"
                    continue
                print("[INFO] End of video file or frame unavailable.")
                break

            frame = resize_frame(frame, args.width)
            output_frame, _ = processor.process(frame, stats)

            fps_frames += 1
            now = time.time()
            if now - fps_started_at >= 1.0:
                stats.fps = fps_frames / (now - fps_started_at)
                fps_frames = 0
                fps_started_at = now

            if now - last_status_at >= max(0.5, args.log_every):
                print_status(stats)
                last_status_at = now

            key = display.show(output_frame, stats)
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                cv2.setWindowProperty(TV_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    finally:
        source.release()
        display.close()


if __name__ == "__main__":
    main()
