"""Helper functions for the License Plate Monitor.

Pipeline:  frame -> YOLO detection -> track each plate across frames ->
           keep the best crop of every plate -> OCR -> one CSV row per plate.
"""
import csv
import re
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RECORDS_DIR = BASE_DIR / "records"
PLATES_DIR = RECORDS_DIR / "plates"
CSV_PATH = RECORDS_DIR / "plate_records.csv"
CSV_FIELDS = ["record_id", "timestamp", "source", "plate_text",
              "ocr_confidence", "detection_confidence", "plate_image"]

PLATES_DIR.mkdir(parents=True, exist_ok=True)
_csv_lock = threading.Lock()


# ---------------------------------------------------------------- detection
def detect_plates(model, frame, conf=0.35, imgsz=640):
    """Run the trained YOLO model on one frame.
    Returns a list of (x1, y1, x2, y2, confidence)."""
    result = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    found = []
    for box in result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        found.append((x1, y1, x2, y2, float(box.conf[0])))
    return found


# ----------------------------------------------------------------- tracking
def iou(a, b):
    """Intersection-over-union of two boxes (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


class PlateTracker:
    """Matches detections between frames so one physical plate = one record.

    Without this, a plate visible for 3 seconds would be saved 75 times.
    """

    def __init__(self, iou_threshold=0.25, max_age=15, min_hits=3):
        self.iou_threshold = iou_threshold  # overlap needed to be "same plate"
        self.max_age = max_age              # frames a plate may vanish
        self.min_hits = min_hits            # frames needed to count as real
        self.tracks = {}
        self._next_id = 1

    def update(self, frame_idx, frame, detections):
        """Add this frame's detections. Returns [(track_id, box, conf)] for drawing,
        plus a list of tracks that just finished."""
        unmatched = list(range(len(detections)))
        shown = []

        # greedy matching: best-overlapping pairs first
        pairs = []
        for tid, tr in self.tracks.items():
            for di in unmatched:
                score = iou(tr["box"], detections[di][:4])
                if score >= self.iou_threshold:
                    pairs.append((score, tid, di))
        used_tracks = set()
        for score, tid, di in sorted(pairs, reverse=True):
            if tid in used_tracks or di not in unmatched:
                continue
            used_tracks.add(tid)
            unmatched.remove(di)
            self._refresh(self.tracks[tid], frame_idx, frame, detections[di])
            shown.append((tid, detections[di][:4], detections[di][4]))

        for di in unmatched:  # new plates
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = {"hits": 0, "best_score": -1, "best_crop": None}
            self._refresh(self.tracks[tid], frame_idx, frame, detections[di])
            shown.append((tid, detections[di][:4], detections[di][4]))

        return shown, self._collect_finished(frame_idx)

    def _refresh(self, track, frame_idx, frame, det):
        x1, y1, x2, y2, conf = det
        track.update(box=(x1, y1, x2, y2), last_seen=frame_idx)
        track["hits"] += 1
        # keep the sharpest-looking crop: big box with high confidence
        score = conf * max(1, (x2 - x1) * (y2 - y1))
        if score > track["best_score"]:
            pad = 6
            h, w = frame.shape[:2]
            crop = frame[max(0, y1 - pad):min(h, y2 + pad),
                         max(0, x1 - pad):min(w, x2 + pad)]
            if crop.size:
                track.update(best_score=score, best_crop=crop.copy(), best_conf=conf)

    def _collect_finished(self, frame_idx):
        done = []
        for tid in list(self.tracks):
            if frame_idx - self.tracks[tid]["last_seen"] > self.max_age:
                tr = self.tracks.pop(tid)
                if tr["hits"] >= self.min_hits and tr["best_crop"] is not None:
                    done.append(tr)
        return done

    def finish_all(self):
        """Call when the video ends / webcam stops."""
        done = [t for t in self.tracks.values()
                if t["hits"] >= self.min_hits and t["best_crop"] is not None]
        self.tracks.clear()
        return done


# --------------------------------------------------------------------- OCR
def prepare_for_ocr(crop):
    """Grayscale -> enlarge -> boost contrast, so small plates are readable."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = max(1.0, 160 / h)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9 \-]", "", text).upper()
    return re.sub(r"\s+", " ", text).strip()


def read_plate(reader, crop):
    """OCR one plate crop. Returns (text, confidence). Text may be empty."""
    if reader is None:
        return "", 0.0
    try:
        results = reader.readtext(prepare_for_ocr(crop))
    except Exception:
        return "", 0.0
    if not results:
        return "", 0.0
    # order pieces top-to-bottom, then left-to-right (two-line plates)
    results.sort(key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))
    text = clean_text(" ".join(r[1] for r in results))
    conf = float(np.mean([r[2] for r in results]))
    return text, conf


# --------------------------------------------------------------------- CSV
def _init_csv():
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def _next_record_id():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f)) + 1


def save_record(track, source, reader=None):
    """OCR the best crop of a finished track, save the image, append a CSV row."""
    text, ocr_conf = read_plate(reader, track["best_crop"])
    with _csv_lock:
        _init_csv()
        record_id = _next_record_id()
        stamp = datetime.now()
        image_name = f"plate_{record_id:04d}_{stamp:%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(PLATES_DIR / image_name), track["best_crop"])
        row = {
            "record_id": record_id,
            "timestamp": stamp.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "plate_text": text,
            "ocr_confidence": f"{ocr_conf:.2f}",
            "detection_confidence": f"{track['best_conf']:.2f}",
            "plate_image": image_name,
        }
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
    return row


def clear_records():
    with _csv_lock:
        for img in PLATES_DIR.glob("*.jpg"):
            img.unlink()
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


# ----------------------------------------------------------------- drawing
def draw_boxes(frame, shown):
    out = frame.copy()
    for tid, (x1, y1, x2, y2), conf in shown:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f"plate #{tid}  {conf:.2f}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return out
