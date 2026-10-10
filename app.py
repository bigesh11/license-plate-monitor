"""License Plate Monitor: Streamlit web app.

Video / Webcam -> YOLO plate detection -> crop plate -> OCR -> CSV records.
Run:  streamlit run app.py
"""
import os
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

import plate_utils as pu

MODEL_PATH = Path(__file__).parent / "model" / "best.pt"

st.set_page_config(page_title="License Plate Monitor", page_icon="🚗", layout="wide")


# ------------------------------------------------------------ cached loaders
@st.cache_resource
def load_model(path):
    from ultralytics import YOLO
    return YOLO(str(path))


@st.cache_resource
def load_ocr(languages):
    try:
        import easyocr
        return easyocr.Reader(list(languages), gpu=False)
    except Exception as err:  # app still works without OCR
        st.warning(f"OCR could not be loaded ({err}). Plate images will still be saved.")
        return None


# ------------------------------------------------------------------- header
st.title("🚗 License Plate Monitor")
st.caption("Upload a video or use your webcam. Plates are detected with our trained YOLO model, "
           "cropped, read with OCR and saved to a CSV file.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Put your trained YOLO weights at "
             "`model/best.pt` (see README, Step 1) and refresh this page.")
    st.stop()

model = load_model(MODEL_PATH)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    conf = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05,
                     help="Higher = fewer false detections, but may miss plates.")
    skip = st.slider("Process every Nth frame", 1, 5, 2,
                     help="Higher = faster, but may miss quick vehicles.")
    min_hits = st.slider("Frames needed to confirm a plate", 1, 10, 3,
                         help="Ignores one-frame false alarms.")
    use_ocr = st.checkbox("Read plate text (OCR)", value=True)
    ocr_langs = st.multiselect("OCR language", ["en", "hi", "ne"], default=["en"])

reader = load_ocr(tuple(ocr_langs or ["en"])) if use_ocr else None

tab_video, tab_cam, tab_records = st.tabs(["📁 Upload video", "📷 Live webcam", "📋 Records"])


# ------------------------------------------------------------------ shared
def run_stream(cap, source_name, frame_slot, status_slot, total_frames=0, progress=None):
    """Detect plates frame by frame; save one record per plate."""
    tracker = pu.PlateTracker(max_age=15, min_hits=min_hits)
    saved = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % skip:
                continue
            dets = pu.detect_plates(model, frame, conf=conf)
            shown, finished = tracker.update(idx, frame, dets)
            for tr in finished:
                saved.append(pu.save_record(tr, source_name, reader))
            frame_slot.image(cv2.cvtColor(pu.draw_boxes(frame, shown), cv2.COLOR_BGR2RGB),
                             channels="RGB", use_container_width=True)
            status_slot.info(f"Frame {idx} · plates in view: {len(shown)} · saved so far: {len(saved)}")
            if progress is not None and total_frames:
                progress.progress(min(idx / total_frames, 1.0))
    finally:
        cap.release()
        for tr in tracker.finish_all():  # plates still visible when it stopped
            saved.append(pu.save_record(tr, source_name, reader))
    return saved


# --------------------------------------------------------------- video tab
with tab_video:
    video = st.file_uploader("Upload a vehicle / road video", type=["mp4", "avi", "mov", "mkv"])
    if video and st.button("▶ Start detection", type="primary"):
        suffix = Path(video.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video.read())
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        bar = st.progress(0.0)
        frame_slot, status_slot = st.empty(), st.empty()
        saved = run_stream(cap, f"video:{video.name}", frame_slot, status_slot, total, bar)
        os.unlink(tmp.name)
        bar.progress(1.0)
        st.success(f"Done. {len(saved)} plate record(s) saved. See the Records tab.")
        if saved:
            st.dataframe(pd.DataFrame(saved), use_container_width=True)


# -------------------------------------------------------------- webcam tab
with tab_cam:
    cam_index = st.number_input("Camera number", 0, 5, 0,
                                help="0 is the built-in webcam. Try 1 for an external camera.")
    run = st.toggle("Start webcam")
    frame_slot, status_slot = st.empty(), st.empty()
    if run:
        cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW) if os.name == "nt" \
            else cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error("Could not open the webcam. Close other apps using it, or try another camera number.")
        else:
            st.caption("Switch the toggle off to stop. Plates still in view are saved when you stop.")
            run_stream(cap, "webcam", frame_slot, status_slot)


# ------------------------------------------------------------ records tab
with tab_records:
    if pu.CSV_PATH.exists() and pu.CSV_PATH.stat().st_size > 0:
        df = pd.read_csv(pu.CSV_PATH, dtype=str).fillna("")
    else:
        df = pd.DataFrame(columns=pu.CSV_FIELDS)

    left, mid, right = st.columns([1, 1, 4])
    left.metric("Plates saved", len(df))
    mid.metric("Plates read", int((df["plate_text"] != "").sum()) if len(df) else 0)

    if len(df):
        st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button("⬇ Download CSV", df.to_csv(index=False).encode("utf-8"),
                           "plate_records.csv", "text/csv")
        st.subheader("Latest plate crops")
        cols = st.columns(6)
        for col, (_, row) in zip(cols, df.iloc[::-1].head(6).iterrows()):
            path = pu.PLATES_DIR / row["plate_image"]
            if path.exists():
                col.image(str(path), caption=row["plate_text"] or "(unread)")
        if st.button("🗑 Clear all records"):
            pu.clear_records()
            st.rerun()
    else:
        st.info("No records yet. Run a video or the webcam first.")
