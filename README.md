# 🚗 License Plate Monitor

A web application that detects vehicle license plates in recorded videos or a live webcam, crops each plate, reads it with OCR and saves the records to a CSV file. Built with a **YOLOv8 model we trained ourselves**, OpenCV, EasyOCR and Streamlit.

```
Video / Webcam → YOLO detection → plate extraction → OCR → CSV storage
```

## Features
- Upload a recorded vehicle or road video (mp4, avi, mov, mkv)
- Live webcam monitoring
- Plate detection with our trained YOLO model (`model/best.pt`)
- Each plate is cropped and saved as an image
- Plate text read with EasyOCR
- **One record per plate**: a simple tracker links detections across frames, so a plate visible for 3 seconds is saved once, not 75 times
- Records tab with a table, plate crops, CSV download and a clear button

## Possible use
No-parking vehicle monitoring, parking management, vehicle entry/exit logging or traffic monitoring.

## Project structure
```
├── app.py                              # Streamlit web app
├── plate_utils.py                      # detection, tracking, OCR, CSV helpers
├── train_yolo_plate_detector.ipynb     # Colab notebook that trains the YOLO model
├── model/best.pt                       # our trained weights
├── records/
│   ├── plate_records.csv               # saved detections (auto-generated)
│   └── plates/                         # cropped plate images (auto-generated)
└── requirements.txt
```

## How it works
1. **Training** (`train_yolo_plate_detector.ipynb`): fine-tunes `yolov8n` on a labeled license-plate dataset in Google Colab and exports `best.pt`.
2. **Detection:** for every processed frame, the model returns boxes around plates with a confidence score.
3. **Tracking:** boxes that overlap with the previous frame's boxes are treated as the same plate. Short-lived detections (fewer than the "frames to confirm" setting) are ignored as false alarms.
4. **Extraction:** for each plate the sharpest crop (large box, high confidence) is kept.
5. **OCR:** the crop is converted to grayscale, enlarged and contrast-boosted, then read with EasyOCR.
6. **Storage:** one row per plate is appended to `records/plate_records.csv`.

CSV columns: `record_id, timestamp, source, plate_text, ocr_confidence, detection_confidence, plate_image`.

## Model and dataset
- Model: YOLOv8n fine-tuned for one class (`license_plate`)
- Dataset: **Car License Plate Detection** by Larxel, from Kaggle
- Results on the validation set: **Precision ___, Recall ___, mAP50 ___**

## How to run
Tested on Windows with Python 3.13. You need about 4 GB of free disk space for the libraries (PyTorch is large).

```bash
# 1. the trained model is already included at model/best.pt
#    (to retrain it, open train_yolo_plate_detector.ipynb in Google Colab,
#     run all cells and replace model/best.pt with the downloaded file)

# 2. create a virtual environment
python -m venv venv
venv\Scripts\activate           # Mac/Linux: source venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt

# 4. start the app
streamlit run app.py
```
The app opens at **http://localhost:8501**. The first run also downloads the EasyOCR text-reading models (about 100 MB).

## Settings (sidebar)
| Setting | What it does |
|---|---|
| Detection confidence | Higher = fewer false detections but may miss plates |
| Process every Nth frame | Higher = faster but may miss quick vehicles |
| Frames needed to confirm | Ignores one-frame false alarms |
| OCR language | Language(s) EasyOCR should read |

## Limitations
- OCR accuracy depends on video quality, angle and plate style; the plate crop is always saved, so unread plates can still be reviewed.
- At low confidence settings the model can box things that are not plates (for example railings or signs). Raising the detection confidence reduces this.
- Tracking is simple overlap matching, so two plates crossing each other may be confused.
- Trained on a relatively small public dataset.

## References
- Dataset: Car License Plate Detection (Larxel), Kaggle.
- Referenced a classmate's License-Plate-Detector repo (with permission) to check the assignment requirements; the code, model and training were written separately.
- Libraries: Ultralytics YOLOv8, OpenCV, EasyOCR, Streamlit.

## Tech stack
Python · YOLOv8 (Ultralytics) · OpenCV · EasyOCR · Streamlit · Pandas · Git/GitHub