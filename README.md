# OCR Text Extraction - EasyOCR Streamlit App

**Project**: AI/ML Developer Assessment – OCR Text Extraction.  
**Goal**: Extract lines containing the target pattern (e.g. `_1_`) from shipping label / waybill images with high accuracy using local, open-source tools only. See the provided assessment instructions for full requirements. fileciteturn0file0

## Structure
```
project-root/
├── README.md
├── requirements.txt
├── src/
│   ├── ocr_engine.py
│   ├── preprocessing.py
│   ├── text_extraction.py
│   └── utils.py
├── app.py
├── tests/
│   └── test_ocr.py
├── notebooks/
└── results/
```

## Quick setup (recommended inside a virtual environment)
1. Create venv and activate:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note**: EasyOCR depends on PyTorch. Installing `torch` can download platform-specific wheels. If you have disk constraints, consider installing `torch` with the CPU-only wheel appropriate for your platform (see PyTorch website).

## Run the Streamlit app
```bash
streamlit run app.py
```

## Files overview
- `src/preprocessing.py` — image preprocessing functions (resize, denoise, binarize, deskew helper).
- `src/ocr_engine.py` — wraps EasyOCR Reader and returns detected lines with confidences.
- `src/text_extraction.py` — logic to find the target line(s) containing the `_1_` pattern or digit '1' tokens.
- `src/utils.py` — small utilities for drawing and saving results.
- `app.py` — Streamlit front-end for uploading an image, running OCR, and showing highlighted target line and confidences.
- `tests/test_ocr.py` — unit test creating a synthetic image and validating extraction logic.

## Accuracy & Evaluation
- The code is built to prioritize accuracy: heavy preprocessing and per-box recognition are used.
- The delivered project does not include the private test dataset; run the app on the provided dataset and place results in `results/` for reporting.
- You should compute accuracy on your test set by comparing extracted target lines against ground-truth labels and reporting the percentage of exact matches.

## Future improvements
- Add a small KIE (key-value extraction) model to robustly extract invoice fields.
- Fine-tune a lightweight recognition model for domain-specific fonts (if allowed).
- Add multiprocessing/batch processing for larger datasets.
