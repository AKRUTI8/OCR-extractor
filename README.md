📦 Waybill OCR Extraction (EasyOCR + Streamlit)

This project is part of my AI/ML Developer OCR assessment, where I had to build an offline OCR system that extracts the special text pattern containing _1_ from shipping labels/waybills.

The goal was to read images of logistics labels (like Xpressbees, Delhivery, etc.), preprocess them properly, run OCR, and finally extract the correct waybill identifier such as:

156387426414724544_1_wni


Since these labels are often blurry, low contrast, and noisy, I focused heavily on preprocessing + ROI cropping to improve accuracy.

🚀 What this project does

Preprocesses the image (contrast boost, gamma correction, sharpening, deskew, threshold)

Crops the bottom area where the _1_ pattern normally appears

Uses EasyOCR to read text from the cleaned ROI

Finds the line that contains _1_

Selects the best match based on confidence

Displays everything using a Streamlit app

🧩 Tech Stack

Python 3

EasyOCR

PyTorch (CPU)

OpenCV

Streamlit

NumPy / PIL

PyTest (for one simple synthetic test)

I kept everything lightweight and offline as required.
