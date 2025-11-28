import easyocr
import numpy as np
from .preprocessing import preprocess_image
from PIL import Image
import cv2

class OCREngine:
    def __init__(self, lang_list=['en']):
        # lazy load reader to avoid long startup if not needed
        self.lang_list = lang_list
        self.reader = None

    def _ensure_reader(self):
        if self.reader is None:
            # CPU-only; EasyOCR will use the local torch installation
            self.reader = easyocr.Reader(self.lang_list, gpu=False)

    def read_image(self, path_or_bytes):
        # preprocess image and run easyocr
        img_bw = preprocess_image(path_or_bytes)
        # easyocr expects color images for some parts — convert binary to 3-channel
        img_color = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2BGR)
        self._ensure_reader()
        results = self.reader.readtext(img_color, detail=1)  # returns (box, text, confidence)
        # normalize results to list of dicts
        out = []
        for box, text, conf in results:
            out.append({'box': np.array(box).tolist(), 'text': text, 'confidence': float(conf)})
        return out

    def read_image_bytes(self, image_bytes):
        return self.read_image(image_bytes)
