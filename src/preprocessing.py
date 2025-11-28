import cv2
import numpy as np
from PIL import Image

def read_image(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        # read from bytes
        arr = np.frombuffer(path_or_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path_or_bytes)
    return img

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resize_for_ocr(img, target_height=800):
    """Resize while keeping aspect ratio. Target height helps stabilize OCR scale."""
    h, w = img.shape[:2]
    scale = target_height / float(h)
    new_w = int(w * scale)
    resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    return resized

def denoise(img):
    # bilateral filter preserves edges while reducing noise
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def binarize(img_gray):
    # adaptive threshold works well for illumination variations
    th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 15)
    return th

def deskew(img_gray):
    # approximate deskew by computing minAreaRect of edges
    edges = cv2.Canny(img_gray, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if coords.shape[0] < 10:
        return img_gray  # nothing to deskew
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image(path_or_bytes):
    """Full pipeline: read -> resize -> grayscale -> denoise -> deskew -> binarize"""
    img = read_image(path_or_bytes)
    if img is None:
        raise ValueError("Could not read image")
    img = resize_for_ocr(img, target_height=900)
    gray = to_grayscale(img)
    den = denoise(gray)
    desk = deskew(den)
    bw = binarize(desk)
    return bw  # return binary image (numpy array)
