import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_boxes_on_image(image_path, ocr_results, highlight=None, save_path=None):
    # image_path can be bytes or file path
    if isinstance(image_path, (bytes, bytearray)):
        import numpy as np
        arr = np.frombuffer(image_path, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(image_path)
    if img is None:
        raise ValueError('Could not read image for drawing')
    # draw boxes
    for r in ocr_results:
        box = r.get('box')
        if not box:
            continue
        pts = np.array(box, np.int32).reshape((-1,1,2))
        color = (0,255,0)
        thickness = 2
        cv2.polylines(img, [pts], True, color, thickness)
        # put text near box
        text = r.get('text','')
        org = tuple(pts[0][0])
        cv2.putText(img, text[:30], org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
    # highlight one box in red if requested
    if highlight is not None:
        box = highlight.get('box')
        if box:
            pts = np.array(box, np.int32).reshape((-1,1,2))
            cv2.polylines(img, [pts], True, (0,0,255), 3)
    if save_path:
        cv2.imwrite(save_path, img)
    return img
