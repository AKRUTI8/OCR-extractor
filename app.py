
import streamlit as st
from src.ocr_engine import OCREngine
from src.text_extraction import find_target_lines, pick_best_match
from src.utils import draw_boxes_on_image
from src.preprocessing import preprocess_image
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title='OCR - Assessment', layout='centered')

st.title('OCR Text Extraction - EasyOCR (Assessment)')
st.markdown('Upload a shipping label / waybill image. The app will preprocess, run EasyOCR, and try to extract the target line containing the pattern `_1_`.')

uploaded = st.file_uploader('Upload image', type=['jpg','jpeg','png','tiff'])
engine = OCREngine(lang_list=['en'])

if uploaded is not None:
    # read bytes for both display and processing
    image_bytes = uploaded.read()
    st.image(image_bytes, caption='Uploaded image', use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Run OCR'):
            try:
                with st.spinner('Preprocessing and running OCR (this may take a few seconds)...'):
                    ocr_results = engine.read_image_bytes(image_bytes)
                st.success(f"Detected {len(ocr_results)} text boxes.")
                # show raw results table
                import pandas as pd
                df = pd.DataFrame([{'text': r['text'], 'confidence': r['confidence']} for r in ocr_results])
                st.dataframe(df)
                # find target lines
                matches = find_target_lines(ocr_results, pattern='_1_')
                best = pick_best_match(matches)
                if best:
                    st.markdown('### ✅ Best match found')
                    st.write('**Text**:', best.get('text'))
                    st.write('**Confidence**: ', best.get('confidence'))
                else:
                    st.markdown('### ⚠️ No direct match found (try different images or check preprocessing)')
                    st.write('You can inspect the raw OCR table above.')
                # show image with boxes and highlight best match
                img_with_boxes = draw_boxes_on_image(image_bytes, ocr_results, highlight=best)
                st.image(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB), caption='OCR boxes (highlighted match in red)')
            except Exception as e:
                st.error(f'Error during OCR: {e}')
    with col2:
        st.markdown('### Preprocessing preview')
        try:
            bw = preprocess_image(image_bytes)
            st.image(bw, caption='Preprocessed (binary) image', use_container_width=True)
        except Exception as e:
            st.warning('Could not preprocess image: ' + str(e))
