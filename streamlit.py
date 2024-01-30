import streamlit as st
from PIL import Image
import detect
import cv2
import preproccess
import numpy as np
import fitz

def main():
    st.set_page_config(layout="wide")

    with st.container():
        st.title("Epoch Technology OCR")
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            uploaded_file = st.file_uploader("Upload a file", type=["pdf", "jpg", "jpeg", "png"])
            if uploaded_file is not None:
                # Görüntüyü yükle
                file_extension = get_file_extension(uploaded_file.name)
                if file_extension  == "pdf":
                    pdf_to_images = pdf_to_image(uploaded_file)
                else:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded File", use_column_width=True)

        with col2:
            if uploaded_file is not None:
                is_paragraph = st.toggle('Paragraph')
                x_threshold = st.slider('Width threshold (Maximum horizontal distance to merge boxes)', value=1.2, min_value=0.1, max_value=20.0, step=0.1)
                y_threshold = st.slider('Height threshold (Maximum different in box height. Boxes with very different text size should not be merged)', value= 0.5, min_value=0.1, max_value=20.0, step=0.1)

                if st.button(key="read",label="Read Document"):
                    if file_extension  == "pdf":
                        extracted_text = extract_text_on_pdf(pdf_to_images, is_paragraph, x_threshold, y_threshold)
                        st.write(extracted_text)
                    else:
                        extracted_text = extract_text_on_image(image, is_paragraph, x_threshold, y_threshold)
                        st.write(extracted_text)

def extract_text_on_image(image,  is_paragraph, x_thresh, y_thresh):

    img_array = np.array(image)
    preproccessed_image = preproccess.preprocess_image(img_array)
    result = detect.detect_OCR(preproccessed_image, is_paragraph, x_thresh, y_thresh)

    return result
def extract_text_on_pdf(pdf_to_images,is_paragraph, x_thresh, y_thresh):
    results = []
    for image in pdf_to_images:
        result = detect.detect_OCR(image, is_paragraph, x_thresh, y_thresh)
        results.append(result)
    
    return results
def pdf_to_image(pdf_file):
    pdf_document = fitz.open("pdf", pdf_file.read())
    images = []
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        pixmap = page.get_pixmap(dpi=400)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        img_array = np.array(image)
        img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        image = preproccess.preprocess_image(img_array_bgr)
        images.append(image)
    pdf_document.close()

    for image in images:
        pil_image = Image.fromarray(image)
        st.image(pil_image, caption="Uploaded File", use_column_width=True)

    return images


def get_file_extension(file_name):
    # Dosya adındaki en son kısmı alarak dosya uzantısını bulma
    parts = file_name.split(".")
    if len(parts) > 1:
        return parts[-1].lower()
    else:
        return None
    
if __name__ == "__main__":
    main()