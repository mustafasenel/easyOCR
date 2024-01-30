import cv2
import os
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import easyocr
import fitz
import preproccess 
import detect

app = FastAPI()
current_directory = os.getcwd()

@app.get('/')
async def index():
    return "Welcome EasyOCR By SENEL"

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Dosyayı bellekte geçici bir nesne olarak tut
        file_content = await file.read()
        result = perform_ocr(file_content, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def perform_ocr(file_content, file_name):
    # Dosyayı bellekte geçici bir dosya olarak kullan
    with BytesIO(file_content) as buffer:
        if file_name.lower().endswith('.pdf'):
            # PDF dosyasını işle
            results = apply_ocr_on_pdf(buffer)
        elif file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Resmi doğrudan işle
            results = apply_ocr_on_image(buffer)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
    
    return results

def apply_ocr_on_pdf(buffer):
    pdf_document = fitz.open("pdf", buffer.read())
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

    results = []
    for image in images:
        result = detect.detect_OCR(image)
        results.extend(result)
    
    return results

def apply_ocr_on_image(buffer):
    image = cv2.imdecode(np.frombuffer(buffer.read(), np.uint8), cv2.IMREAD_COLOR)
    image = preproccess.preprocess_image(image)
    result = detect.detect_OCR(image)
    return result

# Rotate the image around its center
