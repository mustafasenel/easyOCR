import easyocr
import os

current_directory = os.getcwd()
def detect_OCR(image, is_paragraph=False, x_thresh=1.2, y_thresh=0.5):
    reader = easyocr.Reader(['tr'], gpu=True, model_storage_directory=f"{current_directory}/model", user_network_directory=f"{current_directory}/user_network", download_enabled=False)
    result = reader.readtext(image, min_size=50, detail=0, paragraph=is_paragraph, width_ths=x_thresh, height_ths=y_thresh, blocklist="|[]{}")
    return result