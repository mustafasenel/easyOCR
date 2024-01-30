import requests
import time

url = 'https://senel-easyocr.hf.space/uploadfile/'

file = "doc09313820230801104352.pdf"

files = {'file': (file, open(file, 'rb'), 'application/pdf')}
headers = {'accept': 'application/json'}

print(f"{file} dosyası için istek gönderildi.")
start_time = time.time()  

response = requests.post(url, files=files, headers=headers)

end_time = time.time()  
elapsed_time = end_time - start_time 

print(f"Geçen Süre: {elapsed_time} saniye")
print("Status Code:", response.status_code)
print("Response:", response.text)
