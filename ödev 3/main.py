from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO
import os

image_urls = [
    "https://i.pinimg.com/736x/0f/65/80/0f658044aa0a995db9cc2f356600f8ed.jpg",
    "https://i.pinimg.com/control/1200x/b4/cc/b8/b4ccb8248e8d513a692d9bb7ca07ad9d.jpg",
    "https://i.pinimg.com/1200x/80/78/06/807806a8cd6e0d4bc10388d95e815993.jpg",
    "https://i.pinimg.com/1200x/e0/99/48/e09948a0896a55320e5c0cd6b347e2f2.jpg",
    "https://i.pinimg.com/control/1200x/0f/65/80/0f658044aa0a995db9cc2f356600f8ed.jpg",
    "https://i.pinimg.com/1200x/68/12/16/681216439ece0c5cb663229c0de76a9f.jpg",
    "https://i.pinimg.com/control/1200x/17/fe/c2/17fec24d9af8649b7f46379b4c5d6a25.jpg",
    "https://i.pinimg.com/1200x/46/7f/4e/467f4eaf31670428b1bc0dccb1711754.jpg",
    "https://i.pinimg.com/1200x/2c/e4/25/2ce4253ed3b2d74365ecc8e2b61cbe38.jpg",
    "https://i.pinimg.com/1200x/e5/e8/4d/e5e84d8dda8d8f4c451bb3aa3442ed71.jpg"
]

model = YOLO("yolov8n.pt")

for idx, url in enumerate(image_urls):
    try:

        response = requests.get(url)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))

        img_path = f'image_{idx}.jpg'
        img.save(img_path)

        results = model.predict(source=img_path, show=False)

        detected_humans = [det for det in results[0].boxes if det.cls == 0]

        if detected_humans:
            print(f'Görüntü {img_path}: İnsan tespit edildi.')
        else:
            print(f'Görüntü {img_path}: İnsan tespit edilmedi.')

        output_dir = 'runs/detect/predict'
        os.makedirs(output_dir, exist_ok=True)
        results[0].save(output_dir)

    except Exception as e:
        print(f"Görüntü {url} ile ilgili hata: {e}")