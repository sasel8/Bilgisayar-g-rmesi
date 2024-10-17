from ultralytics import YOLO
import cv2  # OpenCV kütüphanesini kullanacağız

# YOLOv8 modelini yükleyin
model = YOLO('yolov8n.pt')

# Bir görüntü üzerinde tahmin yapın
results = model.predict(source="gör.jpg", show=True)

# Sonuçları görüntüleyin
for result in results:
    # Ekranda görüntüyü göster
    cv2.imshow("Tahmin Sonuçları", result.orig_img)  # orijinal resmi gösterir
    cv2.waitKey(0)  # Tuşa basılana kadar görüntü ekranda kalır
    cv2.destroyAllWindows()  # Pencereyi kapatır
