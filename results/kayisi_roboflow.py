from roboflow import Roboflow

# Roboflow API'ye bağlan
rf = Roboflow(api_key="1OiSLmmRhdtwMHHPl5Ta")
project = rf.workspace("bilgisayar").project("kayisi-projesi")
version = project.version(1)
model = version.model

# Yerel bir görüntü üzerinde çıkarım yapın
print(model.predict("C:/Users/asele/Downloads/kayisi projesi.v1i.yolov8/images/g1.jpg", confidence=40, overlap=30).json())

# Çıkarımı görselleştirin ve sonucu 'prediction.jpg' olarak kaydedin
# İlk tahmini çalıştıran kısım
print(model.predict("C:/Users/asele/Downloads/kayisi projesi.v1i.yolov8/images/g1.jpg", confidence=40, overlap=30).json())

# Tahmini görselleştirme kısmı
model.predict("C:/Users/asele/Downloads/kayisi projesi.v1i.yolov8/images/g1.jpg", confidence=40, overlap=30).save("prediction.jpg")

# Başka bir URL'de barındırılan bir görüntü üzerinde çıkarım yapın
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
