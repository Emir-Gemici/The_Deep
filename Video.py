from ultralytics import YOLO

# Nihai 5-sinifli model
model = YOLO("son_model.pt")

# imgsz=960 + GPU (device=0); show=canli izle, save=anotasyonlu video kaydet
sonuc = model.predict(source="test (2).mp4", imgsz=960, conf=0.25, device=0, save=True, show=True)
