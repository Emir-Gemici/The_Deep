from ultralytics import YOLO

# Nihai 5-sinifli model (forklift, insan, palet, baret, yelek)
model = YOLO("son_model.pt")

# imgsz=960 -> kucuk nesneler (baret/yelek/palet) icin daha iyi tespit
results = model.predict("test2.jpeg", imgsz=960, conf=0.25, save=True, show=True)
