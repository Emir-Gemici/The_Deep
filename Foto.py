"""Tek gorselde tespit. Kullanim: python Foto.py <gorsel.jpg>"""
import os
import sys

from ultralytics import YOLO

gorsel = sys.argv[1] if len(sys.argv) > 1 else "test2.jpeg"
if not os.path.exists(gorsel):
    raise SystemExit(f"Gorsel bulunamadi: {gorsel}\nKullanim: python Foto.py <gorsel.jpg>")

# Nihai 5-sinifli model (forklift, insan, palet, baret, yelek)
model = YOLO("son_model.pt")

# imgsz=960 -> kucuk nesneler (baret/yelek/palet) icin daha iyi tespit
results = model.predict(gorsel, imgsz=960, conf=0.25, save=True, show=True)
