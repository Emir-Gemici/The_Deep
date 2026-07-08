"""Videoda tespit. Kullanim: python Video.py <video.mp4>"""
import os
import sys

from ultralytics import YOLO

video = sys.argv[1] if len(sys.argv) > 1 else "test (2).mp4"
if not os.path.exists(video):
    raise SystemExit(f"Video bulunamadi: {video}\nKullanim: python Video.py <video.mp4>")

# Nihai 5-sinifli model
model = YOLO("son_model.pt")

# imgsz=960 + GPU (device=0); show=canli izle, save=anotasyonlu video kaydet
sonuc = model.predict(source=video, imgsz=960, conf=0.25, device=0, save=True, show=True)
