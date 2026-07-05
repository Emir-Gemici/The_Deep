"""
FINAL egitim — gercek-dunya performansini duzeltmek icin.

Neden: yolo26n 25-epoch diagnostic'i gosterdi ki VERI temiz (insan duzeldi)
ama nano model + az epoch gercek sahada zayif. test1.jpg'de net insan/yelek/
forklift kacti, sadece baret bulundu = domain gap + model kapasitesi.

Bu egitim: daha buyuk model (yolo26s) + tam epoch (100, patience=20) ile
gercek-dunya genellemesini artirir. imgsz=640 cunku kacan nesneler buyuktu
(cozunurluk degil kapasite sorunu); 640 ayni zamanda ~2x hizli.

CALISTIRMADAN ONCE:
  - GPU OVERCLOCK KAPALI (OC -> cuDNN execution hatasi).
  - torch GPU build olmali: torch.cuda.is_available()==True (cu128).
  - workers=4, cache='ram' ZORUNLU (workers=8 Windows'ta RAM'i doldurur).

Calistir:  .venv\\Scripts\\python.exe yolo26s_final.py
Sonuc:     runs/detect/final_yolo26s/weights/best.pt
"""
import os
from ultralytics import YOLO

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # relative yollar her zaman dogru
    model = YOLO("yolo26s.pt")          # COCO-pretrained backbone -> 5 sinifa fine-tune
    model.train(
        data="birlesik/data.yaml",
        epochs=100,
        patience=20,
        imgsz=640,
        batch=16,
        cache="disk",   # buyuk dataset (18k) icin: RAM'i sismez, Gen5 SSD'de hizli
        workers=4,
        name="final_yolo26s",
    )
