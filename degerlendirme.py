"""
degerlendirme.py — BENCHMARK: nihai modelin (son_model.pt) olcumlemesi.
  1) birlesik val uzerinde SINIF BAZLI mAP/recall yazdirir (veri seti mevcutsa).
  2) GERCEK gorsellerde tahmin yapar, anotasyonlu kaydeder, sinif sayimi verir.
  3) HIZ OLCUMU: kare basina cikarim suresi (on-isleme / cikarim / son-isleme, ms).
Kullanim: python degerlendirme.py   (README'deki ~9 ms/kare degeri buradan dogrulanir)
"""
import os
import collections
from ultralytics import YOLO

os.chdir(os.path.dirname(os.path.abspath(__file__)))

WEIGHTS = "son_model.pt"
IMGSZ = 960                      # fine-tune ile dogrulanmis cikarim ayari
TEST_IMAGES = ["test1.jpg", "YOLO_Test_0.jpeg", "test2.jpeg", "test3.jpg"]

print("=" * 64)
print("BENCHMARK — MODEL:", WEIGHTS, "| imgsz:", IMGSZ)
print("=" * 64)
if not os.path.exists(WEIGHTS):
    raise SystemExit(f"HATA: agirlik bulunamadi: {WEIGHTS}")

model = YOLO(WEIGHTS)

# 1) SINIF BAZLI VALIDATION (birlesik/ veri seti indirilip kurulduysa)
print("\n##### 1) VALIDATION (sinif bazli mAP/recall) #####", flush=True)
if os.path.exists("birlesik/data.yaml"):
    try:
        model.val(data="birlesik/data.yaml", imgsz=IMGSZ, verbose=True, plots=False)
    except Exception as e:
        print("  [UYARI] val sirasinda hata:", repr(e))
else:
    print("  birlesik/ veri seti yok -> atlandi. (Veri linkleri README 'Veri Setleri' bolumunde.)")

# 2) GERCEK GORSEL TESTI + 3) HIZ OLCUMU
print(f"\n##### 2) GERCEK GORSEL TESTI + HIZ (imgsz={IMGSZ}) #####", flush=True)
hizlar = []
for img in TEST_IMAGES:
    if not os.path.exists(img):
        continue
    for conf in (0.25, 0.10):
        try:
            r = model.predict(img, imgsz=IMGSZ, conf=conf, save=True,
                              name=f"benchmark_c{int(conf * 100)}",
                              exist_ok=True, verbose=False)[0]
            cc = collections.Counter(model.names[int(c)] for c in r.boxes.cls.tolist())
            txt = ", ".join(f"{k}={v}" for k, v in cc.items()) if cc else "TESPIT YOK"
            hizlar.append(r.speed["inference"])
            print(f"  [{img} @conf{conf}] {txt}"
                  f"   | cikarim {r.speed['inference']:.1f} ms"
                  f" (on {r.speed['preprocess']:.1f} + son {r.speed['postprocess']:.1f})"
                  f"   -> {r.save_dir}", flush=True)
        except Exception as e:
            print(f"  [{img} @conf{conf}] HATA: {e!r}")

if hizlar:
    # ilk cagri model isinmasi icerir; varsa isinma haric ortalama da ver
    ort = sum(hizlar) / len(hizlar)
    ort_sicak = sum(hizlar[1:]) / len(hizlar[1:]) if len(hizlar) > 1 else ort
    print(f"\n  HIZ OZETI: ortalama cikarim {ort:.1f} ms/kare"
          f" | isinma haric {ort_sicak:.1f} ms/kare ({len(hizlar)} olcum)")

print("\n>>> Benchmark bitti.")
