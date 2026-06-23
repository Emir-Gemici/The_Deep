# Vision / Detection Katmanı — Dokümantasyon

> TEKNOFEST 3. Senaryo — "The Deep". Bu katman **video → nesne tespiti → takip →
> olay-türetme → zaman damgalı yapılandırılmış çıktı** üretir ve VLM/Reasoning
> katmanına (Bera/Selim) **girdi** sağlar.
>
> **Sınır:** Bu katman SADECE *sinyal* üretir. Olayın anlamlandırılması (kaza mı,
> ne kadar riskli, Türkçe açıklama, aksiyon önerisi) **multimodal VLM'in işidir**
> (kareleri o görüyor).

---

## 1. Tespit Modeli (5 sınıf)

- **Mimari:** YOLO26s (ultralytics 8.4), COCO-pretrained → 5 sınıfa fine-tune.
- **Sınıflar:** `forklift`, `insan`, `palet`, `baret`, `yelek`.
- **Model:** `runs/detect/final_yolo26s-5/weights/best.pt` (kopya: `son_model.pt`).
- **Çıkarım:** `imgsz=960`, `conf=0.25`.

**Performans (doğrulama, mAP50):** genel **0.877** — insan 0.927 · forklift 0.982 ·
yelek 0.881 · baret 0.836 · palet 0.759.

**Veri (`birlesik/`):** train 18.004 / valid 3.516 / test 2.267. Kritik düzeltmeler:
insan etiket çakışması (otomatik etiketleme, `autolabel.py`) ve palet/yelek veri
takviyesi (`integrate_sets.py`).

---

## 2. Takip — `tracking.py`

- `model.track(persist=True)` + **BoT-SORT + ReID** + uzun buffer (`custom_tracker.yaml`).
- Her objeye **kalıcı ID** + zaman serisi: `history[id] = [{frame,t,cls,cx,cy,w,h}, ...]`.
- Kısa (gürültü) track'ler elenir (`MIN_TRACK=30`).

```python
from tracking import track_video
history, fps = track_video("video.mp4")
```

---

## 3. Olay-Türetme — `olaylar.py`

**Aktif (güvenilir, geometrik) sinyaller:**

| Sinyal | Mantık |
|---|---|
| `hareketsizlik_adayi` | insan track'in merkezi ~2.5 sn bbox boyutuna göre ~sabit |
| `toplanma_adayi` | aynı anda ≥3 insan track'in merkezleri yakın |

Her sinyalde **zaman damgası** = `frame / fps`.

**Devre dışı (denendi → güvenilmez → VLM'e devredildi):**
- **`devrilme` (bbox w/h):** forklift'in *dönüşünü* (önden→yandan görünüm) gerçek
  devrilmeyle karıştırıyor → yanlış pozitif; gerçek devrilmeyi de w/h zaten geniş
  olduğu için ıskalıyor. **Devrilme tanıma multimodal VLM'e devredildi.**
- **`forklift_insan_yakinlık`:** forklift sahnesinde insan-forklift yakınlığı *norm*
  olduğu için çıktıyı boğuyor (anomali değil). Tehlikeli-yakınlık kararı VLM'e bırakıldı.

> Fonksiyonlar (`devrilme`, `yakinlik`) referans/şeffaflık için kodda durur ama çağrılmaz.

---

## 4. Çıktı Arayüzü (VLM'e girdi) — **ARAYÜZ SÖZLEŞMESİ**

`python olaylar.py <video>` → **`olaylar_cikti.json`**:

```json
{
  "video": "Test.mp4",
  "track_sayisi": 108,
  "olay_adaylari": [
    { "t": "00:39", "saniye": 39.0, "tip": "toplanma_adayi",
      "objeler": ["insan#51","insan#76","insan#84"], "detay": "3 kisi yakin" },
    { "t": "00:41", "saniye": 41.7, "tip": "hareketsizlik_adayi",
      "obje": "insan#94", "detay": "~2.5sn hareketsiz" }
  ]
}
```

- `tip` ∈ `{toplanma_adayi, hareketsizlik_adayi}`
- `t` insan-okunur, `saniye` sayısal. VLM bu listeyi + kareleri kullanıp anlamlandırır.

---

## 5. Bilinen Kısıtlar ve Gelecek İş

- **Gerçek CCTV + devrilmiş forklift:** Model dik forklift + net görsellerle eğitildi.
  Gerçek gözetim kamerası (grainy, geniş açı) ve **yan yatmış forklift** pozunda tespit
  zayıf; devrilmiş forklift için hazır veri **yok** (test edilip doğrulandı).
  - **Gelecek iş:** gerçek CCTV kaza footage'ından kare çıkarıp **manuel etiketleme**
    (özellikle devrilmiş forklift) → retrain. Domain-spesifik bir veri-toplama projesi.
- **Olay-türetme sınırı:** Basit 2D geometrik sinyaller yalnızca *konum/hareket* yakalar
  (toplanma, hareketsizlik). Görsel-anlamsal olaylar (devrilme, tehlikeli yakınlık)
  multimodal VLM'e bırakıldı. Eşikler tek videoda ayarlandı → gerçek footage'da tunelanmalı.
- **3D animasyon (CGI):** tutarsız (eğitim gerçek-fotoğraf domaini).
- **En iyi sonuç:** eğitim domaine benzer **gerçek/net görsellerde** (mAP 0.88).

---

## 6. Çalıştırma

```powershell
# Tespit
.\.venv\Scripts\python.exe Foto.py            # tek görsel
.\.venv\Scripts\python.exe Webcam.py          # canlı kamera
.\.venv\Scripts\python.exe Video.py           # video

# Takip (kalıcı ID görselleştirme)
.\.venv\Scripts\python.exe tracking.py <video>

# Olay-türetme (VLM girdisi)
.\.venv\Scripts\python.exe olaylar.py <video> # -> olaylar_cikti.json

# Görsel demo (kutu + ID + olay banner)
.\.venv\Scripts\python.exe demo.py <video>    # -> demo_<video>.mp4
```

**Dosyalar:** `son_model.pt` · `custom_tracker.yaml` · `tracking.py` · `olaylar.py` ·
`demo.py` · veri araçları: `yolo_index_remap.py`, `autolabel.py`, `integrate_sets.py`.
Ekip yönergesi: `EKIP_YONERGESI.md`.
