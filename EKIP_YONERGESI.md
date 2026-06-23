# Ekip Yönergesi — The Deep (TEKNOFEST 3. Senaryo)

> **Vision/Detection katmanı tamamlandı ve devre hazır.** Bu dosya: ekip arkadaşlarının
> sıradaki işini + Vision katmanının **arayüz sözleşmesini** tanımlar.
> Teknik detay: `VISION_KATMANI.md`.

---

## Vision katmanı ne üretiyor (deliverable'lar)

| Dosya | Ne |
|---|---|
| `son_model.pt` | 5 sınıf YOLO26s (forklift, insan, palet, baret, yelek), val mAP50 **0.88** |
| `tracking.py` | kalıcı ID takibi (BoT-SORT+ReID) → `history` (her ID'nin zaman serisi) |
| `olaylar.py` → `olaylar_cikti.json` | **zaman damgalı olay adayları** (VLM girdisi) |
| `demo.py` → `demo_Test.mp4` | görsel demo (kutu+ID+olay banner'ı) |
| `custom_tracker.yaml` | tracker ayarı |

---

## ARAYÜZ SÖZLEŞMESİ — herkes buna göre çalışsın

Vision katmanının çıktısı (`olaylar_cikti.json`):

```json
{
  "video": "...", "track_sayisi": 108,
  "olay_adaylari": [
    {"t":"00:39","saniye":39.0,"tip":"toplanma_adayi","objeler":["insan#51","insan#76","insan#84"],"detay":"3 kisi yakin"},
    {"t":"00:41","saniye":41.7,"tip":"hareketsizlik_adayi","obje":"insan#94","detay":"~2.5sn hareketsiz"}
  ]
}
```

- `tip` ∈ `{toplanma_adayi, hareketsizlik_adayi}` (güvenilir, geometrik sinyaller)
- İhtiyaç olursa **kare-bazlı tespit/track** de erişilebilir: `from tracking import track_video`.

---

## Sıradaki iş — roller

### 🧠 Bera / Selim — VLM / Reasoning
**Alacağınız girdi:** `olaylar_cikti.json` (nereye bakılacağını söyleyen adaylar) + **video kareleri**
(Qwen2.5-VL multimodal, kendisi görüyor) + gerekiyorsa kare-bazlı tespitler.
**Yapacağınız:**
- Olay adaylarını + kareleri **anlamlandırın**: olay türü, risk seviyesi, Türkçe özet, aksiyon önerisi.
- **⚠️ Devrilme tanıma SİZE devredildi.** Vision bunu bbox ile güvenilir yapamadı (forklift'in
  dönüşünü devrilmeyle karıştırıyor — bkz. VISION_KATMANI.md). Siz kareleri gördüğünüz için
  "forklift devrildi"yi **görsel olarak** belirleyin.
- Şartnamenin nihai JSON'unu (summary / events / risk / actions) siz üretin.

### 🔗 Selin — Pipeline / Orkestrasyon
- Akış: `video → Vision (olaylar.py + tracking) → VLM → nihai JSON`.
- Vision çıktısını (olaylar_cikti.json + kareler) VLM'e besleyin.
- Hata yönetimi: tespit/track boş, VLM zaman aşımı, bozuk video vb.

### 📄 Doküman / Demo / Ölçümleme
- `VISION_KATMANI.md`'yi proje dokümanına entegre edin.
- `demo_Test.mp4` + uçtan uca pipeline demosu.
- Ölçümleme: per-sınıf mAP (VISION_KATMANI.md'de) + pipeline inference süresi.

---

## ⚠️ Bilinen kısıt — herkes bilsin

**Gerçek CCTV (grainy, geniş açı) + DEVRİLMİŞ forklift pozu → tespit zayıf.** Devrilmiş forklift
için hazır veri seti yok (test edildi). Demo'yu **net/temsili footage'da** gösterin; gerçek-CCTV
+ devrilmiş-forklift = **"gelecek iş"** (domain-spesifik kaza-footage'ı çıkarıp manuel etiketleme).
Bu kısıtı dürüstçe dokümante etmek jüride artı.
