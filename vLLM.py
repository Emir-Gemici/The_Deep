# =============================================================================
# The_Deep — VLM katmanı (Qwen2.5-VL-7B-Instruct, 8-bit, tamamen YEREL)
#
# İKİ KULLANIM ŞEKLİ:
#  1. Colab hücre yaması: aşağıdaki HÜCRE A/B/C blokları
#     vLLM_calismalari.ipynb'deki ilgili hücrelerin yerine yapıştırılabilir.
#  2. Yerel bağımsız script:
#     python vLLM.py <video> [--max-kare N] [--olay-json DOSYA]
#     (varsayılanlar: Test.mp4, 32, olaylar_cikti.json)
#
# NOT (notebook kullanıcıları): vLLM_calismalari.ipynb'de "import cv2 /
# def process_video(...)" içeren ve video_agent.run(...) çağıran BAĞIMSIZ hücre
# SİLİNMELİ — sınıfta run() metodu yok (AttributeError); process_video yeterli.
# Ayrıca __main__ bloğunu hücreye yapıştırmayın (script'e özel).
#
# ÇIKTI / UI ENTEGRASYONU: analiz sonunda "vlm_rapor.json" yazılır;
# arayuz.py (Streamlit) bu dosyayı okur. Şema (İKİSİ DE BUNA UYMALI):
#   { "video": "...",
#     "kare_analizleri": [kare bazlı VLM analizleri = agent.memory],
#     "rapor": final_report() çıktısı (video_ozeti/genel_risk/...),
#     "sartname_json": şartnamedeki mock format (summary/events/risk/actions) }
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# HÜCRE A (LOKAL 8-BİT VERSİYON): Model yükleme — 8-bit'i diske BİR KEZ kaydet
# 16GB VRAM için optimize edilmiştir. (4-bite göre daha az kayıp, yüksek kalite)
# ─────────────────────────────────────────────────────────────────────────────
import os, torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# Modeli kaydedeceğimiz yerel klasör (8-bit için ayrı bir dizin)
Q8_PATH = "./lokal_qwen_8bit"

# GPU yoksa (örn. jüri makinesi) sabit device_map="cuda" yüklemeyi patlatır →
# torch.cuda kontrolüyle "auto"ya düş (accelerate CPU/diske dağıtır) ve uyar.
if torch.cuda.is_available():
    DEVICE_MAP = "cuda"
else:
    DEVICE_MAP = "auto"
    print("⚠️ CUDA bulunamadı → device_map='auto' deneniyor. 8-bit (bitsandbytes) yükleme "
          "GPU'suz makinede BAŞARISIZ olabilir — VLM katmanını GPU'lu makinede çalıştırın.")

if os.path.isdir(Q8_PATH):
    print("Disk üzerindeki 8-bit kopya tespit edildi → Doğrudan yükleniyor (ÇEVRİMDIŞI)...")
    # DEVICE_MAP="cuda" ise model doğrudan 16 GB VRAM'li GPU'nuza alınır
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(Q8_PATH, device_map=DEVICE_MAP)
    processor = AutoProcessor.from_pretrained(Q8_PATH)
else:
    print("Model bulunamadı. İnternetten indiriliyor ve 8-bit olarak diske kaydediliyor (Tek seferlik)...")

    # Sadece 8-bit modunu aktif ediyoruz, diğer karmaşık 4-bit ayarlarına gerek yok
    bnb = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        quantization_config=bnb,
        device_map=DEVICE_MAP,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # Klasörü oluştur ve kaydet
    os.makedirs(Q8_PATH, exist_ok=True)
    model.save_pretrained(Q8_PATH)
    processor.save_pretrained(Q8_PATH)
    print(f"8-bit kopya başarıyla {Q8_PATH} klasörüne kaydedildi.")

print("✅ Model hazır")


# ─────────────────────────────────────────────────────────────────────────────
# HÜCRE B (değişim): TOOL'lar — dil kayması / tekrar / bozuk JSON düzeltmeleri
#
# Değişenler:
#  1. generate: do_sample=False (greedy) + repetition_penalty=1.05 (Qwen'in
#     önerdiği değer) + max_new_tokens=512 (300'de JSON yarıda kesilip parse
#     hatası veriyordu).
#  2. Asistan cevabı "{" ile ÖN-DOLDURULUYOR → model düz metinle başlayamıyor,
#     JSON'a mecbur kalıyor.
#  3. Türkçe zorlaması user mesajının SONUNA da eklendi (son konum > sistem
#     promptu ortası) + prompta doldurulmuş Türkçe ÖRNEK çıktı kondu (few-shot).
#  4. JSON, replace() yerine regex ile ilk {...} bloğu olarak çekiliyor.
#  5. Değerlerde bariz İngilizce yakalanırsa veya parse bozuksa 1 kez daha
#     denenir (toplam 2 deneme).
#  6. risk_skoru int'e zorlanıyor ("7" gelirse max() patlamasın diye).
# ─────────────────────────────────────────────────────────────────────────────
import gc, json, re
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from ultralytics import YOLO

# NOT: Yarışma sınıflarıyla (forklift/insan/palet/baret/yelek) çalışmak için
# kendi modelinizi kullanın. COCO yolov8n "train/truck" gibi alakasız sınıflar
# üretip VLM'i yanıltıyor (rapordaki "Trenin durumu kontrol edin" bundan).
YOLO_WEIGHTS = "son_model.pt" if os.path.exists("son_model.pt") else "yolov8n.pt"
yolo_model = YOLO(YOLO_WEIGHTS)
yolo_model.to("cpu")
print(f"YOLO ağırlıkları: {YOLO_WEIGHTS}")

# ── TOOL 1: YOLO (fine-tune ile doğrulanmış çıkarım ayarları eklendi) ──
def tool_yolo_detect(image: Image.Image) -> dict:
    # imgsz=960 + conf=0.25: son_model.pt fine-tune'unun doğrulanmış ayarları
    # (tracking.py ile aynı — vision katmanıyla tutarlı tespit için)
    results = yolo_model(image, imgsz=960, conf=0.25, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": r.names[int(box.cls)],
                "confidence": round(float(box.conf), 2),
                "bbox": [round(x) for x in box.xyxy[0].tolist()],
            })
    return {"tool": "yolo_detect", "detections": detections}


SYSTEM_PROMPT = """You are a real-time Image Processing and Risk Assessment Agent for industrial facilities.

Your job is to OBJECTIVELY assess the scene shown in the image. Most scenes in a normal facility are routine and safe — do NOT assume danger exists unless you can clearly see evidence of it. Overestimating risk is just as harmful as underestimating it.

STEP 1 — Observe only what is visible:
- How many people are present, and what are they doing
- Are there vehicles or machines, and are they active or idle
- Is there any CLEAR visual evidence of injury or danger: a person lying on the ground, visibly fallen/damaged equipment, someone in the direct path of a moving vehicle, a person missing required PPE (helmet/vest) in a hazardous zone
- Do NOT infer danger from context, camera angle, or the YOLO detection list alone.

STEP 2 — Score honestly:
- risk_skoru 1-3: Normal, routine activity. This is the DEFAULT unless evidence says otherwise.
- risk_skoru 4-6: moderate/ambiguous hazard, NOT a confirmed accident. Action plan must be PRECAUTIONARY only. Do NOT mention injury, first aid, ambulance, or moving a person.
- risk_skoru 7-10: CLEAR, unambiguous evidence of injury, accident, or imminent life-threatening danger.

STEP 3 — Action plan:
- risk 1-3: routine monitoring only. Do NOT invent emergency steps.
- risk 4-10: at least 4 distinct, prioritized steps based on what you actually observed in THIS image. Each item must be an ACTION (imperative), never a status statement. No repeated phrasing.

OUTPUT CONTRACT (MOST IMPORTANT RULE):
Respond with ONLY one JSON object, no extra text, no markdown fences.
ALL string VALUES must be written in TURKISH. Never use English words in values.
Keys must be exactly: olay_turu, olay_yorumu, risk_skoru, acil_mudahale, aksiyon_plani.

Example of a valid response (illustrative values):
{
  "olay_turu": "Rutin depo çalışması",
  "olay_yorumu": "Görüntüde iki işçi paletlerin yanında ayakta duruyor, forklift hareketsiz. Yaralanma veya tehlike belirtisi gözlenmedi.",
  "risk_skoru": 2,
  "acil_mudahale": false,
  "aksiyon_plani": ["Standart izlemeye devam edin"]
}"""

TR_TAIL = ("\nSadece tek bir JSON nesnesi döndür. Tüm alan DEĞERLERİ Türkçe olacak; "
           "İngilizce kelime kullanma. Anahtarları ve şemayı aynen koru.")

_JSON_RE = re.compile(r"\{.*\}", re.S)
_EN_STOPWORDS = re.compile(r"\b(the|is|are|and|of|to|with|worker|ensure|monitor|check)\b", re.I)


def _generate_json(messages, prefill="{"):
    """Chat şablonunu uygular, asistanı '{' ile başlatır, greedy üretir."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += prefill  # ön-doldurma: model JSON'un içinden devam etmek zorunda
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)  # model neredeyse oraya (CUDA yoksa CPU'da da çalışsın)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,            # determinizm → dil kayması ve saçmalama azalır
            repetition_penalty=1.05,    # tekrarlayan cümleleri keser
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return prefill + processor.batch_decode(trimmed, skip_special_tokens=True)[0]


# ── TOOL 2: VLM Analyzer (yenilendi) ──
def tool_vlm_analyze(image: Image.Image, yolo_result: dict, timestamp: str) -> dict:
    # En-boy oranını KORUYARAK küçült: eski resize((640, 480)) 16:9 CCTV karesini
    # 4:3'e DEFORME ediyordu. thumbnail uzun kenarı ~896 px'e indirir → Qwen'in
    # görüntü token maliyeti dengede kalır, insan/baret gibi detaylar ezilmez.
    image = image.copy()  # çağıranın (process_video) karesini bozmamak için
    image.thumbnail((896, 896), Image.Resampling.LANCZOS)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    detection_text = json.dumps(yolo_result["detections"], ensure_ascii=False)

    def build(extra=""):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",
                 "text": f"Zaman: {timestamp}\nYOLO Tespitleri: {detection_text}\nAnaliz et." + TR_TAIL + extra},
            ]},
        ]

    output = ""
    for attempt in range(2):
        extra = "" if attempt == 0 else "\nÖnceki cevap geçersizdi. SADECE şemadaki JSON'u, tamamen Türkçe üret."
        output = _generate_json(build(extra))

        m = _JSON_RE.search(output)
        if not m:
            continue
        try:
            result = json.loads(m.group(0))
        except Exception:
            continue

        # risk_skoru güvenli int
        try:
            result["risk_skoru"] = int(float(result.get("risk_skoru", 0)))
        except Exception:
            result["risk_skoru"] = 0

        # dil bekçisi: değerlerde bariz İngilizce varsa bir kez daha dene
        blob = " ".join(str(v) for v in result.values())
        if attempt == 0 and _EN_STOPWORDS.search(blob):
            continue

        result["tool"] = "vlm_analyze"
        result["timestamp"] = timestamp
        return result

    return {"tool": "vlm_analyze", "timestamp": timestamp,
            "ham_cikti": output, "hata": "JSON parse edilemedi"}


# ── TOOL 3: Risk Evaluator (eşik >=7: SYSTEM_PROMPT ile hizalandı) ──
def tool_risk_evaluate(vlm_result: dict) -> dict:
    skor = vlm_result.get("risk_skoru", 0)
    if skor >= 7:  # SYSTEM_PROMPT 'risk 7-10: CLEAR evidence' der → eşik 7 (8 değil)
        seviye, renk = "KRİTİK", "🔴"
    elif skor >= 5:
        seviye, renk = "ORTA", "🟡"
    else:
        seviye, renk = "DÜŞÜK", "🟢"
    return {"tool": "risk_evaluate", "risk_seviyesi": seviye,
            "renk": renk, "acil_alarm": skor >= 7}

print("✅ Tool'lar hazır")


# ─────────────────────────────────────────────────────────────────────────────
# HÜCRE C (değişim): Agent — "12 kare tavanı" yerine OLAY ODAKLI kare seçimi
#
# Eski davranış: aday kareler tek düze aralıkla 12'ye indiriliyordu →
#   kare aralığı = video süresi / 12  (2 dk'lık videoda ~12 sn'de 1 kare).
# Yeni davranış:
#   1. olaylar_cikti.json varsa (vision katmanının arayüz sözleşmesi!) VLM
#      SADECE olay adaylarının zamanlarına bakar (t ve t+1.5 sn).
#   2. Yoksa süreye orantılı örnekleme: 5 sn'de 1 kare, tavan max_kare.
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import difflib


def _grab_frame(cap, t_sn, fps):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sn * fps))
    ret, frame = cap.read()
    if not ret:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _benzer_var(aksiyon, liste, esik=0.8):
    """'Koruyucu ekipmanları kontrol edin' vb. yakın-tekrarları eler."""
    a = aksiyon.lower().strip(" .!")
    for b in liste:
        if difflib.SequenceMatcher(None, a, b.lower().strip(" .!")).ratio() >= esik:
            return True
    return False


def sartname_json(rapor: dict, memory: list) -> dict:
    """Nihai raporu şartnamedeki mock JSON formatına birebir çevirir.

    Şartname 5. bölümdeki örnek anahtarlar: summary / events / risk / actions.
    final_report şeması Türkçe anahtarlı kaldığı için jüri uyumu buradan üretilir.
    """
    olaylar = rapor.get("tespit_edilen_olaylar") or [
        {"zaman": e.get("timestamp"), "olay": e.get("olay_turu"),
         "risk_skoru": e.get("risk_skoru", 0)} for e in memory
    ]
    # Mock format ONEMLI olaylari listeler: rutin kareler ("Depo işçiliği" vb.)
    # olay sayilmaz -> risk >= 4 filtresi. Tam kare dokumu rapor.tespit_edilen_olaylar'da.
    onemli = [o for o in olaylar if o.get("risk_skoru", 0) >= 4]
    return {
        "summary": rapor.get("video_ozeti", ""),
        "events": [{"time": o.get("zaman"), "event": o.get("olay")} for o in onemli],
        "risk": rapor.get("genel_risk", "Düşük"),
        "actions": rapor.get("genel_aksiyon_plani", []),
    }


def rapor_kaydet(video_path: str, memory: list, rapor: dict,
                 dosya: str = "vlm_rapor.json") -> str:
    """Analiz çıktısını vlm_rapor.json'a yazar; mutlak yolu döndürür.

    ŞEMA SÖZLEŞMESİ (vLLM.py yazar, arayuz.py okur — İKİSİ DE BUNA UYMALI):
      video           → analiz edilen dosya adı
      kare_analizleri → kare bazlı VLM analizleri (agent.memory)
      rapor           → final_report() çıktısı
      sartname_json   → şartnamedeki mock format (summary/events/risk/actions)
    """
    veri = {
        "video": os.path.basename(video_path),
        "kare_analizleri": memory,
        "rapor": rapor,
        "sartname_json": sartname_json(rapor, memory),
    }
    with open(dosya, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=2)
    return os.path.abspath(dosya)


class VideoAnalysisAgent:
    def __init__(self):
        self.memory = []

    def process_video(self, video_path: str,
                      olay_json: str = "olaylar_cikti.json",
                      max_kare: int = 32,
                      uniform_saniye: float = 5.0):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sure = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps

        # 1) Hedef zamanları belirle
        hedefler = []
        if os.path.exists(olay_json):
            adaylar = json.load(open(olay_json, encoding="utf-8")).get("olay_adaylari", [])
            for o in adaylar:
                t = float(o["saniye"])
                hedefler.append(round(t, 1))
                hedefler.append(round(min(t + 1.5, max(sure - 0.2, 0)), 1))  # olayın hemen sonrası
            print(f"🎯 {len(adaylar)} olay adayı bulundu (vision katmanı) → olay odaklı analiz")
        if not hedefler:
            n = int(min(max_kare, max(12, sure // uniform_saniye)))
            hedefler = [round(i * sure / n, 1) for i in range(n)]
            print(f"🎯 Olay dosyası yok → {n} kare, ~{sure / n:.1f} sn arayla")

        hedefler = sorted(set(hedefler))[:max_kare]
        print(f"🧠 {len(hedefler)} kare VLM ile analiz ediliyor...")

        # 2) Her hedef kare: YOLO + VLM
        for t in hedefler:
            img = _grab_frame(cap, t, fps)
            if img is None:
                continue
            ts = f"{int(t) // 60:02d}:{int(t) % 60:02d}"
            yolo_result = tool_yolo_detect(img)
            vlm_result = tool_vlm_analyze(img, yolo_result, ts)

            if vlm_result.get("risk_skoru") is not None and "hata" not in vlm_result:
                self.memory.append({
                    "timestamp": ts,
                    "olay_turu": vlm_result.get("olay_turu"),
                    "olay_yorumu": vlm_result.get("olay_yorumu"),
                    "risk_skoru": vlm_result.get("risk_skoru", 0),
                    "acil_mudahale": vlm_result.get("acil_mudahale"),
                    "aksiyon_plani": vlm_result.get("aksiyon_plani"),
                })
            print(f"  [{ts}] → {vlm_result.get('olay_turu')} (risk: {vlm_result.get('risk_skoru')})")

        cap.release()

        # 3) Nihai rapor + kalıcı çıktı (arayuz.py vlm_rapor.json'u okur)
        rapor = self.final_report()
        rapor_yolu = rapor_kaydet(video_path, self.memory, rapor)
        print(f"💾 Rapor kaydedildi: {rapor_yolu}")
        return rapor

    def final_report(self) -> dict:
        if not self.memory:
            return {"video_ozeti": "Herhangi bir olay tespit edilmedi", "olaylar": []}

        max_risk_event = max(self.memory, key=lambda x: x["risk_skoru"])
        kritik_olaylar = [e for e in self.memory if e.get("acil_mudahale")]

        # Aksiyonları SADECE risk >= 4 olaylardan topla + yakın-tekrarları ele
        tum_aksiyonlar = []
        for e in sorted(self.memory, key=lambda x: -x["risk_skoru"]):
            if e["risk_skoru"] < 4:
                continue
            for a in (e.get("aksiyon_plani") or []):
                if isinstance(a, str) and a.strip() and not _benzer_var(a, tum_aksiyonlar):
                    tum_aksiyonlar.append(a.strip())

        return {
            "video_ozeti": (f"Videoda {len(self.memory)} sahne analiz edildi. "
                            f"En riskli olay: '{max_risk_event['olay_turu']}' "
                            f"({max_risk_event['timestamp']})."),
            # eşik >=7: SYSTEM_PROMPT'un 'risk 7-10' tanımıyla tutarlı (eskiden 8'di)
            "genel_risk": ("Yüksek" if max_risk_event["risk_skoru"] >= 7
                           else "Orta" if max_risk_event["risk_skoru"] >= 5 else "Düşük"),
            "kritik_olay_sayisi": len(kritik_olaylar),
            "tespit_edilen_olaylar": [
                {"zaman": e["timestamp"], "olay": e["olay_turu"], "risk_skoru": e["risk_skoru"]}
                for e in self.memory
            ],
            "genel_aksiyon_plani": tum_aksiyonlar[:6],
        }


# Kullanım (olaylar_cikti.json'u da Colab'a yükleyin — Osman'ın vision çıktısı):
# agent = VideoAnalysisAgent()
# report = agent.process_video("demo_Test.mp4")
# print(json.dumps(report, ensure_ascii=False, indent=2))
print("✅ Agent hazır")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="The Deep — VLM video analiz ajanı (Qwen2.5-VL-7B, tamamen yerel)")
    parser.add_argument("video", nargs="?", default="Test.mp4",
                        help="Analiz edilecek video dosyası (varsayılan: Test.mp4)")
    parser.add_argument("--max-kare", type=int, default=32,
                        help="VLM ile analiz edilecek en fazla kare (hızlı test için 5 verin)")
    parser.add_argument("--olay-json", default="olaylar_cikti.json",
                        help="Vision katmanının olay adayı dosyası")
    # parse_known_args: notebook/Colab kernel'inin kendi argv'si (-f kernel.json)
    # argparse'ı SystemExit(2) ile patlatmasın — hücreye yapıştırılınca da çalışır
    args, _ = parser.parse_known_args()

    print("🚀 Test başlatılıyor. Şebnus devrede...")
    agent = VideoAnalysisAgent()

    # --max-kare 5 → sistem sadece 5 kareyi analiz eder (hızlı test için)
    report = agent.process_video(args.video, olay_json=args.olay_json,
                                 max_kare=args.max_kare)

    print("\n" + "=" * 50)
    print("📊 VLM ANALİZ RAPORU:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 50)
    print(f"💾 Rapor dosyası: {os.path.abspath('vlm_rapor.json')}")