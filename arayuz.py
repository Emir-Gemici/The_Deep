"""
arayuz.py — The Deep: Streamlit tabanli CANLI IZLEME ARAYUZU.

Video uzerinde YOLO (son_model.pt) tespitlerini canli cizer; vision katmaninin
olay adaylarini (olaylar_cikti.json) ve VLM ajaninin analizlerini
(vlm_rapor.json) oynatma zamaniyla SENKRON gosterir.

ONEMLI: VLM verisi DOSYADAN okunur — canli VLM cagrisi YOKTUR, dolayisiyla
transformers / torch-VLM bagimliligi gerektirmez. vlm_rapor.json'u vLLM.py
uretir; iki dosya ayni semaya uyar (kare_analizleri / rapor / sartname_json).

Calistirma:
    streamlit run arayuz.py

Girdi dosyalari (ayni dizinde):
    son_model.pt        — YOLO26s fine-tune agirliklari (yoksa kutusuz oynatma)
    olaylar_cikti.json  — python olaylar.py <video> ile uretilir (istege bagli)
    vlm_rapor.json      — python vLLM.py <video>    ile uretilir (istege bagli)
"""
import glob
import json
import os
import time

import cv2
import streamlit as st
from ultralytics import YOLO

os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODEL_YOLU = "son_model.pt"
OLAY_DOSYASI = "olaylar_cikti.json"
VLM_DOSYASI = "vlm_rapor.json"
OLAY_TOLERANS_SN = 3.0  # olay adayi vurgulama penceresi (± sn)

# Sinif basina SABIT kutu rengi (BGR — cv2 cizimi icin)
SINIF_RENKLERI = {
    "forklift": (0, 165, 255),   # turuncu
    "insan": (80, 200, 80),      # yesil
    "palet": (200, 120, 40),     # mavi
    "baret": (0, 220, 220),      # sari
    "yelek": (220, 80, 200),     # mor/pembe
}


# ── Yardimcilar ──────────────────────────────────────────────────────────────

def mmss(saniye):
    """Saniyeyi mm:ss metnine cevirir."""
    saniye = max(0, int(saniye))
    return f"{saniye // 60:02d}:{saniye % 60:02d}"


def zaman_sn(ts):
    """'mm:ss' (veya 'hh:mm:ss') metnini saniyeye cevirir; bozuksa 0."""
    try:
        toplam = 0.0
        for parca in str(ts).split(":"):
            toplam = toplam * 60 + float(parca)
        return toplam
    except (ValueError, TypeError):
        return 0.0


def sinif_rengi(ad):
    """Bilinen siniflara sabit renk; bilinmeyene ad'dan turetilmis sabit renk."""
    if ad in SINIF_RENKLERI:
        return SINIF_RENKLERI[ad]
    h = sum(ord(c) for c in ad)
    return (50 + (h * 37) % 200, 50 + (h * 91) % 200, 50 + (h * 53) % 200)


def json_oku(yol):
    """JSON dosyasini guvenli okur; yoksa/bozuksa None dondurur (COKME YOK)."""
    if not os.path.exists(yol):
        return None
    try:
        with open(yol, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_resource
def model_yukle(yol):
    """YOLO modelini BIR KEZ yukler (cache_resource: rerun'larda tekrar yuklenmez)."""
    return YOLO(yol)


def tespit_et(model, kare_bgr, conf, imgsz):
    """Tek karede YOLO tahmini; arayuz sozlesmesine uygun tespit listesi dondurur."""
    if model is None:
        return []
    r = model.predict(kare_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    tespitler = []
    for box in r.boxes:
        tespitler.append({
            "class": r.names[int(box.cls)],
            "conf": round(float(box.conf), 2),
            "bbox": [int(round(v)) for v in box.xyxy[0].tolist()],
        })
    return tespitler


def kutulari_ciz(kare_bgr, tespitler):
    """Kutu + 'sinif conf' etiketini kare uzerine cizer (sinif basina sabit renk)."""
    for t in tespitler:
        x1, y1, x2, y2 = t["bbox"]
        renk = sinif_rengi(t["class"])
        cv2.rectangle(kare_bgr, (x1, y1), (x2, y2), renk, 2)
        etiket = f"{t['class']} {t['conf']:.2f}"
        (ew, eh), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ey = max(y1, eh + 8)  # etiket kare disina tasmasin
        cv2.rectangle(kare_bgr, (x1, ey - eh - 8), (x1 + ew + 6, ey), renk, -1)
        cv2.putText(kare_bgr, etiket, (x1 + 3, ey - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return kare_bgr


# ── Sayfa ve durum ───────────────────────────────────────────────────────────

st.set_page_config(page_title="The Deep — Canlı İzleme", page_icon="🎥", layout="wide")
st.title("🎥 The Deep — Canlı İzleme Arayüzü")
st.caption("YOLO tespiti + Vision olay adayları + VLM analizi (dosyadan, canlı VLM çağrısı yok)")

if "oynat" not in st.session_state:
    st.session_state.oynat = False
if "kare_no" not in st.session_state:
    st.session_state.kare_no = 0
if "video_adi" not in st.session_state:
    st.session_state.video_adi = None


def _baslat():
    st.session_state.oynat = True


def _duraklat():
    st.session_state.oynat = False


def _sifirla():
    st.session_state.oynat = False
    st.session_state.kare_no = 0


# ── Kenar cubugu ─────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Ayarlar")

videolar = sorted(glob.glob("*.mp4"))
if not videolar:
    st.error("Çalışma dizininde .mp4 dosyası bulunamadı. Lütfen bir video ekleyin.")
    st.stop()

# Varsayilan secim: paneller dolu acilsin diye once VLM raporundaki video,
# sonra olay dosyasindaki, sonra Test.mp4; hicbiri yoksa listedeki ilk video.
_vlm_v = (json_oku(VLM_DOSYASI) or {}).get("video")
_olay_v = (json_oku(OLAY_DOSYASI) or {}).get("video")
varsayilan = 0
for _aday in (_vlm_v, _olay_v, "Test.mp4"):
    if _aday in videolar:
        varsayilan = videolar.index(_aday)
        break
video_yolu = st.sidebar.selectbox("Video", videolar, index=varsayilan)

conf = st.sidebar.slider("Güven eşiği (conf)", 0.05, 0.90, 0.25, 0.05)
imgsz = st.sidebar.selectbox("Görüntü boyutu (imgsz)", [640, 960], index=1)
kare_atlama = st.sidebar.slider("Kare atlama (performans)", 1, 10, 2,
                                help="Her N. kare işlenir; büyük değer = hızlı ama seyrek tespit.")

b1, b2, b3 = st.sidebar.columns(3)
b1.button("▶ Başlat", on_click=_baslat, width="stretch")
b2.button("⏸ Duraklat", on_click=_duraklat, width="stretch")
b3.button("⟲ Sıfırla", on_click=_sifirla, width="stretch")

# Video degisince konumu sifirla
if st.session_state.video_adi != video_yolu:
    st.session_state.video_adi = video_yolu
    st.session_state.kare_no = 0
    st.session_state.oynat = False

# ── Model ve veri dosyalari (dayanikli yukleme) ──────────────────────────────

model = None
if os.path.exists(MODEL_YOLU):
    model = model_yukle(MODEL_YOLU)
else:
    st.error(f"Model dosyası bulunamadı: `{MODEL_YOLU}` — video kutusuz oynatılacak. "
             "Eğitilmiş ağırlık dosyasını çalışma dizinine kopyalayın.")

olay_verisi = json_oku(OLAY_DOSYASI)
vlm_verisi = json_oku(VLM_DOSYASI)

# Rapor SECILI videoya mi ait? (baska videonun analizi yanlislikla gosterilmesin)
vlm_video = (vlm_verisi or {}).get("video")
vlm_eslesme = vlm_verisi is not None and vlm_video == os.path.basename(video_yolu)

# ── Yerlesim: sol video, sag paneller ────────────────────────────────────────

kol_sol, kol_sag = st.columns([2, 1])

with kol_sol:
    zaman_ph = st.empty()
    ilerleme_ph = st.empty()
    video_ph = st.empty()

with kol_sag:
    st.markdown("##### 📡 Canlı YOLO JSON")
    yolo_ph = st.empty()
    st.markdown("##### 👁 Olay Adayları (Vision)")
    olay_ph = st.empty()
    st.markdown("##### 🧠 VLM Analizi")
    vlm_ph = st.empty()

# ── Panel guncelleme fonksiyonlari ───────────────────────────────────────────

def yolo_panel(t_sn, tespitler):
    """Son karenin tespit listesi (class/conf/bbox)."""
    if model is None:
        yolo_ph.info("Model yok — tespit yapılamıyor.")
        return
    yolo_ph.json({"t": mmss(t_sn), "tespit_sayisi": len(tespitler), "tespitler": tespitler})


def olay_panel(t_sn):
    """olaylar_cikti.json adaylari; oynatma zamani ±3sn icindekiler vurgulanir."""
    with olay_ph.container():
        if olay_verisi is None:
            st.info("`olaylar_cikti.json` bulunamadı — `python olaylar.py <video>` ile üretin.")
            return
        if olay_verisi.get("video") != os.path.basename(video_yolu):
            st.info(f"Bu video için olay dosyası yok (mevcut: `{olay_verisi.get('video')}`) — "
                    f"`python olaylar.py {video_yolu}` ile üretin.")
            return
        adaylar = olay_verisi.get("olay_adaylari", [])
        if not adaylar:
            st.info("Olay adayı bulunmadı.")
            return
        for aday in adaylar:
            sn = float(aday.get("saniye", zaman_sn(aday.get("t"))))
            objeler = aday.get("objeler") or ([aday["obje"]] if aday.get("obje") else [])
            metin = (f"**{aday.get('t', mmss(sn))}** · {aday.get('tip', '?')} · "
                     f"{aday.get('detay', '')} — {', '.join(objeler)}")
            if abs(t_sn - sn) <= OLAY_TOLERANS_SN:
                # aktif aday: hareketsizlik daha kritik → kirmizi, digeri sari
                if aday.get("tip") == "hareketsizlik_adayi":
                    st.error(metin)
                else:
                    st.warning(metin)
            else:
                st.caption(metin)


def vlm_panel(t_sn):
    """vlm_rapor.json'dan gecerli zamana en yakin (t <= simdi) kare analizi."""
    with vlm_ph.container():
        if vlm_verisi is None:
            st.info("`vlm_rapor.json` bulunamadı — `python vLLM.py <video>` ile üretin.")
            return
        if not vlm_eslesme:
            st.info(f"Bu video için VLM raporu yok (mevcut rapor: `{vlm_video}`) — "
                    f"`python vLLM.py {video_yolu}` ile üretin.")
            return
        analizler = vlm_verisi.get("kare_analizleri", [])
        gecmis = [a for a in analizler if zaman_sn(a.get("timestamp")) <= t_sn + 0.01]
        if not gecmis:
            st.info("Bu ana kadar VLM analizi yok — video ilerledikçe görünecek.")
            return
        a = max(gecmis, key=lambda x: zaman_sn(x.get("timestamp")))
        try:
            skor = int(float(a.get("risk_skoru", 0)))
        except (ValueError, TypeError):
            skor = 0
        baslik = f"[{a.get('timestamp', '?')}] {a.get('olay_turu', '?')} — risk {skor}/10"
        if skor >= 7:  # vLLM.py tool_risk_evaluate/final_report ile AYNI esik (7)
            st.error(baslik)
        elif skor >= 5:
            st.warning(baslik)
        else:
            st.success(baslik)
        if a.get("acil_mudahale"):
            st.markdown("🚨 **Acil müdahale gerekli!**")
        st.markdown(a.get("olay_yorumu", ""))
        plan = a.get("aksiyon_plani") or []
        if plan:
            st.markdown("**Aksiyon planı:**")
            for adim in plan:
                st.markdown(f"- {adim}")


def kare_goster(kare_bgr, t_sn, toplam_sn, oran):
    """Tek karenin tum panellerle birlikte ekrana basilmasi."""
    tespitler = tespit_et(model, kare_bgr, conf, imgsz)
    kutulari_ciz(kare_bgr, tespitler)
    rgb = cv2.cvtColor(kare_bgr, cv2.COLOR_BGR2RGB)  # her kare BGR→RGB
    zaman_ph.markdown(f"⏱ **{mmss(t_sn)} / {mmss(toplam_sn)}** &nbsp;·&nbsp; `{video_yolu}`")
    ilerleme_ph.progress(min(max(oran, 0.0), 1.0))
    video_ph.image(rgb, width="stretch")
    yolo_panel(t_sn, tespitler)
    olay_panel(t_sn)
    vlm_panel(t_sn)


# ── Alt bolum: nihai rapor + sartname ciktisi ────────────────────────────────

st.divider()
with st.expander("📄 Nihai VLM Raporu", expanded=False):
    if vlm_verisi is None:
        st.info("`vlm_rapor.json` bulunamadı — `python vLLM.py <video>` ile üretin.")
    elif not vlm_eslesme:
        st.info(f"Bu video için VLM raporu yok (mevcut rapor: `{vlm_video}`).")
    else:
        rapor = vlm_verisi.get("rapor")
        if not rapor:
            st.info("Rapor alanı henüz üretilmemiş.")
        else:
            risk = str(rapor.get("genel_risk", "?"))
            if risk == "Yüksek":
                st.error(f"Genel risk: **{risk}** · Kritik olay: {rapor.get('kritik_olay_sayisi', 0)}")
            elif risk == "Orta":
                st.warning(f"Genel risk: **{risk}** · Kritik olay: {rapor.get('kritik_olay_sayisi', 0)}")
            else:
                st.success(f"Genel risk: **{risk}** · Kritik olay: {rapor.get('kritik_olay_sayisi', 0)}")
            st.markdown(rapor.get("video_ozeti", ""))
            st.json(rapor)

with st.expander("📋 Şartname JSON Çıktısı", expanded=False):
    if vlm_verisi is None:
        st.info("`vlm_rapor.json` bulunamadı — `python vLLM.py <video>` ile üretin.")
    elif not vlm_eslesme:
        st.info(f"Bu video için VLM raporu yok (mevcut rapor: `{vlm_video}`).")
    else:
        sartname = vlm_verisi.get("sartname_json")
        if not sartname:
            st.info("Şartname çıktısı henüz üretilmemiş.")
        else:
            st.json(sartname)

# ── Video acilisi ve oynatma dongusu ─────────────────────────────────────────

cap = cv2.VideoCapture(video_yolu)
if not cap.isOpened():
    st.error(f"Video açılamadı: `{video_yolu}`")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
toplam_kare = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
toplam_sn = toplam_kare / fps if toplam_kare else 0.0

if toplam_kare <= 0:
    st.error(f"Videoda kare bulunamadı: `{video_yolu}`")
    cap.release()
    st.stop()

if st.session_state.oynat:
    # OYNATMA: tek while dongusu + placeholder guncelleme (st.rerun akisi YOK).
    # Duraklat'a basilinca Streamlit calisan script'i keser ve yeniden calistirir;
    # kare_no her adimda session_state'e yazildigi icin konum KORUNUR.
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.kare_no)
    while st.session_state.oynat and st.session_state.kare_no < toplam_kare:
        t0 = time.time()
        ok, kare = cap.read()
        if not ok:
            break
        for _ in range(kare_atlama - 1):  # atlanan karelerin sadece basligini oku (hizli)
            cap.grab()
        t_sn = st.session_state.kare_no / fps
        kare_goster(kare, t_sn, toplam_sn, st.session_state.kare_no / toplam_kare)
        st.session_state.kare_no = min(st.session_state.kare_no + kare_atlama, toplam_kare)
        # ~gercek hiz: islenen sure dusulerek kare basi bekleme
        bekle = kare_atlama / fps - (time.time() - t0)
        if bekle > 0:
            time.sleep(bekle)
    st.session_state.oynat = False  # video sonu: durdur, konum sonda kalir
    cap.release()
else:
    # DURAKLATILMIS/BEKLEMEDE: mevcut konumdaki kareyi statik goster
    kare_no = min(st.session_state.kare_no, toplam_kare - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, kare_no)
    ok, kare = cap.read()
    cap.release()
    if ok:
        kare_goster(kare, kare_no / fps, toplam_sn, kare_no / toplam_kare)
    else:
        st.warning("Kare okunamadı — Sıfırla düğmesiyle başa dönün.")
