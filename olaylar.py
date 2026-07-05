"""
olaylar.py — tracking history'sinden OLAY ADAYI sinyalleri turetir.

Cikti: zaman damgali olay adayi listesi (JSON) -> Bera/Selim'in VLM'ine girdi.
NOT: Bu katman SADECE *sinyal* uretir. Anlamlandirma (kaza mi, risk, Turkce
     aciklama, aksiyon) multimodal VLM'in isidir (kareleri o goruyor).

AKTIF SINYALLER (guvenilir, konum/hareket tabanli):
  - hareketsizlik_adayi : insan track'in merkezi N sn ~sabit
  - toplanma_adayi      : >=K insan track'in merkezleri yakinlasir

DEVRE DISI (denendi, guvenilmez/gurultu -> VLM'e devredildi; fonksiyonlar referans icin durur):
  - devrilme: bbox-orani (w/h) forklift'in DONUSUNU (onden->yandan goruntu) gercek
    devrilmeyle karistiriyor -> yanlis pozitif; gercek devrilmeyi de w/h zaten genis
    oldugu icin iskaliyor. Devrilme tanima multimodal VLM'e devredildi (kareleri goruyor).
  - forklift_insan_yakinlik: forklift sahnesinde insan-forklift yakinligi NORM oldugu
    icin ciktiyi boguyor (anomali degil). Tehlikeli-yakinlik karari VLM'e birakildi.

Takip pahali oldugu icin history cache'lenir (_history_<video>.json).
Kullanim: python olaylar.py <video>   (varsayilan Test.mp4)
"""
import sys
import os
import json
import math
import statistics
import collections
from tracking import track_video

MIN_TRACK = 30  # bu kareden kisa track'ler gurultu -> ele


def mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def _smooth(vals, k=15):
    h = k // 2
    return [statistics.median(vals[max(0, i - h):i + h + 1]) for i in range(len(vals))]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ---- DENEYSEL / KULLANILMIYOR: bbox-orani devrilme (guvenilmez, VLM'e devredildi) ----
def devrilme(history, fps):
    """DENEYSEL — cagrilmaz. bbox w/h dik->yatik gecisi; forklift DONUSUNU devrilmeyle
    karistirdigi icin guvenilmez bulundu. Devrilme tanima VLM'e devredildi."""
    out = []
    win = max(5, int(1.5 * fps))
    for tid, h in history.items():
        if h[0]["cls"] != "forklift" or len(h) < 2 * win:
            continue
        sm = _smooth([d["w"] / max(d["h"], 1e-6) for d in h], 15)
        for i in range(win, len(sm) - win):
            pre = sum(sm[i - win:i]) / win
            post = sum(sm[i:i + win]) / win
            if pre < 1.05 and post > 1.4:
                out.append({"saniye": round(h[i]["t"], 1), "tip": "devrilme_adayi",
                            "obje": f"forklift#{tid}", "detay": f"w/h {pre:.2f}->{post:.2f}"})
                break
    return out


def hareketsizlik(history, fps, pencere_sn=2.5, oran=0.12):
    """Insan track'in merkezi pencere boyunca bbox boyutuna gore cok az hareket ederse."""
    out = []
    win = max(5, int(pencere_sn * fps))
    for tid, h in history.items():
        if h[0]["cls"] != "insan" or len(h) < win:
            continue
        for i in range(0, len(h) - win, max(1, win // 2)):
            seg = h[i:i + win]
            dx = max(d["cx"] for d in seg) - min(d["cx"] for d in seg)
            dy = max(d["cy"] for d in seg) - min(d["cy"] for d in seg)
            size = sum(max(d["w"], d["h"]) for d in seg) / len(seg)
            if math.hypot(dx, dy) < oran * size:
                out.append({"saniye": round(seg[0]["t"], 1), "tip": "hareketsizlik_adayi",
                            "obje": f"insan#{tid}", "detay": f"~{pencere_sn}sn hareketsiz"})
                break
    return out


def toplanma(history, fps, k=3, ornek_sn=1.0):
    """Ayni anda >=k insan track'in merkezi birbirine yakinsa = toplanma adayi."""
    by_frame = collections.defaultdict(list)
    for tid, h in history.items():
        if h[0]["cls"] != "insan" or len(h) < MIN_TRACK:
            continue
        for d in h:
            by_frame[d["frame"]].append((tid, d["cx"], d["cy"], max(d["w"], d["h"])))
    out = []
    step = max(1, int(ornek_sn * fps))
    son = -999.0
    for fno in sorted(by_frame):
        if fno % step:
            continue
        pts = by_frame[fno]
        if len(pts) < k:
            continue
        for ti, xi, yi, si in pts:
            near = {tj for tj, xj, yj, _ in pts if math.hypot(xi - xj, yi - yj) < 2.0 * si}
            if len(near) >= k:
                t = fno / fps
                if t - son > 3.0:
                    out.append({"saniye": round(t, 1), "tip": "toplanma_adayi",
                                "objeler": [f"insan#{x}" for x in sorted(near)[:8]],
                                "detay": f"{len(near)} kisi yakin"})
                    son = t
                break
    return out


def yakinlik(history, fps, ornek_sn=1.0, mesafe=1.0, surucu_iou=0.3):
    """DENEYSEL — cagrilmaz. Forklift sahnesinde insan-forklift yakinligi NORM oldugu icin
    ciktiyi boguyor (anomali degil). Tehlikeli-yakinlik karari VLM'e birakildi."""
    fk = collections.defaultdict(list)
    ins = collections.defaultdict(list)
    for tid, h in history.items():
        if len(h) < MIN_TRACK:
            continue
        tgt = fk if h[0]["cls"] == "forklift" else (ins if h[0]["cls"] == "insan" else None)
        if tgt is None:
            continue
        for d in h:
            box = (d["cx"] - d["w"] / 2, d["cy"] - d["h"] / 2, d["cx"] + d["w"] / 2, d["cy"] + d["h"] / 2)
            tgt[d["frame"]].append((tid, d["cx"], d["cy"], max(d["w"], d["h"]), box))
    out = []
    step = max(1, int(ornek_sn * fps))
    last = {}
    for fno in sorted(set(fk) & set(ins)):
        if fno % step:
            continue
        t = fno / fps
        for fi, fx, fy, fs, fbox in fk[fno]:
            for ii, ix, iy, isz, ibox in ins[fno]:
                if math.hypot(fx - ix, fy - iy) < mesafe * fs and _iou(fbox, ibox) < surucu_iou:
                    key = (ii, fi)
                    if t - last.get(key, -999) > 4.0:
                        out.append({"saniye": round(t, 1), "tip": "forklift_insan_yakinlik_adayi",
                                    "objeler": [f"insan#{ii}", f"forklift#{fi}"],
                                    "detay": "insan forklifte yakin (surucu degil)"})
                        last[key] = t
    return out


def olaylari_turet(video, use_cache=True):
    cache = f"_history_{os.path.splitext(os.path.basename(video))[0]}.json"
    if use_cache and os.path.exists(cache):
        d = json.load(open(cache, encoding="utf-8"))
        fps = d["fps"]
        history = {int(k): v for k, v in d["history"].items()}
        print(f"(history cache yuklendi: {cache})")
    else:
        history, fps = track_video(video, save=False, show=False)
        json.dump({"fps": fps, "history": {str(k): v for k, v in history.items()}},
                  open(cache, "w", encoding="utf-8"))
    fps = fps or 30.0
    # AKTIF guvenilir sinyaller. devrilme + yakinlik DEVRE DISI -> VLM'e devredildi
    # (devrilme: donus/aci ile karisir; yakinlik: forklift sahnesinde norm, ciktiyi bogar)
    ev = hareketsizlik(history, fps) + toplanma(history, fps)
    for e in ev:
        e["t"] = mmss(e["saniye"])
    ev.sort(key=lambda o: o["saniye"])
    return ev, len(history)


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "Test.mp4"
    print(f"Olay turetiliyor: {video} ...")
    olaylar, ntrack = olaylari_turet(video)
    cikti = {"video": video, "track_sayisi": ntrack, "olay_adaylari": olaylar}
    print(f"\n=== {len(olaylar)} olay adayi ({ntrack} track) ===")
    print(json.dumps(cikti, ensure_ascii=False, indent=2))
    with open("olaylar_cikti.json", "w", encoding="utf-8") as f:
        json.dump(cikti, f, ensure_ascii=False, indent=2)
    print("\n-> olaylar_cikti.json kaydedildi (VLM girdisi)")
