"""
kare_cikar.py — Video(lar)dan ETIKETLENEBILIR kareler cikarir (dataset olusturma araci).

Neden: Video saniyede ~30 kare uretir; ardisik kareler neredeyse AYNI. Hepsini
etiketlemek gereksiz (cesitlilik yok) ve cok fazla. Bu arac:
  1) belirli araliklarla kare alir (her N saniyede 1),
  2) bir oncekine cok benzeyen kareleri eler (cesitlilik),
  3) bulanik kareleri eler (netlik),
sonucta CESITLI + NET kareleri .jpg olarak kaydeder -> Roboflow'a yuklenip etiketlenir.

Kullanim:
  python kare_cikar.py <video_veya_klasor>
  python kare_cikar.py kazalar/ --her-sn 1.0 --cikti kareler --benzerlik 6 --bulaniklik 100

Parametreler:
  --her-sn      : kac saniyede 1 kare alinsin (default 1.5). Kucuk = daha cok kare.
  --benzerlik   : onceki kareye benzerlik esigi (default 6). Kucuk = daha siki eleme.
  --bulaniklik  : min netlik skoru (default 100). Kucuk = bulanik kareleri de tut.
  --cikti       : cikti klasoru (default 'kareler').
"""
import os
import sys
import glob
import argparse
import cv2

VIDEO_UZANTI = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def netlik_skoru(img):
    """Laplacian varyansi: dusuk deger = bulanik kare."""
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gri, cv2.CV_64F).var()


def benzer_mi(a, b, esik):
    """Iki kare (kucultulmus gri) ortalama farki esikten kucukse benzer kabul."""
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (32, 32))
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (32, 32))
    return float(cv2.absdiff(ga, gb).mean()) < esik


def video_isle(video, outdir, her_sn, benzerlik, bulaniklik):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(her_sn * fps))
    stem = os.path.splitext(os.path.basename(video))[0]
    fno = 0
    kaydedilen = 0
    atlanan_bulanik = 0
    atlanan_benzer = 0
    son_kayit = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fno % step == 0:
            if netlik_skoru(frame) < bulaniklik:
                atlanan_bulanik += 1
            elif son_kayit is not None and benzer_mi(frame, son_kayit, benzerlik):
                atlanan_benzer += 1
            else:
                cv2.imwrite(os.path.join(outdir, f"{stem}_{fno:06d}.jpg"), frame)
                son_kayit = frame
                kaydedilen += 1
        fno += 1
    cap.release()
    return kaydedilen, atlanan_bulanik, atlanan_benzer


def main():
    ap = argparse.ArgumentParser(description="Videolardan etiketlenebilir kare cikar")
    ap.add_argument("kaynak", help="video dosyasi VEYA video klasoru")
    ap.add_argument("--her-sn", type=float, default=1.5)
    ap.add_argument("--benzerlik", type=float, default=6.0)
    ap.add_argument("--bulaniklik", type=float, default=100.0)
    ap.add_argument("--cikti", default="kareler")
    a = ap.parse_args()

    os.makedirs(a.cikti, exist_ok=True)
    if os.path.isfile(a.kaynak):
        vids = [a.kaynak]
    else:
        vids = [v for v in glob.glob(os.path.join(a.kaynak, "*.*"))
                if v.lower().endswith(VIDEO_UZANTI)]
    if not vids:
        sys.exit("Video bulunamadi.")

    toplam = 0
    for v in vids:
        k, b, s = video_isle(v, a.cikti, a.her_sn, a.benzerlik, a.bulaniklik)
        print(f"  {os.path.basename(v):40} -> {k:4} kare (bulanik atlandi: {b}, benzer atlandi: {s})")
        toplam += k
    print(f"\nToplam {toplam} kare -> '{a.cikti}/'  | Sonraki: Roboflow'a yukle + etiketle")


if __name__ == "__main__":
    main()
