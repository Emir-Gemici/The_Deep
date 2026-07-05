"""
tracking.py — final modelle nesne TAKIBI (kalici ID).

model.track(persistent=True) her objeye kareler arasi KALICI ID verir.
Her ID'nin zaman icindeki konum/boyutu (track gecmisi) tutulur — bu gecmis,
olay-turetme (devrilme / hareketsizlik / toplanma) icin TEMEL veridir.

Kullanim:
    python tracking.py <video>            # varsayilan: Test.mp4
Pipeline icin:
    from tracking import track_video
    history, fps = track_video("video.mp4")   # history: {track_id: [{frame,t,cls,cx,cy,w,h}, ...]}
"""
import os
import sys
import collections
import cv2
from ultralytics import YOLO

os.chdir(os.path.dirname(os.path.abspath(__file__)))
MODEL = "son_model.pt"


def track_video(video, imgsz=960, conf=0.25, save=True, show=True):
    """Videoyu takip eder; her kalici ID icin zaman serisi gecmisi dondurur."""
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    history = collections.defaultdict(list)  # track_id -> [{frame,t,cls,cx,cy,w,h}, ...]
    results = model.track(source=video, show=show, imgsz=imgsz, conf=conf, persist=True,
                          tracker="custom_tracker.yaml", stream=True, save=save, verbose=False)
    for fno, r in enumerate(results):
        if r.boxes is None or r.boxes.id is None:
            continue
        ids = r.boxes.id.int().tolist()
        clss = r.boxes.cls.int().tolist()
        xyxy = r.boxes.xyxy.tolist()
        for tid, c, (x1, y1, x2, y2) in zip(ids, clss, xyxy):
            history[tid].append({
                "frame": fno,
                "t": round(fno / fps, 2),
                "cls": model.names[c],
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "w": x2 - x1,
                "h": y2 - y1,
            })
    return history, fps


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "Test (2).mp4"
    print(f"Takip ediliyor: {video} ...")
    history, fps = track_video(video)

    by_cls = collections.Counter(h[0]["cls"] for h in history.values())
    print(f"\n=== {video} (fps={fps:.0f}) ===")
    print(f"Toplam benzersiz track (kalici ID): {len(history)}")
    print(f"Sinif bazli track sayisi          : {dict(by_cls)}")
    print("\nEn uzun 5 track:")
    for tid in sorted(history, key=lambda k: -len(history[k]))[:5]:
        h = history[tid]
        print(f"  ID {tid:3} ({h[0]['cls']:8}): {len(h):4} kare | t={h[0]['t']}-{h[-1]['t']} sn")
