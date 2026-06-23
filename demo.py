"""
demo.py — Vision pipeline GORSEL DEMO videosu uretir.
Takip kutulari + kalici ID + olay aninda ekran banner'i (DEVRILME/TOPLANMA/HAREKETSIZLIK).
Cache'lenmis history'den calisir (model tekrar kosmaz). Kullanim: python demo.py <video>
"""
import os, sys, json, collections
import cv2
from olaylar import hareketsizlik, toplanma, mmss
from tracking import track_video

os.chdir(os.path.dirname(os.path.abspath(__file__)))

COL = {"insan": (0, 0, 255), "forklift": (0, 200, 0), "palet": (255, 150, 0),
       "baret": (0, 200, 200), "yelek": (200, 0, 200)}
EVCOL = {"devrilme_adayi": (0, 0, 255), "toplanma_adayi": (0, 140, 255), "hareketsizlik_adayi": (0, 210, 255)}
EVTXT = {"devrilme_adayi": "DEVRILME", "toplanma_adayi": "TOPLANMA", "hareketsizlik_adayi": "HAREKETSIZLIK"}


def get_history(video):
    cache = f"_history_{os.path.splitext(os.path.basename(video))[0]}.json"
    if os.path.exists(cache):
        d = json.load(open(cache, encoding="utf-8"))
        return {int(k): v for k, v in d["history"].items()}, d["fps"]
    history, fps = track_video(video, save=False, show=False)
    json.dump({"fps": fps, "history": {str(k): v for k, v in history.items()}},
              open(cache, "w", encoding="utf-8"))
    return history, fps


def main(video):
    history, fps = get_history(video)
    events = hareketsizlik(history, fps) + toplanma(history, fps)  # devrilme/yakinlik VLM'e devredildi
    for e in events:
        e["t"] = mmss(e["saniye"])
    by_frame = collections.defaultdict(list)
    for tid, h in history.items():
        if len(h) < 30:
            continue
        for dd in h:
            by_frame[dd["frame"]].append((tid, dd["cls"], dd["cx"], dd["cy"], dd["w"], dd["h"]))

    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save = "demo_" + os.path.splitext(os.path.basename(video))[0] + ".mp4"
    vw = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fno = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fno / fps
        for tid, cls, cx, cy, bw, bh in by_frame.get(fno, []):
            x1, y1, x2, y2 = int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)
            c = COL.get(cls, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f"{cls}#{tid}", (x1, max(y1 - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        y0 = 46
        for e in [ev for ev in events if 0 <= t - ev["saniye"] < 3.0]:
            col = EVCOL[e["tip"]]; txt = f">> {EVTXT[e['tip']]}  {e['t']}"
            cv2.rectangle(frame, (8, y0 - 30), (8 + int(15.5 * len(txt)), y0 + 8), (0, 0, 0), -1)
            cv2.putText(frame, txt, (14, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)
            y0 += 42
        vw.write(frame); fno += 1
    cap.release(); vw.release()
    print(f"{fno} kare islendi -> {save} | {len(events)} olay banner'i gomuldu")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Test.mp4")
