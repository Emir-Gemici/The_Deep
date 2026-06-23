"""
autolabel.py — birlesik'teki ETIKETSIZ insanlari base COCO yolo26s ile otomatik etiketler.
Sorun: PPE/forklift gorsellerinde insanlar var ama 'insan' etiketsiz -> model insani BASTIRIYOR.
Cozum: COCO 'person' tespitlerini insan(1) olarak ekle (mevcut etiketlere dokunmadan, IoU dedup).
  python autolabel.py --preview   # hesapla + ornek + sayim (DEGISIKLIK YOK)
  python autolabel.py --apply     # yedekle + etiketleri guncelle
"""
import os, sys, json, argparse, shutil
from pathlib import Path
from datetime import datetime
import cv2
from ultralytics import YOLO

CONF, SPLITS, PROP = 0.4, ["train", "valid", "test"], "auto_insan_proposed.json"
N = {0:"forklift",1:"insan",2:"palet",3:"baret",4:"yelek"}
COL = {0:(0,200,0),1:(0,0,255),2:(255,150,0),3:(0,200,200),4:(200,0,200)}

def iou(a, b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0.,ix2-ix1)*max(0.,iy2-iy1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.

def preview():
    base=YOLO("yolo26s.pt"); proposals={}; samples=[]
    for split in SPLITS:
        ld,idir=Path(f"birlesik/{split}/labels"),Path(f"birlesik/{split}/images")
        timg=tbox=nf=0
        for txt in sorted(ld.glob("*.txt")):
            nf+=1; ip=idir/(txt.stem+".jpg")
            if not ip.exists(): continue
            img=cv2.imread(str(ip))
            if img is None: continue
            h,w=img.shape[:2]
            rows=[l.strip() for l in txt.read_text().splitlines() if l.strip()]
            existing=[]; cls_here=set()
            for r in rows:
                p=r.split(); c=int(p[0]); cls_here.add(c)
                if c==1:
                    xc,yc,bw,bh=map(float,p[1:5])
                    existing.append([(xc-bw/2)*w,(yc-bh/2)*h,(xc+bw/2)*w,(yc+bh/2)*h])
            newb=[]
            for b in base.predict(img,imgsz=640,conf=CONF,verbose=False)[0].boxes:
                if base.names[int(b.cls)]!="person": continue
                box=[float(v) for v in b.xyxy[0].tolist()]
                if any(iou(box,e)>0.5 for e in existing): continue
                xc=min(max(((box[0]+box[2])/2)/w,0.),1.); yc=min(max(((box[1]+box[3])/2)/h,0.),1.)
                bw=min(max((box[2]-box[0])/w,0.),1.); bh=min(max((box[3]-box[1])/h,0.),1.)
                newb.append(f"1 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            if newb:
                proposals[f"{split}|{txt.stem}"]=newb; timg+=1; tbox+=len(newb)
                if split=="train" and (3 in cls_here or 4 in cls_here) and 1 not in cls_here and len(samples)<4:
                    vis=img.copy()
                    for r in rows:
                        p=r.split(); c=int(p[0]); xc,yc,bw,bh=map(float,p[1:5])
                        x1,y1=int((xc-bw/2)*w),int((yc-bh/2)*h); x2,y2=int((xc+bw/2)*w),int((yc+bh/2)*h)
                        cv2.rectangle(vis,(x1,y1),(x2,y2),COL[c],2); cv2.putText(vis,N[c],(x1,max(y1-4,12)),cv2.FONT_HERSHEY_SIMPLEX,0.5,COL[c],2)
                    for nb in newb:
                        _,xc,yc,bw,bh=nb.split(); xc,yc,bw,bh=float(xc),float(yc),float(bw),float(bh)
                        x1,y1=int((xc-bw/2)*w),int((yc-bh/2)*h); x2,y2=int((xc+bw/2)*w),int((yc+bh/2)*h)
                        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,0,255),3); cv2.putText(vis,"insan YENI",(x1,max(y1-4,12)),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),2)
                    sp=f"autolabel_ornek_{len(samples)}.jpg"; cv2.imwrite(sp,vis); samples.append(sp)
        print(f"  {split:5}: {nf} dosya | insan EKLENECEK gorsel: {timg} | yeni insan kutusu: {tbox}")
    json.dump(proposals,open(PROP,"w"))
    print(f"\nOneriler -> {PROP} ({len(proposals)} gorsel). Ornekler: {samples}")

def apply():
    if not os.path.exists(PROP): sys.exit("Once --preview calistir.")
    proposals=json.load(open(PROP))
    bk=Path("_yedek_etiketler")/("autolabel_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    for split in SPLITS:
        src=Path(f"birlesik/{split}/labels"); dst=bk/split; dst.mkdir(parents=True,exist_ok=True)
        for f in src.glob("*.txt"): shutil.copy2(f,dst/f.name)
    print("Yedek:",bk)
    t=a=0
    for key,boxes in proposals.items():
        split,stem=key.split("|",1); txt=Path(f"birlesik/{split}/labels/{stem}.txt")
        cur=txt.read_text().rstrip("\n"); cur=(cur+"\n") if cur else ""
        txt.write_text(cur+"\n".join(boxes)+"\n"); t+=1; a+=len(boxes)
    print(f"Guncellendi: {t} gorsel, +{a} insan kutusu.")

if __name__=="__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ap=argparse.ArgumentParser(); ap.add_argument("--preview",action="store_true"); ap.add_argument("--apply",action="store_true")
    a=ap.parse_args()
    apply() if a.apply else preview()
