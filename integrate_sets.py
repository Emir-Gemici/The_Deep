"""5 yeni seti birlesik'e entegre eder (remap + sadece hedef siniflar + cakisma korumasi)."""
import os, shutil, random
from pathlib import Path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# (set_klasoru, {kaynak_idx: hedef_idx}, train_cap)
CONFIG = [
    ("Mask_RCNN.v4i.yolo26", {0:2}, None),                                        # pallet->palet
    ("pallet.v1i.yolo26", {1:2}, None),                                           # pallet(1)->palet, '0' at
    ("Safety helmet and reflection vest detection.v1i.yolo26", {0:4, 1:3}, None), # vest->yelek, helmet->baret
    ("warehouse.v2i.yolo26", {0:0, 1:2}, None),                                   # forklift->forklift, pallet->palet
    ("Safety Vests.v14-rf-detr-medium-576x576.yolo26", {1:4}, 4000),              # safety_vest->yelek (train cap 4000)
]
IMG_EXT = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
TARGET = {0:"forklift", 1:"insan", 2:"palet", 3:"baret", 4:"yelek"}

def find_img(idir, stem):
    for e in IMG_EXT:
        p = idir / f"{stem}{e}"
        if p.exists():
            return p
    return None

audit, collisions = [], 0
for setname, mapping, cap in CONFIG:
    for split in ['train', 'valid', 'test']:
        ld, idir = Path(f"{setname}/{split}/labels"), Path(f"{setname}/{split}/images")
        if not ld.is_dir():
            continue
        bld, bidir = Path(f"birlesik/{split}/labels"), Path(f"birlesik/{split}/images")
        files = sorted(ld.glob("*.txt"))
        if cap and split == 'train':
            random.seed(42); random.shuffle(files)
        copied, cc = 0, {}
        for txt in files:
            if cap and split == 'train' and copied >= cap:
                break
            rows = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
            kept = []
            for r in rows:
                p = r.split()
                try:
                    c = int(p[0])
                except ValueError:
                    continue
                if c in mapping:
                    p[0] = str(mapping[c]); kept.append(" ".join(p)); cc[mapping[c]] = cc.get(mapping[c], 0) + 1
            if not kept:
                continue
            stem = txt.stem; img = find_img(idir, stem)
            if img is None:
                continue
            di, dl = bidir / (stem + img.suffix), bld / (stem + ".txt")
            if di.exists() or dl.exists():
                collisions += 1; continue
            shutil.copy2(img, di); dl.write_text("\n".join(kept) + "\n")
            audit.append(f"{split}\t{stem}{img.suffix}"); copied += 1
        print(f"  {setname[:34]:34} {split:5}: +{copied:5}  { {TARGET[k]:v for k,v in cc.items()} }")
Path("eklenen_yeni_setler_listesi.txt").write_text("\n".join(audit) + "\n", encoding="utf-8")
print(f"\nTOPLAM +{len(audit)} gorsel eklendi. Cakisma (atlandi): {collisions}")
