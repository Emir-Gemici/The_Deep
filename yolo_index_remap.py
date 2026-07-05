#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_index_remap.py
YOLO etiketlerini GUVENLI sekilde yeniden indeksleme + temizleme araci.

NE ISE YARAR
  Ham bir kaynak veri setini (kendi index sirasiyla gelen) bizim 5 sinifli
  hedef semamiza cevirir:
      0 = forklift   1 = insan   2 = palet   3 = baret   4 = yelek
  - SOURCE_TO_TARGET'taki kaynak indeksleri hedef indekslerle degistirir.
  - Bu eslemede OLMAYAN her satiri (istenmeyen COCO/diger sinif) tamamen SILER.
  - Tum satirlari silinen .txt'yi ve eslesen gorseli KARANTINAYA tasir (silmez).
  - Islem oncesi etiketlerin yedegini alir.
  - --dry-run ile hicbir seyi degistirmeden once rapor verir.
  - Koordinatlar STRING olarak aynen korunur (yalnizca ilk token = sinif degisir).

!!! BU PROJEYE OZEL UYARILAR
  * birlesik/ ZATEN temiz (etiketler 0-4, insan=1 -- bu oturumda dogrulandi).
    Bu araci birlesik/ UZERINDE CALISTIRMA: gereksiz ve veriyi BOZAR.
  * Arac, HAM kaynak setleri birlestirmeden ONCE temizlemek icindir.
  * Her kaynak setin index sirasi FARKLIDIR: o setin KENDI data.yaml'ini OKU,
    SOURCE_TO_TARGET'i ona gore doldur. ASLA tahminle doldurma.
  * ONCE her zaman --dry-run calistir, raporu oku, sonra gercegini calistir.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

# ============================== AYARLAR ====================================

# Hedef sema (sadece raporlama icin)
TARGET_NAMES: dict[int, str] = {
    0: "forklift",
    1: "insan",
    2: "palet",
    3: "baret",
    4: "yelek",
}

# KAYNAK index -> HEDEF index.
# Her kaynak setin data.yaml'ina gore DOLDUR. Burada OLMAYAN kaynak index = SIL.
# Ornekler (bu oturumda entegre ettigimiz setler):
#   Warehouse Detection  names=['Forklift','Person'] -> {0: 0, 1: 1}
#   Industrial Person    names=['Person']            -> {0: 1}
SOURCE_TO_TARGET: dict[int, int] = {
    0: 0,   # ornek: kaynak 'Forklift' -> hedef 0 (forklift)
    1: 1,   # ornek: kaynak 'Person'   -> hedef 1 (insan)
}

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# ===========================================================================


def find_image(images_dir: Path, stem: str) -> Path | None:
    """Etiket koku ile eslesen gorseli (uzantisi ne olursa olsun) bulur."""
    if not images_dir.is_dir():
        return None
    for suf in IMAGE_SUFFIXES:
        cand = images_dir / f"{stem}{suf}"
        if cand.exists():
            return cand
    for cand in images_dir.glob(f"{stem}.*"):   # uzanti varyasyonlari icin emniyet
        if cand.suffix.lower() in IMAGE_SUFFIXES:
            return cand
    return None


def remap_lines(lines: list[str], mapping: dict[int, int], stats: Counter) -> list[str]:
    """Bir etiketin satirlarini remap/filtrele eder; yeni satir listesi doner."""
    out: list[str] = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        # KRITIK: ilk token'i INTEGER olarak al; ' 1 ' gibi substring araması YAPMA.
        try:
            src = int(parts[0])
        except ValueError:
            stats["bozuk_satir"] += 1
            continue
        if src in mapping:
            parts[0] = str(mapping[src])
            out.append(" ".join(parts))      # koordinatlar aynen korunur
            stats["guncellenen_satir"] += 1
            stats[f"hedef_{mapping[src]}"] += 1
        else:
            stats["silinen_nesne"] += 1
            stats[f"silinen_kaynak_{src}"] += 1
    return out


def backup_labels(labels_dir: Path, backup_root: Path) -> Path:
    """labels klasorunu SPLIT adiyla yedekler (train/labels ve valid/labels cakismaz)."""
    split = labels_dir.parent.name or "split"
    dest = backup_root / split / labels_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for txt in labels_dir.glob("*.txt"):
        shutil.copy2(txt, dest / txt.name)
    return dest


def quarantine(txt: Path, img: Path | None, removed_root: Path) -> None:
    """Bosalan etiketi ve gorselini karantina klasorune TASIR (kalici silmez)."""
    split = txt.parent.parent.name or "split"
    (removed_root / split / "labels").mkdir(parents=True, exist_ok=True)
    (removed_root / split / "images").mkdir(parents=True, exist_ok=True)
    shutil.move(str(txt), str(removed_root / split / "labels" / txt.name))
    if img is not None and img.exists():
        shutil.move(str(img), str(removed_root / split / "images" / img.name))


def process_split(labels_dir: Path, images_dir: Path, mapping: dict[int, int],
                  backup_root: Path | None, removed_root: Path,
                  dry_run: bool, stats: Counter) -> None:
    if not labels_dir.is_dir():
        print(f"  [ATLA] labels klasoru bulunamadi: {labels_dir}")
        return

    txts = sorted(labels_dir.glob("*.txt"))
    print(f"\n>>> {labels_dir}  ({len(txts)} .txt)")
    if not dry_run and backup_root is not None:
        print(f"    yedek: {backup_labels(labels_dir, backup_root)}")

    for txt in txts:
        stats["dosya"] += 1
        lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        new_lines = remap_lines(lines, mapping, stats)

        if new_lines:
            stats["kalan_dosya"] += 1
            if not dry_run:
                txt.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        else:
            stats["bosalan_dosya"] += 1
            img = find_image(images_dir, txt.stem)
            if img is None:
                stats["gorselsiz_bosalan"] += 1
            if not dry_run:
                quarantine(txt, img, removed_root)


def parse_map_args(pairs: list[str] | None) -> dict[int, int]:
    if not pairs:
        return dict(SOURCE_TO_TARGET)
    m: dict[int, int] = {}
    for p in pairs:
        src, dst = p.split(":")
        m[int(src)] = int(dst)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO etiket index remap + temizleme araci")
    ap.add_argument("--labels", action="append", required=True,
                    help="Bir split'in labels klasoru. Birden cok kez verilebilir: "
                         "--labels .../train/labels --labels .../valid/labels")
    ap.add_argument("--images", action="append", default=None,
                    help="Karsilik gelen images klasoru (ayni sira). Verilmezse "
                         "labels yolundaki 'labels' -> 'images' ile turetilir.")
    ap.add_argument("--map", action="append", default=None,
                    help="kaynak:hedef eslesme, or. --map 0:0 --map 1:1. "
                         "Verilmezse koddaki SOURCE_TO_TARGET kullanilir.")
    ap.add_argument("--backup-dir", default="_yedek_etiketler",
                    help="Islem oncesi etiket yedegi (default: _yedek_etiketler)")
    ap.add_argument("--removed-dir", default="_silinenler",
                    help="Bosalan dosyalarin karantina klasoru (default: _silinenler)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Hicbir seyi degistirmeden sadece raporla (ONCE BUNU CALISTIR)")
    args = ap.parse_args()

    mapping = parse_map_args(args.map)
    if not mapping:
        ap.error("Eslesme (mapping) bos olamaz.")

    label_dirs = [Path(p) for p in args.labels]
    if args.images:
        image_dirs = [Path(p) for p in args.images]
    else:
        image_dirs = [Path(str(p).replace("labels", "images")) for p in label_dirs]
    if len(image_dirs) != len(label_dirs):
        ap.error("--images sayisi --labels sayisina esit olmali.")

    backup_root = (None if args.dry_run
                   else Path(args.backup_dir) / datetime.now().strftime("%Y%m%d_%H%M%S"))
    removed_root = Path(args.removed_dir)

    print("=" * 66)
    print("MOD:", "DRY-RUN (DEGISIKLIK YOK)" if args.dry_run else "GERCEK (DEGISIKLIK VAR)")
    print("Eslesme (kaynak -> hedef):")
    for s, d in sorted(mapping.items()):
        print(f"   {s} -> {d} ({TARGET_NAMES.get(d, '??')})")
    print("Bu eslesmede OLMAYAN tum kaynak indeksleri SILINIR.")
    print("=" * 66)

    stats: Counter = Counter()
    for ld, idd in zip(label_dirs, image_dirs):
        process_split(ld, idd, mapping, backup_root, removed_root, args.dry_run, stats)

    # ------------------------------- RAPOR ---------------------------------
    print("\n" + "=" * 66)
    print("RAPOR")
    print("=" * 66)
    print(f"Islenen .txt dosyasi             : {stats['dosya']}")
    print(f"Guncellenen satir                : {stats['guncellenen_satir']}")
    print(f"Silinen (istenmeyen) nesne       : {stats['silinen_nesne']}")
    print(f"Bozuk (sayisal olmayan ilk token): {stats['bozuk_satir']}")
    print(f"Korunan dosya                    : {stats['kalan_dosya']}")
    msg = f"Bosalip karantinaya tasinan dosya: {stats['bosalan_dosya']}"
    if stats["gorselsiz_bosalan"]:
        msg += f"  (gorseli bulunamayan: {stats['gorselsiz_bosalan']})"
    print(msg)

    print("\nHedef sinif dagilimi (guncellenen satirlar):")
    for d in sorted(TARGET_NAMES):
        c = stats.get(f"hedef_{d}", 0)
        if c:
            print(f"   {d} {TARGET_NAMES[d]:<9}: {c}")

    silinen = {k: v for k, v in stats.items() if k.startswith("silinen_kaynak_")}
    if silinen:
        print("\nSilinen kaynak indeks dagilimi (atilan siniflar):")
        for k, v in sorted(silinen.items(), key=lambda kv: int(kv[0].split("_")[-1])):
            print(f"   kaynak {k.split('_')[-1]}: {v}")

    if args.dry_run:
        print("\n[DRY-RUN] Hicbir dosya degismedi. Gercek icin --dry-run'i kaldir.")


if __name__ == "__main__":
    main()
