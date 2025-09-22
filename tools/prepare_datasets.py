import argparse, os, shutil
from pathlib import Path
from tqdm import tqdm
import cv2

def read_list(txt_path):
    with open(txt_path, 'r') as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_annotations_for_stem(ann_dir: Path, stem: str):
    p = ann_dir / f"{stem}.txt"
    if not p.exists():
        return []
    boxes = []
    with open(p, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 5:
                if len(parts) == 4:
                    x1, y1, x2, y2 = map(float, parts)
                    cls = 1
                else:
                    continue
            else:
                x1, y1, x2, y2, cls = map(float, parts[:5])
            boxes.append((x1, y1, x2, y2, int(cls)))
    return boxes

def to_yolo_line(x1,y1,x2,y2,w,h):
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

def convert_split(image_dir, ann_dir, stems, out_img, out_lbl):
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    for stem in tqdm(stems, desc=f"Converting {out_img.name}"):
        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
            cand = image_dir / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            print(f"[WARN] Missing image for stem={stem}")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read {img_path}")
            continue
        h, w = img.shape[:2]
        dst_img = out_img / img_path.name
        if not dst_img.exists():
            shutil.copyfile(img_path, dst_img)
        boxes = load_annotations_for_stem(ann_dir, stem)
        yolo_txt = out_lbl / (img_path.stem + ".txt")
        with open(yolo_txt, "w") as f:
            for (x1,y1,x2,y2,cls) in boxes:
                f.write(to_yolo_line(x1,y1,x2,y2,w,h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpk-root", type=str, required=False)
    ap.add_argument("--pucpr-root", type=str, required=False)
    ap.add_argument("--out", type=str, default="datasets/yolo")
    ap.add_argument("--val-split", type=float, default=0.0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images/train").mkdir(parents=True, exist_ok=True)
    (out / "labels/train").mkdir(parents=True, exist_ok=True)
    (out / "images/val").mkdir(parents=True, exist_ok=True)
    (out / "labels/val").mkdir(parents=True, exist_ok=True)
    (out / "images/test").mkdir(parents=True, exist_ok=True)
    (out / "labels/test").mkdir(parents=True, exist_ok=True)

    def process_devkit(root_dir):
        root = Path(root_dir)
        img_dir = root / "Images"
        ann_dir = root / "Annotations"
        isets = root / "ImageSets"
        train_stems = read_list(isets / "train.txt")
        test_stems = read_list(isets / "test.txt")
        return img_dir, ann_dir, train_stems, test_stems

    if args.carpk_root:
        img_dir, ann_dir, tr, te = process_devkit(args.carpk_root)
        convert_split(img_dir, ann_dir, tr, out/"images/train", out/"labels/train")
        convert_split(img_dir, ann_dir, te, out/"images/test", out/"labels/test")

    if args.pucpr_root:
        img_dir, ann_dir, tr, te = process_devkit(args.pucpr_root)
        convert_split(img_dir, ann_dir, tr, out/"images/train", out/"labels/train")
        convert_split(img_dir, ann_dir, te, out/"images/test", out/"labels/test")

    yaml_path = out / "parkinglot.yaml"
    yaml = f"""
# Single-class car detection
path: {out.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: car
"""
    yaml_path.write_text(yaml.strip()+"\n", encoding="utf-8")
    print(f"Done. YAML at {yaml_path}")

if __name__ == "__main__":
    main()
