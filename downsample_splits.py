#!/usr/bin/env python3
import os, math, argparse, shutil

def gather_sorted(root):
    files = []
    for cur, _, fnames in os.walk(root, followlinks=True):
        for f in fnames:
            if f.lower().endswith((".tif",".tiff")):
                files.append(os.path.join(cur, f))
    files.sort()
    return files

def take_stride(files, target):
    if target >= len(files):
        return files[:]  # già pochi
    stride = max(1, len(files)//target)
    picked = files[::stride]
    # taglia esatti target
    return picked[:target]

def link_preserve(src_root, dst_root, files):
    for src in files:
        rel = os.path.relpath(src, src_root)
        dst = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            if not os.path.exists(dst):
                os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="cartella split sorgente (con train/val/test)")
    ap.add_argument("--dst", required=True, help="cartella split destinazione ridotti")
    ap.add_argument("--total", type=int, default=5000, help="totale desiderato (default 5000)")
    args = ap.parse_args()

    # proporzioni 70/15/15
    t_train = int(round(args.total*0.70))
    t_val   = int(round(args.total*0.15))
    t_test  = args.total - t_train - t_val

    print(f"Target: total={args.total} -> train={t_train}, val={t_val}, test={t_test}")

    src_train = os.path.join(args.src, "train")
    src_val   = os.path.join(args.src, "val")
    src_test  = os.path.join(args.src, "test")

    dst_train = os.path.join(args.dst, "train")
    dst_val   = os.path.join(args.dst, "val")
    dst_test  = os.path.join(args.dst, "test")

    for d in (dst_train, dst_val, dst_test):
        os.makedirs(d, exist_ok=True)

    f_train = gather_sorted(src_train)
    f_val   = gather_sorted(src_val)
    f_test  = gather_sorted(src_test)

    print(f"Sorgente: train={len(f_train)}, val={len(f_val)}, test={len(f_test)}")

    pick_train = take_stride(f_train, t_train)
    pick_val   = take_stride(f_val,   t_val)
    pick_test  = take_stride(f_test,  t_test)

    print(f"Selezionati: train={len(pick_train)}, val={len(pick_val)}, test={len(pick_test)}")

    link_preserve(src_train, dst_train, pick_train)
    link_preserve(src_val,   dst_val,   pick_val)
    link_preserve(src_test,  dst_test,  pick_test)

    # check finali
    def count(dst):
        c=0
        for cur,_,fn in os.walk(dst):
            for f in fn:
                if f.lower().endswith((".tif",".tiff")):
                    c+=1
        return c

    print("Conteggi finali (dest):",
          "train=", count(dst_train),
          "val=", count(dst_val),
          "test=", count(dst_test))

if __name__ == "__main__":
    main()

