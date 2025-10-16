# make_splits_clean.py (estratto minimo)
import os, glob, math, shutil

ROOT = "/storage/external_01/hiwefi/data"
DATA_DIR = os.path.join(ROOT, "rdr0_3k")
OUT_DIR  = os.path.join(ROOT, "rdr0_3k_splits_clean")
os.makedirs(OUT_DIR, exist_ok=True) 

# raccogli file robustamente
def gather_images(root):
    files = []
    for cur, _, fnames in os.walk(root, followlinks=True):
        for f in fnames:
            if f.lower().endswith((".tif", ".tiff")):
                files.append(os.path.join(cur, f))
    return sorted(files)

files = gather_images(DATA_DIR)
n = len(files)
r_train, r_val, r_test = 0.90, 0.09, 0.01
n_train = int(math.floor(n*r_train))
n_val   = int(math.floor(n*r_val))
train, val, test = files[:n_train], files[n_train:n_train+n_val], files[n_train+n_val:]

def link_into(subset, subset_dir):
    for src in subset:
        rel = os.path.relpath(src, DATA_DIR)
        dst_dir = os.path.join(subset_dir, os.path.dirname(rel))
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        try:
            if not os.path.exists(dst):
                os.symlink(src, dst)
        except Exception:
            # fallback copia se symlink non permesso
            shutil.copy2(src, dst)

for name, subset in [("train", train), ("val", val), ("test", test)]:
    link_into(subset, os.path.join(OUT_DIR, name))
print("Split creato in", OUT_DIR)
