import os, glob, math

# === CONFIG ===
ROOT = "/home/v.bucciero/data/instruments/"     # base dir
DATA_DIR = os.path.join(ROOT, "rdr0")         # dove sono tutte le .tiff
OUT_DIR  = os.path.join(ROOT, "rdr0_splits")  # dove creeremo train/val/test
PATTERN = "**/*.tiff"                         # adatta se serve
RATIOS = (0.90, 0.09, 0.01)                   # train, val, test
DRY_RUN = False                               # True per provare senza creare symlink

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, PATTERN), recursive=True))
    if not files:
        raise SystemExit(f"Nessun file trovato in {DATA_DIR} con pattern {PATTERN}")

    n = len(files)
    n_train = int(math.floor(n * RATIOS[0]))
    n_val   = int(math.floor(n * RATIOS[1]))
    n_test  = n - n_train - n_val

    train_files = files[:n_train]
    val_files   = files[n_train:n_train+n_val]
    test_files  = files[n_train+n_val:]

    print(f"Totale: {n}  | train: {len(train_files)}  val: {len(val_files)}  test: {len(test_files)}")

    # cartelle output (manteniamo una struttura semplice)
    train_dir = os.path.join(OUT_DIR, "train")
    val_dir   = os.path.join(OUT_DIR, "val")
    test_dir  = os.path.join(OUT_DIR, "test")
    for d in (train_dir, val_dir, test_dir):
        ensure_dir(d)

    def link_into(subset_files, subset_dir):
        for src in subset_files:
            # mantieni l'eventuale sotto-struttura (se avevi sottocartelle)
            rel = os.path.relpath(src, DATA_DIR)
            dst_dir = os.path.join(subset_dir, os.path.dirname(rel))
            ensure_dir(dst_dir)
            dst = os.path.join(dst_dir, os.path.basename(src))
            try:
                if os.path.islink(dst) or os.path.exists(dst):
                    continue
                if not DRY_RUN:
                    os.symlink(src, dst)
            except FileExistsError:
                pass

    link_into(train_files, train_dir)
    link_into(val_files,   val_dir)
    link_into(test_files,  test_dir)

    print(f"Split creato in: {OUT_DIR}")
    print("Esempi:")
    print(f"  train -> {train_dir}")
    print(f"  val   -> {val_dir}")
    print(f"  test  -> {test_dir}")

if __name__ == "__main__":
    main()
