# fill_holes.py
import os, shutil, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/v.bucciero/data/instruments/rdr0_splits/train/"
MIN_SIZE = 1024  # byte: < MIN_SIZE lo consideriamo buco (es. 128 B)
LOG_PATH = os.path.join(ROOT, "_fillholes_log.txt")

replaced = 0
first_bad = 0
checked = 0

with open(LOG_PATH, "w") as log:
    for cur, _, files in os.walk(ROOT):
        # ordina per nome: di solito è già ordine temporale (timestamp nel filename)
        tif = sorted([f for f in files if f.lower().endswith((".tif", ".tiff"))])
        prev_valid = None
        for fname in tif:
            path = os.path.join(cur, fname)
            checked += 1
            try:
                size = os.path.getsize(path)
            except FileNotFoundError:
                continue

            if size < MIN_SIZE:
                if prev_valid:
                    shutil.copy2(prev_valid, path)
                    replaced += 1
                    log.write(f"REPL {path} <- {prev_valid}\n")
                else:
                    # primo file del gruppo è già bucato: non abbiamo un "precedente" da copiare
                    first_bad += 1
                    log.write(f"SKIP(no-prev) {path}\n")
            else:
                prev_valid = path

print(f"Controllati: {checked}  |  Rimpiazzati: {replaced}  |  Primi buchi: {first_bad}")
print(f"Log: {LOG_PATH}")
