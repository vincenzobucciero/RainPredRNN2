import os
import glob
import shutil

# === Percorsi sorgente e destinazione ===
SRC_ROOT = os.path.expanduser("~/data/instruments/rdr0_splits/val")
DST_DIR  = os.path.expanduser("~/projects/hiwefi/RainPredRNN2")

# === Numero di file da estrarre ===
N_FILES = 18

def main():
    # trova tutti i TIFF ricorsivamente
    tiff_files = sorted(
        glob.glob(os.path.join(SRC_ROOT, "**", "*.tif"), recursive=True)
        + glob.glob(os.path.join(SRC_ROOT, "**", "*.tiff"), recursive=True)
    )

    if len(tiff_files) < N_FILES:
        print(f"❌ Trovati solo {len(tiff_files)} file in {SRC_ROOT}, servono almeno {N_FILES}.")
        return

    print(f"✅ Trovati {len(tiff_files)} file totali. Estraggo i primi {N_FILES} contigui...\n")

    os.makedirs(DST_DIR, exist_ok=True)
    selected_files = tiff_files[:N_FILES]

    for src_path in selected_files:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(DST_DIR, filename)
        shutil.copy2(src_path, dst_path)
        print(f"📄 Copiato: {src_path}  →  {dst_path}")

    print(f"\n✅ Copiati {len(selected_files)} file in: {DST_DIR}")

if __name__ == "__main__":
    main()
