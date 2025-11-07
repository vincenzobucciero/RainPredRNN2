'''
import os
import glob
import shutil

# === Percorsi sorgente e destinazione ===
SRC_ROOT = os.path.expanduser("/storage/external_01/hiwefi/data/rdr0_splits")
DST_DIR  = os.path.expanduser("/home/vbucciero/projects/RainPredRNN2/extracted_tiff")

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
'''

#!/usr/bin/env python3
import os, glob, shutil, json, random, argparse
from pathlib import Path

STATE_FILE_NAME = ".extract_state.json"

def list_tiffs(root):
    return sorted(
        glob.glob(os.path.join(root, "**", "*.tif"), recursive=True)
        + glob.glob(os.path.join(root, "**", "*.tiff"), recursive=True)
    )

def load_state(state_path):
    if state_path.exists():
        try:
            return json.loads(state_path.read_text()).get("start", 0)
        except Exception:
            return 0
    return 0

def save_state(state_path, start):
    try:
        state_path.write_text(json.dumps({"start": start}))
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser(description="Estrai una finestra contigua di TIFF.")
    ap.add_argument("--src", default="/storage/external_01/hiwefi/data/rdr0_splits/test",
                    help="Root sorgente (ricorsivo).")
    ap.add_argument("--dst", default="/home/vbucciero/projects/RainPredRNN2/extracted_tiff",
                    help="Cartella destinazione.")
    ap.add_argument("--n", type=int, default=18, help="Numero di file da estrarre (finestra).")
    ap.add_argument("--mode", choices=["start", "random", "next"], default="start",
                    help="Selezione finestra: 'start'=usa --start; 'random'=casuale; 'next'=scorrevole con stato.")
    ap.add_argument("--start", type=int, default=0, help="Indice di partenza (usato con mode=start).")
    ap.add_argument("--seed", type=int, default=None, help="Seed RNG (per ripetibilità in mode=random).")
    ap.add_argument("--clear-dst", action="store_true", help="Svuota la cartella di destinazione prima di copiare.")
    ap.add_argument("--dry-run", action="store_true", help="Non copiare, solo stampa cosa farebbe.")
    args = ap.parse_args()

    src_root = os.path.expanduser(args.src)
    dst_dir  = Path(os.path.expanduser(args.dst))
    dst_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = list_tiffs(src_root)
    if len(tiff_files) < args.n:
        print(f"❌ Trovati solo {len(tiff_files)} file in {src_root}, servono almeno {args.n}.")
        return

    total = len(tiff_files)
    total_windows = total - args.n + 1
    if total_windows <= 0:
        print(f"❌ Non esistono finestre contigue di dimensione {args.n} nei {total} file.")
        return

    # determina lo start in base alla modalità
    state_path = dst_dir / STATE_FILE_NAME
    if args.mode == "start":
        start = max(0, min(args.start, total_windows - 1))
    elif args.mode == "random":
        if args.seed is not None:
            random.seed(args.seed)
        start = random.randint(0, total_windows - 1)
    else:  # "next"
        last = load_state(state_path)
        start = (last + args.n) if (last is not None) else 0
        # avanzare di n salta alla finestra successiva non sovrapposta; usa +1 se vuoi scorrimento a passo 1
        start = start % total_windows
        save_state(state_path, start)

    end = start + args.n
    selected = tiff_files[start:end]

    print(f"✅ File totali: {total} | Finestre possibili: {total_windows}")
    print(f"📦 Modalità: {args.mode} | start={start} | n={args.n}")
    print("🗂️  Selezionati:")
    for p in selected:
        print("  -", p)

    if args.dry_run:
        print("\n(Modalità dry-run: nessuna copia eseguita.)")
        return

    if args.clear_dst:
        for old in dst_dir.glob("*.tif*"):
            try: old.unlink()
            except Exception: pass

    copied = 0
    for src_path in selected:
        dst_path = dst_dir / os.path.basename(src_path)
        shutil.copy2(src_path, dst_path)
        copied += 1
    print(f"\n✅ Copiati {copied} file in: {dst_dir}")

if __name__ == "__main__":
    main()
