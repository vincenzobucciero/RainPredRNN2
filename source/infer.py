# -*- coding: utf-8 -*-

import os, re, time, statistics, datetime, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio

# ====== importa dal tuo app9.py ======
# Assumo che RainPredRNN e PRED_LENGTH=6 siano definiti lì.
# In alternativa copia le classi nel presente file.
from app9 import RainPredRNN, PRED_LENGTH

# ====== CONFIG ======
CKPT_PATH  = "/home/vbucciero/projects/RainPredRNN2/source/checkpoints/best_model.pth"
INPUT_DIR  = "/home/vbucciero/projects/RainPredRNN2/extracted_tiff"     
OUTPUT_DIR = "./pred_out"                    # dove salvare le 6 predizioni
REPEATS    = 50                              # n. misure
WARMUP     = 10                              # warm-up
USE_AMP    = True                            # mixed precision su GPU
RESIZE_HW  = (224, 224)                      # coerente col training
STEP_MIN   = 10                              # passo temporale atteso (minuti) per il naming

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_image(img):
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0, None)
    min_dbz, max_dbz = 0, 70
    img = np.clip(img, min_dbz, max_dbz)
    return (img - min_dbz) / (max_dbz - min_dbz)  # [0,1]

def load_18_stack(input_dir):
    paths = sorted(
        glob.glob(os.path.join(input_dir, "*.tif"))
        + glob.glob(os.path.join(input_dir, "*.tiff"))
    )
    if len(paths) != 18:
        raise RuntimeError(f"Attesi 18 TIFF in {input_dir}, trovati {len(paths)}.")
    frames = []
    for p in paths:
        with rasterio.open(p) as src:
            img = src.read(1).astype(np.float32)  # banda 1
        img = normalize_image(img)                       # [0,1]
        img = Image.fromarray((img * 255.0).astype(np.uint8))
        img = img.resize(RESIZE_HW, Image.Resampling.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0  # [0,1]
        arr = (arr - 0.5) / 0.5                           # Normalize(mean=0.5,std=0.5) -> [-1,1]
        arr = arr[None, ...]                               # (C=1, H, W)
        frames.append(arr)
    x = np.stack(frames, axis=0)                           # (T=18, C, H, W)
    return torch.from_numpy(x), paths

_ts_re = re.compile(r".*_(\d{8}Z\d{4})_.*")  # es: _20251008Z0110_
def next_names_from_last(last_name, k=6, step_minutes=10):
    m = _ts_re.match(last_name)
    out = []
    if not m:
        # fallback generico
        stem = os.path.splitext(os.path.basename(last_name))[0]
        for i in range(1, k+1):
            out.append(os.path.join(OUTPUT_DIR, f"{stem}_t+{i:02d}_pred.tiff"))
        return out
    ts = m.group(1)  # es "20251008Z0110"
    dt = datetime.datetime.strptime(ts, "%Y%m%dZ%H%M")
    prefix = last_name.split(ts)[0]
    suffix = last_name.split(ts)[1]
    for i in range(1, k+1):
        ndt = dt + datetime.timedelta(minutes=step_minutes*i)
        nts = ndt.strftime("%Y%m%dZ%H%M")
        out_name = f"{prefix}{nts}{suffix}"
        base, ext = os.path.splitext(out_name)
        out.append(os.path.join(OUTPUT_DIR, f"{base}_pred{ext if ext else '.tiff'}"))
    return out

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = RainPredRNN(input_dim=1, num_hidden=256, max_hidden_channels=128,
                        patch_height=16, patch_width=16).to(device)
    model.eval()

    # carica pesi
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Pesi caricati: {CKPT_PATH}")

    # carica 18 frame
    inputs_np, paths = load_18_stack(INPUT_DIR)  # (18,1,H,W)
    print("Input frames (18):")
    for p in paths:
        print("  -", os.path.basename(p))

    # prepara tensore (B=1, T, C, H, W)
    inputs = inputs_np.unsqueeze(0).to(device, non_blocking=True)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


    # warm-up
    with torch.inference_mode():
        for _ in range(WARMUP):
            if device == "cuda": torch.cuda.synchronize()
            if USE_AMP and device == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(inputs, PRED_LENGTH)
            else:
                _ = model(inputs, PRED_LENGTH)
            if device == "cuda": torch.cuda.synchronize()

    # misure
    times = []
    with torch.inference_mode():
        for _ in range(REPEATS):
            if device == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            if USE_AMP and device == "cuda":
                with torch.cuda.amp.autocast():
                    preds, _ = model(inputs, PRED_LENGTH)  # preds in [-1,1]
            else:
                preds, _ = model(inputs, PRED_LENGTH)
            if device == "cuda": torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    mean_t = statistics.mean(times)
    std_t  = statistics.pstdev(times)
    p50    = statistics.median(times)
    p95    = sorted(times)[int(0.95 * len(times)) - 1]
    fps    = 1.0 / mean_t if mean_t > 0 else float("inf")

    print("\n=== Inference 18→6 (batch=1) ===")
    print(f"Ripetizioni       : {REPEATS}")
    print(f"Tempo medio (s)   : {mean_t:.4f}")
    print(f"Dev std (s)       : {std_t:.4f}")
    print(f"P50 (s)           : {p50:.4f}")
    print(f"P95 (s)           : {p95:.4f}")
    print(f"Predizioni/s      : {fps:.2f}")
    print("Nota: solo forward; niente I/O/metriche.")

    # ===== salva le 6 predizioni =====
    preds = preds.detach().cpu().numpy()  # (1, T=6, C=1, H, W)
    preds = preds[0, :, 0]                # (6, H, W)
    preds = preds * 0.5 + 0.5             # [-1,1] -> [0,1]
    out_names = next_names_from_last(os.path.basename(paths[-1]), k=PRED_LENGTH, step_minutes=STEP_MIN)

    for t in range(PRED_LENGTH):
        frame = (preds[t] * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(frame).save(out_names[t])

    print(f"\nSalvate {PRED_LENGTH} predizioni in: {OUTPUT_DIR}")
    for n in out_names:
        print("  -", os.path.basename(n))

if __name__ == "__main__":
    main()
