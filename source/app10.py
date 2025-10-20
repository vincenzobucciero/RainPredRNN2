import os
import math
import glob
import datetime
import time
import numpy as np
from functools import partial
from PIL import Image
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision.transforms as transforms
import torchvision.ops as vops

import rasterio
from rasterio.errors import RasterioIOError

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix
from pytorch_msssim import SSIM

import torchio as tio
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.utils.tensorboard.writer import SummaryWriter

# =========================================
# Config "pro-GPU"
# =========================================
# Suggerimento cluster:
# export OMP_NUM_THREADS=1
# #SBATCH --cpus-per-task=32 (o più, se disponibili)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Batch iniziale alto: verrà auto-ridotto se OOM
INIT_BATCH_SIZE = 64
MAX_NUM_WORKERS = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1
PRED_LENGTH = 6
PIN_MEMORY = True
USE_AMP = True  # mixed precision
USE_BF16 = True # H100: bfloat16 consigliato
TORCH_COMPILE = True  # PyTorch 2.x: prova torch.compile
BENCH_WARMUP = 10
BENCH_MEASURE = 50

# Ottimizzazioni globali per H100
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

# =========================================
# Utils / Seed
# =========================================
def set_seed(seed=15):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

def normalize_image(img):
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0, None)
    min_dbz, max_dbz = 0, 70
    img = np.clip(img, min_dbz, max_dbz)
    return (img - min_dbz) / (max_dbz - min_dbz)

def get_augmentation_transforms():
    return tio.Compose([
        tio.RandomFlip(axes=(0, 1), p=0.5),
        tio.RandomAffine(scales=(0.8, 1.2), degrees=90, p=0.5),
    ])

def tiff_is_readable_quick(path):
    try:
        with rasterio.open(path) as src:
            _ = src.read(1, window=((0, 1), (0, src.width)))
        return True
    except Exception:
        return False

def _hms(sec: float):
    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

@contextmanager
def autocast_ctx():
    if DEVICE == "cuda" and USE_AMP:
        if USE_BF16 and torch.cuda.is_bf16_supported():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield
    else:
        yield

# =========================================
# Dataset
# =========================================
class RadarDataset(Dataset):
    def __init__(self, data_path, input_length=18, pred_length=6, is_train=True,
                 generate_mask=True, return_paths=False, min_size=1024):
        self.input_length = input_length
        self.pred_length = pred_length
        self.seq_length = input_length + pred_length
        self.min_size = min_size

        self.files = []
        self.files += glob.glob(os.path.join(data_path, '**/*.tif'), recursive=True)
        self.files += glob.glob(os.path.join(data_path, '**/*.tiff'), recursive=True)
        self.files = sorted(self.files)

        if len(self.files) == 0:
            raise RuntimeError(f"Nessun file trovato in {data_path}. Controlla path/permessi/estensioni.")
        if len(self.files) < self.seq_length:
            raise RuntimeError(f"Troppi pochi file in {data_path}: {len(self.files)} < seq_length={self.seq_length}")

        self.is_train = is_train
        # Canale singolo → torchvision Normalize(0.5,0.5)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),                       # (C,H,W) in [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.file_validity = {}
        self.valid_indices = []
        self.total_possible_windows = max(0, len(self.files) - self.seq_length + 1)
        self.augmentation_transforms = get_augmentation_transforms() if is_train else None

        for start_idx in range(self.total_possible_windows):
            ok = True
            for i in range(self.seq_length):
                f = self.files[start_idx + i]
                if f not in self.file_validity:
                    try:
                        size = os.path.getsize(f)
                    except FileNotFoundError:
                        size = 0
                    if size < self.min_size:
                        valid = False
                    else:
                        valid = tiff_is_readable_quick(f)
                        if not valid:
                            print(f"File non valido (read fail): {f}")
                    self.file_validity[f] = valid
                if not self.file_validity[f]:
                    ok = False
                    break
            if ok:
                self.valid_indices.append(start_idx)

        self.total_files = len(self.files)
        self.invalid_files = sum(1 for v in self.file_validity.values() if not v)
        self.valid_windows = len(self.valid_indices)
        self.invalid_windows = self.total_possible_windows - self.valid_windows

        print(f"\nStatistiche Dataset ({'train' if is_train else 'val'}):")
        print(f"1. File totali: {self.total_files}")
        print(f"2. File non validi: {self.invalid_files}")
        print(f"3. Finestre totali possibili: {self.total_possible_windows}")
        print(f"4. Finestre valide: {self.valid_windows}")
        print(f"5. Finestre non valide: {self.invalid_windows}")
        print(" ===================================================== \n")

        self.generate_mask = generate_mask
        self.return_paths = return_paths

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        max_resamples = 8
        attempts = 0
        base_idx = idx

        while attempts < max_resamples:
            start = self.valid_indices[idx]
            images = []
            target_frame_paths = []
            failed = False

            for i in range(self.seq_length):
                f = self.files[start + i]
                try:
                    with rasterio.open(f) as src:
                        img = src.read(1).astype(np.float32)
                except Exception:
                    self.file_validity[f] = False
                    failed = True
                    if self.is_train:
                        idx = (idx + np.random.randint(1, 64)) % len(self.valid_indices)
                    else:
                        idx = (idx + 1) % len(self.valid_indices)
                    attempts += 1
                    break

                img = normalize_image(img)
                img = Image.fromarray((img * 255.0).astype(np.uint8))
                img = self.transform(img)
                images.append(img)
                if i >= self.input_length:
                    target_frame_paths.append(f)

            if failed:
                continue

            all_frames = torch.stack(images)            # (seq, C, H, W)
            all_frames = all_frames.permute(1, 0, 2, 3) # (C, seq, H, W)

            if self.is_train and self.augmentation_transforms is not None:
                all_frames = self.augmentation_transforms(all_frames)

            if not isinstance(all_frames, torch.Tensor):
                all_frames = torch.as_tensor(all_frames)

            inputs  = all_frames[:, :self.input_length]       # (C, Tin, H, W)
            targets = all_frames[:, self.input_length:]       # (C, Tout, H, W)

            inputs  = inputs.permute(1, 0, 2, 3).contiguous(memory_format=torch.channels_last)   # (Tin, C, H, W)
            targets = targets.permute(1, 0, 2, 3).contiguous(memory_format=torch.channels_last)  # (Tout, C, H, W)

            mask = None
            if self.generate_mask:
                mask = torch.where(targets > -1.0, 1.0, 0.0).contiguous(memory_format=torch.channels_last)

            if self.return_paths:
                if len(target_frame_paths) != self.pred_length:
                    raise RuntimeError(
                        f"target_frame_paths len={len(target_frame_paths)} diversa da pred_length={self.pred_length} "
                        f"per indice finestra {start}. Controlla i file nella sequenza."
                    )
                return inputs, targets, mask, target_frame_paths
            return inputs, targets, mask

        raise RasterioIOError(f"Nessuna finestra valida dopo {max_resamples} tentativi; indice iniziale {base_idx}.")

# =========================================
# Modello (encoder/decoder + transformer)
# =========================================
class UNet_Encoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels,64,3,1,1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(64,64,3,1,1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Sequential(nn.Conv2d(64,128,3,1,1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(128,128,3,1,1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); skip1 = x
        x = self.pool1(x); x = self.conv3(x); x = self.conv4(x)
        return x, skip1

class UNet_Decoder(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.conv5 = nn.Sequential(nn.Conv2d(256,128,3,1,1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.conv6 = nn.Sequential(nn.Conv2d(128,128,3,1,1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.up1   = nn.Sequential(nn.ConvTranspose2d(128,64,2,2,bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.conv7 = nn.Sequential(nn.Conv2d(128,64,3,1,1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.conv8 = nn.Sequential(nn.Conv2d(64,64,3,1,1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)
    def forward(self, x, skip0, skip1):
        x = torch.cat([skip0, x], dim=1)
        x = self.conv5(x); x = self.conv6(x)
        x = self.up1(x)
        x = torch.cat([skip1, x], dim=1)
        x = self.conv7(x); x = self.conv8(x)
        x = self.final_conv(x)
        x_last = x
        x = torch.tanh(x)
        return x, x_last

def generate_positional_encoding(seq_len, d_model, device):
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0)/d_model))
    pe[:,0::2] = torch.sin(position*div_term)
    pe[:,1::2] = torch.cos(position*div_term)
    return pe.unsqueeze(0)  # (1, seq, d)

class TemporalTransformerBlock(nn.Module):
    def __init__(self, channels, d_model, nhead, num_encoder_layers, pred_length, patch_height, patch_width):
        super().__init__()
        self.pred_length = pred_length
        self.patch_height = patch_height
        self.patch_width  = patch_width
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b (t h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_encoder_layers
        )
        self.to_feature_map = nn.Sequential(nn.Linear(d_model, patch_dim), nn.LayerNorm(patch_dim))

    def forward(self, x):
        # x: (B, Tin, C, H, W)
        H, W = x.shape[-2:]
        ph = H // self.patch_height
        pw = W // self.patch_width

        x = self.to_patch_embedding(x)  # (B, Tin*ph*pw, d)
        B, T, D = x.shape

        pe = generate_positional_encoding(T, D, x.device)
        mem = self.encoder(x + pe)  # (B, Tin*ph*pw, d)

        tokens_per_frame = ph * pw
        needed = self.pred_length * tokens_per_frame
        assert needed <= T, f"pred_length ({self.pred_length}) * ph*pw ({tokens_per_frame}) > Tin*ph*pw ({T})"
        mem = mem[:, -needed:, :]  # (B, pred_length*ph*pw, d)

        out = self.to_feature_map(mem)  # (B, pred_length*ph*pw, p1*p2*C)
        out = rearrange(
            out,
            'b (t h w) (p1 p2 c) -> b t c (h p1) (w p2)',
            t=self.pred_length, h=ph, w=pw,
            p1=self.patch_height, p2=self.patch_width
        )
        return out

class RainPredRNN(nn.Module):
    def __init__(self, input_dim=1, num_hidden=256, max_hidden_channels=128, patch_height=16, patch_width=16):
        super().__init__()
        self.encoder = UNet_Encoder(input_dim)
        self.decoder = UNet_Decoder(input_dim)
        self.transformer_block = TemporalTransformerBlock(
            channels=max_hidden_channels, d_model=num_hidden, nhead=8,
            num_encoder_layers=3, pred_length=PRED_LENGTH,
            patch_height=patch_height, patch_width=patch_width
        )
    def forward(self, input_sequence, pred_length):
        B, Tin, C, H, W = input_sequence.size()
        enc_feats, skip1 = [], []
        for t in range(Tin):
            x, sk1 = self.encoder(input_sequence[:, t])
            enc_feats.append(x); skip1.append(sk1)
        enc_feats = torch.stack(enc_feats, dim=1)  # (B, Tin, Cenc, H/2, W/2)
        skip1 = torch.stack(skip1, dim=1)

        pred_feats = self.transformer_block(enc_feats)   # (B, Tout, Cenc, H/2, W/2)

        preds, preds_noact = [], []
        for t in range(pred_length):
            y, y_no = self.decoder(pred_feats[:, t], enc_feats[:, t], skip1[:, t])
            preds.append(y); preds_noact.append(y_no)
        return torch.stack(preds, dim=1), torch.stack(preds_noact, dim=1)

# =========================================
# Collate Fn (VALIDAZIONE)
# =========================================
def collate_val(batch):
    item = batch[0]
    if len(item) != 4:
        raise RuntimeError("Validation loader deve restituire (inputs, targets, mask, paths).")
    inputs, targets, mask, paths = item  # (Tin, C, H, W)

    if isinstance(paths, (list, tuple)) and paths and isinstance(paths[0], (list, tuple)):
        paths = list(paths[0])
    elif isinstance(paths, (list, tuple)):
        paths = list(paths)
    else:
        paths = [paths]

    if inputs.dim() == 4:
        inputs = inputs.unsqueeze(0)
    if targets.dim() == 4:
        targets = targets.unsqueeze(0)
    if mask is not None and mask.dim() == 4:
        mask = mask.unsqueeze(0)

    # channels_last per velocizzare convoluzioni
    inputs  = inputs.contiguous(memory_format=torch.channels_last)
    targets = targets.contiguous(memory_format=torch.channels_last)
    if mask is not None:
        mask = mask.contiguous(memory_format=torch.channels_last)

    return inputs, targets, mask, paths

# =========================================
# Dataloaders + autotuning batch size
# =========================================
def _pin_kwargs():
    kwargs = {}
    if hasattr(torch.utils.data, "DataLoader"):
        # pin_memory_device (PyTorch 2.0+)
        try:
            kwargs["pin_memory_device"] = "cuda"
        except TypeError:
            pass
    return kwargs

def create_dataloaders(data_path, batch_size=64, num_workers=32):
    train_dataset = RadarDataset(os.path.join(data_path, 'train'), is_train=True, pred_length=PRED_LENGTH)
    val_dataset   = RadarDataset(os.path.join(data_path, 'val'),   is_train=False, return_paths=True, pred_length=PRED_LENGTH)

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=True,
        prefetch_factor=8,
        drop_last=True,
        **_pin_kwargs()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs
    )
    val_loader   = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_val,
        **{**loader_kwargs, "drop_last": False}
    )
    return train_loader, val_loader

def autotune_batch_size(model, data_path, init_bs=64, num_workers=32):
    """Prova batch size decrescente finché una short-run forward/backward non va OOM."""
    bs = max(1, int(init_bs))
    while bs >= 1:
        try:
            train_loader, val_loader = create_dataloaders(data_path, bs, num_workers)
            # Prova un singolo step
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))
            batch = next(iter(train_loader))
            inputs, targets, mask = batch[:3]
            inputs  = inputs.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
            targets = targets.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
            mask    = mask.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                outputs, logits = model(inputs, PRED_LENGTH)
                loss = F.smooth_l1_loss(outputs, targets)*10 + vops.sigmoid_focal_loss(logits, mask, reduction='mean')*10
            if torch.cuda.is_available(): torch.cuda.synchronize()
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward(); optimizer.step()
            del optimizer, scaler, batch, inputs, targets, mask, outputs, logits, loss
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print(f"[AutoBS] Batch size OK: {bs}")
            return bs, train_loader, val_loader
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = bs // 2
            print(f"[AutoBS] OOM: riduco batch size a {bs}")
    raise RuntimeError("Autotuning batch size fallito anche con bs=1.")

# =========================================
# Benchmark
# =========================================
def benchmark_train(loader, model, optimizer, device, scaler=None, warmup=10, measure=50):
    model.train()
    it = iter(loader)

    for _ in range(warmup):
        batch = next(it)
        inputs, targets, mask = batch[:3]
        inputs  = inputs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        mask    = mask.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            outputs, logits = model(inputs, PRED_LENGTH)
            loss = criterion_sl1(outputs, targets) * criterion_mae_lambda + criterion_fl(logits, mask) * criterion_mae_lambda
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    counted = 0
    for _ in range(measure):
        try:
            batch = next(it)
        except StopIteration:
            break
        inputs, targets, mask = batch[:3]
        inputs  = inputs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        mask    = mask.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            outputs, logits = model(inputs, PRED_LENGTH)
            loss = criterion_sl1(outputs, targets) * criterion_mae_lambda + criterion_fl(logits, mask) * criterion_mae_lambda
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        counted += 1
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return counted / dt if dt > 0 else 0.0

def benchmark_val(loader, model, device, warmup=10, measure=50):
    model.eval()
    it = iter(loader)
    with torch.no_grad():
        for _ in range(warmup):
            batch = next(it)
            inputs = batch[0].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with autocast_ctx():
                _ = model(inputs, PRED_LENGTH)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        counted = 0
        for _ in range(measure):
            try:
                batch = next(it)
            except StopIteration:
                break
            inputs = batch[0].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with autocast_ctx():
                _ = model(inputs, PRED_LENGTH)
            counted += 1
        if torch.cuda.is_available(): torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return counted / dt if dt > 0 else 0.0

# =========================================
# Metriche & Train/Eval
# =========================================
criterion_sl1 = nn.SmoothL1Loss()
criterion_mae = nn.L1Loss()
criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=1, win_size=5)
criterion_fl  = partial(vops.sigmoid_focal_loss, reduction='mean')
criterion_mae_lambda = 10

def calculate_metrics(preds, targets, logits=None, mask=None, threshold_dbz=15):
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    sl1 = F.smooth_l1_loss(preds, targets)
    mae = F.l1_loss(preds, targets)
    fl = 0
    if (logits is not None) and (mask is not None):
        fl = criterion_fl(logits, mask)

    total = sl1 + fl

    p = preds.numpy().squeeze()
    t = targets.numpy().squeeze()
    p = p * 0.5 + 0.5
    t = t * 0.5 + 0.5
    p_dbz = np.clip(p * 70.0, 0, 70)
    t_dbz = np.clip(t * 70.0, 0, 70)

    ssim_values = []
    for b in range(p_dbz.shape[0]):
        for tt in range(p_dbz.shape[1]):
            dr = max(t_dbz[b, tt].max() - t_dbz[b, tt].min(), 1e-6)
            if np.std(t_dbz[b, tt]) < 1e-6 or np.std(p_dbz[b, tt]) < 1e-6:
                ssim_values.append(1.0)
            else:
                ssim_values.append(ssim(p_dbz[b, tt], t_dbz[b, tt], data_range=dr, win_size=5, multichannel=False))
    ssim_val = float(np.mean(ssim_values)) if ssim_values else 0.0

    p_bin = (p_dbz > threshold_dbz).astype(np.uint8)
    t_bin = (t_dbz > threshold_dbz).astype(np.uint8)
    cm = confusion_matrix(t_bin.flatten(), p_bin.flatten(), labels=[0,1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    csi = tp / (tp + fp + fn + 1e-10)

    return {'TOTAL': total, 'SmoothL1': sl1, 'MAE': mae, 'FL': fl, 'SSIM': ssim_val, 'CSI': csi}

def train_epoch(model, loader, optimizer, device, scaler=None, log_window=200):
    model.train()
    total_loss = 0.0
    t_last = time.time()
    for step, (inputs, targets, mask) in enumerate(loader, start=1):
        inputs  = inputs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        mask    = mask.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            outputs, logits = model(inputs, PRED_LENGTH)
            loss = criterion_sl1(outputs, targets) * criterion_mae_lambda
            loss += criterion_fl(logits, mask) * criterion_mae_lambda

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())

        # ETA/bps "reale" a finestra
        if step % log_window == 0:
            now = time.time()
            bps = log_window / (now - t_last + 1e-9)
            eta = (len(loader) - step) / max(bps, 1e-6)
            print(f"[Train] step {step}/{len(loader)} | {bps:.2f} batch/s | ETA {eta/3600:.2f} h")
            t_last = now

    return total_loss / max(1, len(loader))

def evaluate(model, loader, device):
    model.eval()
    agg = {'MAE':0.0,'SSIM':0.0,'CSI':0.0,'SmoothL1':0.0,'FL':0.0,'TOTAL':0.0}
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                inputs, targets, mask, _ = batch
            else:
                inputs, targets, mask = batch
            inputs  = inputs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            mask    = mask.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with autocast_ctx():
                outputs, logits = model(inputs, PRED_LENGTH)
            m = calculate_metrics(outputs, targets, logits, mask)
            for k in agg: agg[k] += float(m[k])
    for k in agg: agg[k] /= float(max(1, len(loader)))
    torch.cuda.empty_cache()
    return agg

# =========================================
# Salvataggi preview dalla VAL
# =========================================
def save_all_val_predictions(model, val_loader, device, out_root, epoch, overwrite=False):
    model.eval()
    ep_dir = os.path.join(out_root, f"epoch_{epoch:03d}")
    out_pred = os.path.join(ep_dir, "predictions")
    os.makedirs(out_pred, exist_ok=True)

    saved = set()

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) != 4:
                raise RuntimeError("Validation loader deve restituire (inputs, targets, mask, paths).")

            inputs, targets, mask, paths = batch
            if not paths or len(paths) == 0:
                raise RuntimeError("Mancano i target_paths nella validation: disabilitato il fallback frame_XX.")

            inputs  = inputs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with autocast_ctx():
                outputs, _ = model(inputs, PRED_LENGTH)  # (1, T, 1, H, W)

            preds = outputs.detach().cpu().numpy()
            preds = preds * 0.5 + 0.5
            if preds.ndim == 5:
                preds = preds[0, :, 0]  # (T, H, W)

            T = preds.shape[0]
            if len(paths) < T:
                raise RuntimeError(f"I target_paths ({len(paths)}) sono meno dei frame predetti ({T}).")
            if len(paths) > T:
                paths = list(paths)[:T]

            for t in range(T):
                stem = os.path.splitext(os.path.basename(paths[t]))[0]
                key = stem.lower()
                if key in saved and not overwrite:
                    continue
                saved.add(key)

                out_path = os.path.join(out_pred, f"{stem}_pred.tiff")
                if (not overwrite) and os.path.exists(out_path):
                    continue

                frame = (preds[t] * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(frame).save(out_path)

def save_all_val_targets(val_loader, out_root, epoch, overwrite=False):
    ep_dir = os.path.join(out_root, f"epoch_{epoch:03d}")
    out_targ = os.path.join(ep_dir, "targets")
    os.makedirs(out_targ, exist_ok=True)

    saved = set()
    for batch in val_loader:
        if len(batch) != 4:
            raise RuntimeError("Validation loader deve restituire (inputs, targets, mask, paths).")

        _, targets, _, paths = batch
        if not paths or len(paths) == 0:
            raise RuntimeError("Mancano i target_paths nella validation: disabilitato il fallback frame_XX.")

        targs = targets.detach().cpu().numpy()
        targs = targs * 0.5 + 0.5
        if targs.ndim == 5:
            targs = targs[0, :, 0]  # (T, H, W)

        T = targs.shape[0]
        if len(paths) < T:
            raise RuntimeError(f"I target_paths ({len(paths)}) sono meno dei frame target ({T}).")
        if len(paths) > T:
            paths = list(paths)[:T]

        for t in range(T):
            stem = os.path.splitext(os.path.basename(paths[t]))[0]
            key = stem.lower()
            if key in saved and not overwrite:
                continue
            saved.add(key)

            out_path = os.path.join(out_targ, f"{stem}_target.tiff")
            if (not overwrite) and os.path.exists(out_path):
                continue

            frame = (targs[t] * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(frame).save(out_path)

# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    DATA_PATH = os.path.abspath("/home/v.bucciero/data/instruments/rdr0_splits/")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    train_writer = SummaryWriter(log_dir=os.path.join(log_dir, "Train"))
    val_writer   = SummaryWriter(log_dir=os.path.join(log_dir, "Validation"))

    # Model
    model = RainPredRNN(input_dim=1, num_hidden=256, max_hidden_channels=128,
                        patch_height=16, patch_width=16)
    model = model.to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    if TORCH_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("[Compile] torch.compile attivato.")
        except Exception as e:
            print(f"[Compile] torch.compile disabilitato: {e}")

    # Autotune batch size + dataloaders
    bs_guess = INIT_BATCH_SIZE
    try:
        bs, train_loader, val_loader = autotune_batch_size(model, DATA_PATH, init_bs=bs_guess, num_workers=MAX_NUM_WORKERS)
    except Exception as e:
        print(f"[AutoBS] Fallito autotuning ({e}), uso batch size 4 come fallback.")
        bs = 4
        train_loader, val_loader = create_dataloaders(DATA_PATH, batch_size=bs, num_workers=MAX_NUM_WORKERS)
    print(f"[Config] Batch size finale: {bs} | Workers: {MAX_NUM_WORKERS}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda" and not USE_BF16))

    # Benchmark (più realistico)
    steps_train = len(train_loader)
    steps_val = len(val_loader)
    bps_train = benchmark_train(train_loader, model, optimizer, DEVICE, scaler=scaler, warmup=BENCH_WARMUP, measure=BENCH_MEASURE)
    bps_val   = benchmark_val(val_loader,   model, DEVICE, warmup=BENCH_WARMUP, measure=BENCH_MEASURE)
    eta_train = steps_train / bps_train if bps_train > 0 else float('inf')
    eta_val   = steps_val   / bps_val   if bps_val   > 0 else float('inf')
    print(f"[Benchmark] Train: ~{bps_train:.2f} batch/s | steps/epoch={steps_train} | ETA epoca ≈ {_hms(eta_train)}")
    print(f"[Benchmark] Val  : ~{bps_val:.2f} batch/s | steps/val  ={steps_val}   | ETA val   ≈ {_hms(eta_val)}")

    VAL_PREVIEW_ROOT = "/home/v.bucciero/data/instruments/rdr0_previews_h100gpu"
    os.makedirs(VAL_PREVIEW_ROOT, exist_ok=True)

    best_val = float('inf')
    CHECKPOINT_DIR = "checkpoints"; os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, scaler=scaler, log_window=200)
        val_metrics = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_metrics['TOTAL'])

        print(f"\tTrain Loss: {train_loss:.4f}")
        print("\tVal " + ", ".join([f"{k}: {float(v):.4f}" for k,v in val_metrics.items()]))

        train_writer.add_scalar("Loss", train_loss, epoch)
        for k, v in val_metrics.items():
            tag = "Loss" if k.lower() == "total" else k
            val_writer.add_scalar(tag, float(v), epoch)

        save_all_val_predictions(model, val_loader, DEVICE, VAL_PREVIEW_ROOT, epoch, overwrite=False)
        save_all_val_targets(val_loader, VAL_PREVIEW_ROOT, epoch, overwrite=False)

        if val_metrics['TOTAL'] < best_val:
            best_val = val_metrics['TOTAL']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))

    train_writer.close()
    val_writer.close()
    print("Training concluso.")
