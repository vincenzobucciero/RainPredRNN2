import os
import math
import glob
import datetime
import time
import numpy as np
from functools import partial
from PIL import Image

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


# ===============================
# Config
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1
PRED_LENGTH = 6
pin_memory = True
USE_AMP = True  # mixed precision su GPU moderne

# ===============================
# Utils / Seed
# ===============================
def set_seed(seed=15):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
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
    """Check veloce: prova a leggere UNA riga dalla banda 1."""
    try:
        with rasterio.open(path) as src:
            _ = src.read(1, window=((0, 1), (0, src.width)))
        return True
    except Exception:
        return False


# ===============================
# Dataset
# ===============================
class RadarDataset(Dataset):
    def __init__(self, data_path, input_length=18, pred_length=6, is_train=True,
                 generate_mask=True, return_paths=False, min_size=1024):
        self.input_length = input_length
        self.pred_length = pred_length
        self.seq_length = input_length + pred_length
        self.min_size = min_size

        # .tif + .tiff
        self.files = []
        self.files += glob.glob(os.path.join(data_path, '**/*.tif'), recursive=True)
        self.files += glob.glob(os.path.join(data_path, '**/*.tiff'), recursive=True)
        self.files = sorted(self.files)

        if len(self.files) == 0:
            raise RuntimeError(f"Nessun file trovato in {data_path}. Controlla path/permessi/estensioni.")
        if len(self.files) < self.seq_length:
            raise RuntimeError(f"Troppi pochi file in {data_path}: {len(self.files)} < seq_length={self.seq_length}")

        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.file_validity = {}
        self.valid_indices = []
        self.total_possible_windows = max(0, len(self.files) - self.seq_length + 1)
        self.augmentation_transforms = get_augmentation_transforms() if is_train else None

        # prescreen finestre valide - read-based (leggi solo una riga)
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
                            # log minimale per capire quali file saltano
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
        # retry/skip guardrail per file corrotti incontrati a runtime
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
                    # marchia non valido e resample indice
                    self.file_validity[f] = False
                    failed = True
                    # resample rapido: randomizza in train, sequenziale in val
                    if self.is_train:
                        idx = (idx + np.random.randint(1, 64)) % len(self.valid_indices)
                    else:
                        idx = (idx + 1) % len(self.valid_indices)
                    attempts += 1
                    break

                img = normalize_image(img)
                # pipeline PIL + torchvision (semplice e sicura)
                img = Image.fromarray((img * 255.0).astype(np.uint8))
                img = self.transform(img)
                images.append(img)
                if i >= self.input_length:
                    target_frame_paths.append(f)

            if failed:
                continue

            # stack e augmentations
            all_frames = torch.stack(images)                  # (seq, C, H, W)
            all_frames = all_frames.permute(1, 0, 2, 3)       # (C, seq, H, W)

            if self.is_train and self.augmentation_transforms is not None:
                all_frames = self.augmentation_transforms(all_frames)

            if not isinstance(all_frames, torch.Tensor):
                all_frames = torch.as_tensor(all_frames)

            inputs  = all_frames[:, :self.input_length]       # (C, Tin, H, W)
            targets = all_frames[:, self.input_length:]       # (C, Tout, H, W)

            inputs  = inputs.permute(1, 0, 2, 3)              # (Tin, C, H, W)
            targets = targets.permute(1, 0, 2, 3)             # (Tout, C, H, W)

            mask = None
            if self.generate_mask:
                mask = torch.where(targets > -1.0, 1.0, 0.0)

            if self.return_paths:
                # Guardrail: i path dei target DEVONO avere la stessa lunghezza di pred_length
                if len(target_frame_paths) != self.pred_length:
                    raise RuntimeError(
                        f"target_frame_paths len={len(target_frame_paths)} diversa da pred_length={self.pred_length} "
                        f"per indice finestra {start}. Controlla i file nella sequenza."
                    )
                return inputs, targets, mask, target_frame_paths
            return inputs, targets, mask

        # se proprio non riusciamo dopo N tentativi
        raise RasterioIOError(f"Nessuna finestra valida dopo {max_resamples} tentativi; indice iniziale {base_idx}.")


# ===============================
# Modello (encoder/decoder + “transformer” temporale)
# ===============================
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

        # 1) embedding a token per patch-tempo
        x = self.to_patch_embedding(x)  # (B, Tin*ph*pw, d)
        B, T, D = x.shape

        # 2) positional encoding + encoder
        pe = generate_positional_encoding(T, D, x.device)
        mem = self.encoder(x + pe)  # (B, Tin*ph*pw, d)

        # 3) prendi SOLO gli ultimi pred_length*ph*pw token
        tokens_per_frame = ph * pw
        needed = self.pred_length * tokens_per_frame
        assert needed <= T, f"pred_length ({self.pred_length}) * ph*pw ({tokens_per_frame}) > Tin*ph*pw ({T})"
        mem = mem[:, -needed:, :]  # (B, pred_length*ph*pw, d)

        # 4) proietta e rimappa a feature map
        out = self.to_feature_map(mem)  # (B, pred_length*ph*pw, p1*p2*C)
        out = rearrange(
            out,
            'b (t h w) (p1 p2 c) -> b t c (h p1) (w p2)',
            t=self.pred_length, h=ph, w=pw,
            p1=self.patch_height, p2=self.patch_width
        )  # (B, Tout, C, H, W)
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
        # ATT: valido finché pred_length <= Tin
        for t in range(pred_length):
            y, y_no = self.decoder(pred_feats[:, t], enc_feats[:, t], skip1[:, t])
            preds.append(y); preds_noact.append(y_no)
        return torch.stack(preds, dim=1), torch.stack(preds_noact, dim=1)


# ===============================
# Collate Fn (VALIDAZIONE)
# ===============================
def collate_val(batch):
    """
    batch_size=1 → batch è una lista con un solo elemento, cioè la tupla
    (inputs, targets, mask, paths). Ripristiniamo la batch dimension.
    """
    item = batch[0]
    if len(item) != 4:
        raise RuntimeError("Validation loader deve restituire (inputs, targets, mask, paths).")
    inputs, targets, mask, paths = item  # tensors 4D: (Tin, C, H, W)

    # --- normalizza paths a lista piatta di stringhe
    if isinstance(paths, (list, tuple)) and paths and isinstance(paths[0], (list, tuple)):
        paths = list(paths[0])
    elif isinstance(paths, (list, tuple)):
        paths = list(paths)
    else:
        paths = [paths]

    # --- aggiungi batch dimension per tutti i tensori
    if inputs.dim() == 4:
        inputs = inputs.unsqueeze(0)   # (1, Tin, C, H, W)
    if targets.dim() == 4:
        targets = targets.unsqueeze(0) # (1, Tout, C, H, W)
    if mask is not None and mask.dim() == 4:
        mask = mask.unsqueeze(0)       # (1, Tout, C, H, W)

    return inputs, targets, mask, paths



# ===============================
# DataLoaders (solo train/val)
# ===============================
def create_dataloaders(data_path, batch_size=4, num_workers=4):
    train_dataset = RadarDataset(os.path.join(data_path, 'train'), is_train=True, pred_length=PRED_LENGTH)
    val_dataset   = RadarDataset(os.path.join(data_path, 'val'),   is_train=False, return_paths=True, pred_length=PRED_LENGTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader   = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_val,   # <--- FIX
    )
    return train_loader, val_loader


# ===== Benchmark =====
def _hms(sec: float):
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def benchmark_train(loader, model, optimizer, device, scaler=None, warmup=2, measure=10):
    """Misura batch/s includendo forward+loss+backward+step (train)."""
    model.train()
    it = iter(loader)
    # warmup
    for _ in range(warmup):
        batch = next(it)
        inputs, targets, mask = batch[:3]
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs, logits = model(inputs, PRED_LENGTH)
                loss = criterion_sl1(outputs, targets) * criterion_mae_lambda + criterion_fl(logits, mask) * criterion_mae_lambda
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, logits = model(inputs, PRED_LENGTH)
            loss = criterion_sl1(outputs, targets) * criterion_mae_lambda + criterion_fl(logits, mask) * criterion_mae_lambda
            loss.backward()
            optimizer.step()

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    counted = 0
    for _ in range(measure):
        try:
            batch = next(it)
        except StopIteration:
            break
        inputs, targets, mask = batch[:3]
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs, logits = model(inputs, PRED_LENGTH)
                loss = criterion_sl1(outputs, targets) * criterion_mae_lambda + criterion_fl(logits, mask) * criterion_mae_lambda
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, logits = model(inputs, PRED_LENGTH)
            loss = criterion_sl1(outputs, targets) * criterion_mae_lambda + criterion_fl(logits, mask) * criterion_mae_lambda
            loss.backward()
            optimizer.step()
        counted += 1
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return counted / dt if dt > 0 else 0.0

def benchmark_val(loader, model, device, warmup=2, measure=20):
    """Misura batch/s della sola forward (validation)."""
    model.eval()
    it = iter(loader)
    with torch.no_grad():
        for _ in range(warmup):
            batch = next(it)
            inputs = batch[0].to(device, non_blocking=True)
            _ = model(inputs, PRED_LENGTH)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        counted = 0
        for _ in range(measure):
            try:
                batch = next(it)
            except StopIteration:
                break
            inputs = batch[0].to(device, non_blocking=True)
            _ = model(inputs, PRED_LENGTH)
            counted += 1
        if torch.cuda.is_available(): torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return counted / dt if dt > 0 else 0.0
# ===== Fine benchmark =====


# ===============================
# Metriche & Train/Eval
# ===============================
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

def train_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    for inputs, targets, mask in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs, logits = model(inputs, PRED_LENGTH)
                loss = criterion_sl1(outputs, targets) * criterion_mae_lambda
                loss += criterion_fl(logits, mask) * criterion_mae_lambda
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, logits = model(inputs, PRED_LENGTH)
            loss = criterion_sl1(outputs, targets) * criterion_mae_lambda
            loss += criterion_fl(logits, mask) * criterion_mae_lambda
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
    
    return total_loss / max(1, len(loader))

def generate_evaluation_report(metrics_dict, conf_matrix, output_dir):
    """Generate a detailed evaluation report with metrics and confusion matrix"""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=== RainPredRNN Evaluation Report ===\n\n")
        
        # Write metrics
        f.write("Metrics:\n")
        f.write("-" * 40 + "\n")
        for metric, value in metrics_dict.items():
            f.write(f"{metric:15s}: {value:.4f}\n")
        f.write("\n")
        
        # Write confusion matrix
        f.write("Confusion Matrix:\n")
        f.write("-" * 40 + "\n")
        f.write("Format: [[TN, FP],\n        [FN, TP]]\n\n")
        f.write(str(conf_matrix))
        f.write("\n\n")
        
        # Calculate and write additional metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        
        f.write("Additional Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Precision:  {precision:.4f}\n")
        f.write(f"Recall:     {recall:.4f}\n")
        f.write(f"F1 Score:   {f1:.4f}\n")
        f.write(f"Accuracy:   {accuracy:.4f}\n")

def evaluate(model, loader, device):
    model.eval()
    agg = {'MAE':0.0,'SSIM':0.0,'CSI':0.0,'SmoothL1':0.0,'FL':0.0,'TOTAL':0.0}
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                inputs, targets, mask, _ = batch
            else:
                inputs, targets, mask = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            outputs, logits = model(inputs, PRED_LENGTH)
            m = calculate_metrics(outputs, targets, logits, mask)
            for k in agg: agg[k] += float(m[k])
            
            # Collect predictions and targets for confusion matrix
            preds = outputs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            preds = preds * 0.5 + 0.5
            targets_np = targets_np * 0.5 + 0.5
            preds_dbz = np.clip(preds * 70.0, 0, 70)
            targets_dbz = np.clip(targets_np * 70.0, 0, 70)
            preds_bin = (preds_dbz > 15).astype(np.uint8)
            targets_bin = (targets_dbz > 15).astype(np.uint8)
            all_preds.extend(preds_bin.flatten())
            all_targets.extend(targets_bin.flatten())
            
    # Calculate final metrics
    for k in agg: agg[k] /= float(max(1, len(loader)))
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds, labels=[0,1])
    
    torch.cuda.empty_cache()
    return agg, conf_matrix


# ===============================
# Salvataggi preview dalla VAL (completi, no duplicati)
# ===============================
def save_all_val_predictions(model, val_loader, device, out_root, epoch,
                             overwrite=False):
    """
    Esegue la forward su TUTTA la validation e salva una sola prediction
    per ciascun TIFF target, usando SEMPRE il nome reale (da target_paths).
    """
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

            # paths è già lista piatta grazie al collate_val
            if not paths or len(paths) == 0:
                raise RuntimeError("Mancano i target_paths nella validation: disabilitato il fallback frame_XX.")

            inputs  = inputs.to(device, non_blocking=True)
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

        # paths è già lista piatta grazie al collate_val
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


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    # Path degli split “clean” (train/val)
    DATA_PATH = os.path.abspath("/home/v.bucciero/data/instruments/rdr0_splits/")

    # logging & writers
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    train_writer = SummaryWriter(log_dir=os.path.join(log_dir, "Train"))
    val_writer   = SummaryWriter(log_dir=os.path.join(log_dir, "Validation"))

    # model & opt
    model = RainPredRNN(input_dim=1, num_hidden=256, max_hidden_channels=128,
                        patch_height=16, patch_width=16).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    # data
    train_loader, val_loader = create_dataloaders(DATA_PATH, BATCH_SIZE, NUM_WORKERS)

    # Stima tempi
    steps_train = len(train_loader)
    steps_val = len(val_loader)

    bps_train = benchmark_train(train_loader, model, optimizer, DEVICE, scaler=scaler, warmup=2, measure=10)
    bps_val   = benchmark_val(val_loader, model, DEVICE, warmup=2, measure=50)

    eta_train = steps_train / bps_train if bps_train > 0 else float('inf')
    eta_val   = steps_val   / bps_val   if bps_val   > 0 else float('inf')

    print(f"[Benchmark] Train: ~{bps_train:.2f} batch/s | steps/epoch={steps_train} | ETA epoca ≈ {_hms(eta_train)}")
    print(f"[Benchmark] Val  : ~{bps_val:.2f} batch/s | steps/val  ={steps_val}   | ETA val   ≈ {_hms(eta_val)}")

    # dove salvo le anteprime dalla validation ad ogni epoca
    VAL_PREVIEW_ROOT = "/home/v.bucciero/data/instruments/rdr0_previews_h100gpu"
    os.makedirs(VAL_PREVIEW_ROOT, exist_ok=True)

    best_val = float('inf')
    CHECKPOINT_DIR = "checkpoints"; os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, scaler=scaler)
        val_metrics, conf_matrix = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_metrics['TOTAL'])

        print(f"\tTrain Loss: {train_loss:.4f}")
        print("\tVal " + ", ".join([f"{k}: {float(v):.4f}" for k,v in val_metrics.items()]))

        train_writer.add_scalar("Loss", train_loss, epoch)
        for k, v in val_metrics.items():
            tag = "Loss" if k.lower() == "total" else k
            val_writer.add_scalar(tag, float(v), epoch)

        # salva tutte le predizioni/target della VAL con nomi reali
        save_all_val_predictions(model, val_loader, DEVICE, VAL_PREVIEW_ROOT, epoch, overwrite=False)
        save_all_val_targets(val_loader, VAL_PREVIEW_ROOT, epoch, overwrite=False)

        # checkpoint sul best validation
        if val_metrics['TOTAL'] < best_val:
            best_val = val_metrics['TOTAL']
            # Genera il report di valutazione
            generate_evaluation_report(val_metrics, conf_matrix, os.path.join(CHECKPOINT_DIR, "evaluation_reports"))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'confusion_matrix': conf_matrix
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))

    train_writer.close()
    val_writer.close()
    print("Training concluso.")
