import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast  # Per mixed-precision
import rasterio
from rasterio.errors import RasterioIOError
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix
from pytorch_msssim import SSIM
import torchio as tio
from PIL import Image
import datetime
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import glob

# === Configurazione multi-GPU ===
def get_device():
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Configurazione base ===
DEVICE = get_device()
NUM_WORKERS = 8
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
INPUT_LENGTH = 6
PRED_LENGTH = 6
LAMBDA_DECOUPLE = 0.001

# Normalizzazione delle immagini
def normalize_image(img):
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0, None)
    img = 10 * np.log1p(img + 1e-8)
    min_dbz, max_dbz = 0, 70
    img = np.clip(img, min_dbz, max_dbz)
    img = (img - min_dbz) / (max_dbz - min_dbz)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return img

# === Inizializzazione seed ===
def set_seed(seed=15):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

set_seed()

# === Dataset ===
def get_augmentation_transforms():
    return tio.Compose([
        tio.RandomFlip(axes=(0, 1), p=0.5),  # Flip casuale lungo gli assi x e y
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10, p=0.5),  # Rotazioni e zoom casuali
    ])

class RadarDataset(Dataset):
    def __init__(self, data_path, input_length=6, pred_length=6, is_train=True):
        self.input_length = input_length
        self.pred_length = pred_length
        self.seq_length = input_length + pred_length
        self.files = sorted(glob.glob(os.path.join(data_path, '**/*.tiff'), recursive=True))
        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            #transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.file_validity = {}
        self.valid_indices = []
        self.total_possible_windows = max(0, len(self.files) - self.seq_length + 1)
        
        for start_idx in range(self.total_possible_windows):
            window_valid = True
            for i in range(self.seq_length):
                file = self.files[start_idx + i]
                if file not in self.file_validity:
                    try:
                        with rasterio.open(file) as src:
                            valid = src.count > 0
                    except RasterioIOError:
                        valid = False
                        print(f"File non valido: {file}")
                    self.file_validity[file] = valid
                
                if not self.file_validity[file]:
                    window_valid = False
                    break
            if window_valid:
                self.valid_indices.append(start_idx)
        
        self.total_files = len(self.files)
        self.invalid_files = sum(1 for valid in self.file_validity.values() if not valid)
        self.valid_windows = len(self.valid_indices)
        self.invalid_windows = self.total_possible_windows - self.valid_windows
        
        print(f"\nStatistiche Dataset:")
        print(f"1. File totali: {self.total_files}")
        print(f"2. File non validi: {self.invalid_files}")
        print(f"3. Finestre totali possibili: {self.total_possible_windows}")
        print(f"4. Finestre valide: {self.valid_windows}")
        print(f"5. Finestre non valide: {self.invalid_windows}")
        print(" ===================================================== \n")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        images = []
        for i in range(self.seq_length):
            file = self.files[start + i]
            with rasterio.open(file) as src:
                img = src.read(1).astype(np.float32)
                img = normalize_image(img)
                img = Image.fromarray(img)
                img = self.transform(img)
                images.append(img)
        
        inputs = torch.stack(images[:self.input_length])
        targets = torch.stack(images[self.input_length:])

        return inputs, targets

# === UNet Encoder e Decoder (invariati) ===
class UNet_Encoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),  # Modificato da 64 a 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),  # Modificato da 64 a 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),  # Modificato da 128 a 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),  # Modificato da 128 a 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip1 = x
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        skip2 = x
        return x, skip1, skip2

class UNet_Decoder(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),  # Modificato da 256 a 64 e da 128 a 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),  # Modificato da 128 a 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)  # Modificato da 64 a 32
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),  # Modificato da 128 a 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),  # Modificato da 64 a 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=1)  # Nessuna modifica necessaria qui

    def forward(self, x, skip1, skip2):
        x = torch.cat([x, skip2], dim=1)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.final_conv(x)
        return x

# === Nuovo modulo: Transformer per la modellazione temporale ===
def generate_positional_encoding(seq_len, d_model, device):
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(1)  # shape: (seq_len, 1, d_model)

class TemporalTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, pred_length):
        super().__init__()
        self.d_model = d_model
        self.pred_length = pred_length
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True  # Aggiunto batch_first=True
        )
        # In questo blocco, la sequenza temporale verrà elaborata per ogni posizione spaziale
        # Pertanto, ogni token ha dimensione d_model (uguale a num_hidden, cioè 128)
        
    def forward(self, input_sequence):
        # input_sequence: (B, T_in, C, H, W) dove C == d_model
        B, T, C, H, W = input_sequence.size()
        # Riorganizza in modo da applicare la Transformer lungo la dimensione temporale per ogni posizione spaziale
        # Portiamo le dimensioni spaziali all'esterno: (B, H, W, T, C)
        x = input_sequence.permute(0, 3, 4, 1, 2)  # (B, H, W, T, C)
        B, H, W, T, C = x.size()
        N = H * W
        x = x.reshape(B * N, T, C)  # (B*N, T, C)
        # Transformer richiede shape (T, batch, C)
        x = x.transpose(0, 1)  # (T, B*N, C)
        
        # Aggiungi positional encoding all'encoder
        pe_enc = generate_positional_encoding(T, C, x.device)  # (T, 1, C)
        encoder_input = x + pe_enc
        
        # Encoder del Transformer
        memory = self.transformer.encoder(encoder_input)
        
        # # Prepara il decoder con target inizializzati a zero per pred_length passi
        # tgt = torch.zeros(self.pred_length, B * N, C, device=x.device)
        # pe_dec = generate_positional_encoding(self.pred_length, C, x.device)  # (pred_length, 1, C)
        # tgt = tgt + pe_dec
        
        # # Decoder del Transformer: predice la sequenza futura
        # out = self.transformer.decoder(tgt, memory)  # (pred_length, B*N, C)
        
        # Ritorna alla forma originale
        out = memory
        out = out.transpose(0, 1)  # (B*N, pred_length, C)
        out = out.reshape(B, H, W, self.pred_length, C)  # (B, H, W, pred_length, C)
        out = out.permute(0, 3, 4, 1, 2)  # (B, pred_length, C, H, W)
        return out

# === Modello principale: RainPredRNN modificato per usare il Transformer ===
class RainPredRNN(nn.Module):
    def __init__(self, input_dim=1, num_hidden=32, num_layers=3, filter_size=3):
        super().__init__()
        self.encoder = UNet_Encoder(input_dim)
        self.decoder = UNet_Decoder(input_dim)
        self.num_hidden = num_hidden  # d_model per il Transformer
        # Inizializza il blocco Transformer: qui i parametri (nhead, num_layers) sono impostabili
        self.transformer_block = TemporalTransformerBlock(
            d_model=num_hidden, nhead=8, num_encoder_layers=3, num_decoder_layers=3, pred_length=PRED_LENGTH
        )

    def forward(self, input_sequence, pred_length, teacher_forcing=False):
        batch_size, seq_len, _, h, w = input_sequence.size()
        device = input_sequence.device
        
        encoder_features = []
        skip1_last, skip2_last = None, None
        # Estrai le feature per ogni frame di input usando l'encoder
        for t in range(seq_len):
            x, skip1, skip2 = self.encoder(input_sequence[:, t])
            encoder_features.append(x)  # x ha forma (B, num_hidden, h/2, w/2)
            skip1_last, skip2_last = skip1, skip2  # utilizziamo i salti dell'ultimo frame
        encoder_features = torch.stack(encoder_features, dim=1)  # (B, T_in, num_hidden, H, W)
        
        # Il blocco Transformer predice le feature future a partire dalla sequenza in input
        pred_features = self.transformer_block(encoder_features)  # (B, pred_length, num_hidden, H, W)
        
        predictions = []
        # Decodifica ciascun frame predetto usando i salti dall'ultimo frame in input
        for t in range(pred_length):
            pred_frame = self.decoder(pred_features[:, t], skip1_last, skip2_last)
            predictions.append(pred_frame)
        predictions = torch.stack(predictions, dim=1)
        
        # Non viene usata più la decouple_loss, per mantenere compatibilità restituiamo 0.0
        decouple_loss = torch.tensor(0.0, device=device)
        return predictions, decouple_loss

# === DataLoaders ===
def create_dataloaders(data_path, batch_size=4, num_workers=4):
    train_dataset = RadarDataset(os.path.join(data_path, 'train'), is_train=True)
    val_dataset = RadarDataset(os.path.join(data_path, 'val'), is_train=False)
    test_dataset = RadarDataset(os.path.join(data_path, 'test'), is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

# === Metriche ===
def calculate_metrics(preds, targets, threshold_dbz=15):
    preds = preds.cpu().numpy().squeeze()
    targets = targets.cpu().numpy().squeeze()

    preds = preds * 0.5 + 0.5
    targets = targets * 0.5 + 0.5

    targets_dbz = np.clip(targets * 70.0, 0, 70)
    preds_dbz = np.clip(preds * 70.0, 0, 70)

    if np.isnan(preds_dbz).any() or np.isnan(targets_dbz).any():
        print("Attenzione: NaN trovati nelle immagini predette o nei target!")
        preds_dbz = np.nan_to_num(preds_dbz, nan=0.0, posinf=0.0, neginf=0.0)
        targets_dbz = np.nan_to_num(targets_dbz, nan=0.0, posinf=0.0, neginf=0.0)

    mae = np.mean(np.abs(preds_dbz - targets_dbz))
    mse = np.mean((preds_dbz - targets_dbz) ** 2)
    ssim_values = []
    for b in range(preds_dbz.shape[0]):
        for t in range(preds_dbz.shape[1]):
            data_range = max(targets_dbz[b, t].max() - targets_dbz[b, t].min(), 1e-6)
            if np.std(targets_dbz[b, t]) < 1e-6 or np.std(preds_dbz[b, t]) < 1e-6:
                ssim_t = 1.0
            else:
                ssim_t = ssim(preds_dbz[b, t], targets_dbz[b, t], data_range=data_range, win_size=5, multichannel=False)
            ssim_values.append(ssim_t)
    ssim_val = np.mean(ssim_values) if ssim_values else 0.0

    preds_bin = (preds_dbz > threshold_dbz).astype(np.uint8)
    targets_bin = (targets_dbz > threshold_dbz).astype(np.uint8)
    cm = confusion_matrix(targets_bin.flatten(), preds_bin.flatten(), labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    csi = tp / (tp + fp + fn + 1e-10)
    return {
        'MAE': mae,
        'MSE': mse,
        'SSIM': ssim_val,
        'CSI': csi
    }

# === Training loop ===
def train_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            outputs, decouple_loss = model(inputs, PRED_LENGTH, teacher_forcing=True)
            decouple_loss = decouple_loss.mean() if isinstance(decouple_loss, torch.Tensor) else torch.tensor(decouple_loss, device=device)
            loss = criterion_mse(outputs, targets) + LAMBDA_DECOUPLE * (decouple_loss / (INPUT_LENGTH + PRED_LENGTH))
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        torch.cuda.empty_cache()
    return total_loss / len(loader)

# === Valutazione ===
def evaluate(model, loader, device):
    model.eval()
    metrics = {'MAE': 0, 'MSE': 0, 'SSIM': 0, 'CSI': 0}
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs, PRED_LENGTH, teacher_forcing=True)
            batch_metrics = calculate_metrics(outputs, targets)
            for k in metrics:
                metrics[k] += batch_metrics[k]
    for k in metrics:
        metrics[k] /= len(loader)
    torch.cuda.empty_cache()
    return metrics

def load_images(image_paths):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    images = []
    for path in image_paths:
        img = Image.open(path).convert("L")
        img = transform(img)
        img = img.unsqueeze(0)
        images.append(img)
    images = torch.stack(images, dim=0)
    images = images.unsqueeze(0)
    return images.to(DEVICE)

# === Salvataggio predizioni ===
def save_predictions(predictions, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    preds = predictions.detach().cpu().numpy()
    if preds.ndim == 5:
        preds = preds.squeeze(2)
    for batch_idx, seq in enumerate(preds):
        for t in range(seq.shape[0]):
            frame = (seq[t] * 70.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(frame)
            filename = os.path.join(output_dir, f"pred_{batch_idx:04d}_t{t+1}.tiff")
            img.save(filename)

def save_predictions_single_test(predictions, output_dir="outputs/custom_test"):
    os.makedirs(output_dir, exist_ok=True)
    preds = predictions.detach().cpu().numpy()
    if preds.ndim == 5:
        preds = preds.squeeze(2)
    for t in range(preds.shape[1]):
        frame = (preds[0, t] * 70.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(frame)
        filename = os.path.join(output_dir, f"pred_t{t+1}.tiff")
        img.save(filename)

# === Inizializzazione modello ===
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("/home/f.demicco/RainPredRNN2/runs/", timestamp)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

torch.cuda.empty_cache()
model = RainPredRNN(input_dim=1, num_hidden=128, num_layers=3, filter_size=3)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=1, win_size=5)
scaler = torch.amp.GradScaler('cuda')

# === Main ===
if __name__ == "__main__":
    DATA_PATH = "/home/f.demicco/RainPredRNN2/dataset"
    CHECKPOINT_DIR = "/home/f.demicco/RainPredRNN2/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    train_loader, val_loader, test_loader = create_dataloaders(DATA_PATH, BATCH_SIZE, NUM_WORKERS)
    
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_metrics['MAE'])
        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\tVal MAE: {val_metrics['MAE']:.4f}, SSIM: {val_metrics['SSIM']:.4f}, CSI: {val_metrics['CSI']:.4f}")

        # writer.add_scalar("Train-Loss", train_loss, epoch)
        # writer.add_scalar("Validation-Loss", val_metrics['MSE'], epoch)

        writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_metrics['MSE']}, epoch)
        
        if val_metrics['MAE'] < best_val_loss:
            best_val_loss = val_metrics['MAE']
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    
    state_dict = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    test_metrics = evaluate(model, test_loader, DEVICE)
    print("Test Results:")
    print(f"\tMAE: {test_metrics['MAE']:.4f}")
    print(f"\tSSIM: {test_metrics['SSIM']:.4f}")
    print(f"\tCSI: {test_metrics['CSI']:.4f}")

    os.makedirs("/home/f.demicco/RainPredRNN2/test_predictions", exist_ok=True)
    model.eval()
    with torch.no_grad():

        # inputs, targets = next(iter(test_loader))  
        # inputs = inputs.to(DEVICE)
        # outputs, _ = model(inputs, PRED_LENGTH)
        # writer.add_images("Input-Sequence", inputs[0])  
        # writer.add_images("targets-Sequence", targets[0])  
        # writer.add_images("Output-Sequence", outputs[0]) 
        # save_predictions(outputs, "/home/f.demicco/RainPredRNN2/test_predictions/batch_0000")

        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            outputs, _ = model(inputs, PRED_LENGTH)
            writer.add_images("Input-Sequence", inputs[0])  # (T, C, H, W)
            writer.add_images("Output-Sequence", outputs[0])
            save_predictions(outputs, f"/home/f.demicco/RainPredRNN2/test_predictions/batch_{i:04d}")

    print("Predizioni salvate correttamente")
    writer.close()
