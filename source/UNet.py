import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import rasterio
from rasterio.errors import RasterioIOError
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix
from pytorch_msssim import SSIM
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
NUM_EPOCHS = 100
INPUT_LENGTH = 6
PRED_LENGTH = 6
RESIZE_DIM = 256

# Normalizzazione delle immagini
def normalize_image(img):
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0, None)
    img = 10 * np.log1p(img + 1e-8)
    min_dbz, max_dbz = 0, 70
    img = np.clip(img, min_dbz, max_dbz)
    img = (img - min_dbz) / (max_dbz - min_dbz)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    if np.any(np.isnan(img)) or np.any(np.isinf(img)):
        print("Errore: Dati non validi trovati nella normalizzazione.")
    return img

# === Inizializzazione seed ===
def set_seed(seed=15):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
set_seed()

# === Dataset ===
class RadarDataset(Dataset):
    def __init__(self, data_path, input_length=INPUT_LENGTH, pred_length=PRED_LENGTH, is_train=True):
        self.input_length = input_length
        self.pred_length = pred_length
        self.seq_length = input_length + pred_length
        self.files = sorted(glob.glob(os.path.join(data_path, '**/*.tiff'), recursive=True))
        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.CenterCrop((RESIZE_DIM, RESIZE_DIM)),
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

        print("\nStatistiche Dataset:")
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
                if np.any(np.isnan(img)) or np.any(np.isinf(img)):
                    print(f"Dati non validi nel file: {file}")
                img = Image.fromarray(img)
                img = self.transform(img)
                images.append(img)
        all_frames = torch.stack(images)  # Shape: (seq_length, C, H, W) -> (12, 1, 200, 200)
        # all_frames = all_frames.permute(1, 0, 2, 3)  # Shape: (1, 12, 200, 200)
        
        inputs = all_frames[:self.input_length]  # Shape: (input_length, C, H, W) -> (6, 1, 200, 200)
        targets = all_frames[self.input_length:]  # Shape: (pred_length, C, H, W) -> (6, 1, 200, 200)
    
        #inputs = inputs.permute(1, 0, 2, 3)  # Shape: (6, 1, 200, 200)
        #targets = targets.permute(1, 0, 2, 3)  # Shape: (6, 1, 200, 200)
        
        return inputs, targets

# === Encoder / Decoder ===
class RainNetEncoder(nn.Module):
    def __init__(self, input_channels=INPUT_LENGTH):
        super(RainNetEncoder, self).__init__()

        # Blocco convoluzionale 1
        self.conv1f = self._conv_block(input_channels, 64)
        self.conv1s = self._conv_block(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Blocco convoluzionale 2
        self.conv2f = self._conv_block(64, 128)
        self.conv2s = self._conv_block(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Blocco convoluzionale 3
        self.conv3f = self._conv_block(128, 256)
        self.conv3s = self._conv_block(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Blocco convoluzionale 4
        self.conv4f = self._conv_block(256, 512)
        self.conv4s = self._conv_block(512, 512)
        self.drop4 = nn.Dropout2d(0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Blocco convoluzionale 5
        self.conv5f = self._conv_block(512, 1024)
        self.conv5s = self._conv_block(1024, 1024)
        self.drop5 = nn.Dropout2d(0.5)

    def _conv_block(self, in_channels, out_channels, num_convs=2):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 5:  # Se ha una dimensione extra
            x = x.squeeze(2)
        # Encoder path
        conv1f = self.conv1f(x)
        conv1s = self.conv1s(conv1f)
        pool1 = self.pool1(conv1s)

        conv2f = self.conv2f(pool1)
        conv2s = self.conv2s(conv2f)
        pool2 = self.pool2(conv2s)

        conv3f = self.conv3f(pool2)
        conv3s = self.conv3s(conv3f)
        pool3 = self.pool3(conv3s)

        conv4f = self.conv4f(pool3)
        conv4s = self.conv4s(conv4f)
        drop4 = self.drop4(conv4s)
        pool4 = self.pool4(drop4)

        conv5f = self.conv5f(pool4)
        conv5s = self.conv5s(conv5f)
        drop5 = self.drop5(conv5s)

        return drop5, conv4s, conv3s, conv2s, conv1s
    
class RainNetDecoder(nn.Module):
    def __init__(self, output_channels=PRED_LENGTH, mode="regression"):
        super(RainNetDecoder, self).__init__()

        # Blocco upsampling 6
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = self._conv_block(1024, 512, num_convs=2)

        # Blocco upsampling 7
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = self._conv_block(512, 256, num_convs=2)

        # Blocco upsampling 8
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = self._conv_block(256, 128, num_convs=2)

        # Blocco upsampling 9
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = self._conv_block(128, 64, num_convs=2)

        # Layer finale
        self.conv_final = nn.Conv2d(64, 2, kernel_size=3, padding=1)

        if mode == "regression":
            self.output_layer = nn.Conv2d(2, output_channels, kernel_size=1)
        elif mode == "segmentation":
            self.output_layer = nn.Sequential(
                nn.Conv2d(2, output_channels, kernel_size=1),
                nn.Sigmoid()
            )

    def _conv_block(self, in_channels, out_channels, num_convs=2):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, conv4s, conv3s, conv2s, conv1s):
        # Decoder path
        up6 = self.up6(x)
        merge6 = torch.cat([up6, conv4s], dim=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        merge7 = torch.cat([up7, conv3s], dim=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        merge8 = torch.cat([up8, conv2s], dim=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        merge9 = torch.cat([up9, conv1s], dim=1)
        conv9 = self.conv9(merge9)

        conv_final = self.conv_final(conv9)
        output = self.output_layer(conv_final)

        return output

# === Modello RainNet ===
class RainNet(nn.Module):
    def __init__(self, input_shape=(INPUT_LENGTH, RESIZE_DIM, RESIZE_DIM), mode="regression"):
        super(RainNet, self).__init__()
        self.mode = mode

        # Encoder
        self.encoder = RainNetEncoder(input_channels=input_shape[0])

        # Decoder
        self.decoder = RainNetDecoder(output_channels=PRED_LENGTH, mode = mode)

    def forward(self, x):
        # Passaggio attraverso l'encoder
        encoder_output, conv4s, conv3s, conv2s, conv1s = self.encoder(x)

        # Passaggio attraverso il decoder
        output = self.decoder(encoder_output, conv4s, conv3s, conv2s, conv1s)

        return output
    
# === DataLoaders ===
def create_dataloaders(data_path, batch_size=4, num_workers=4):
    train_dataset = RadarDataset(os.path.join(data_path, 'train'), is_train=True)
    val_dataset = RadarDataset(os.path.join(data_path, 'val'), is_train=False)
    test_dataset = RadarDataset(os.path.join(data_path, 'test'), is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

# === Metriche ===
def calculate_metrics(preds, targets, threshold_dbz=15):
    preds = preds.cpu().numpy().squeeze()
    targets = targets.cpu().numpy().squeeze()
    #preds = preds * 0.5 + 0.5
    #targets = targets * 0.5 + 0.5
    targets_dbz = np.clip(targets * 70.0, 0, 70)
    preds_dbz = np.clip(preds * 70.0, 0, 70)
    mae = np.mean(np.abs(preds_dbz - targets_dbz))
    mse = np.mean((preds_dbz - targets_dbz) ** 2)
    ssim_values = []
    for b in range(preds_dbz.shape[0]):
        for t in range(preds_dbz.shape[1]):
            data_range = max(targets_dbz[b, t].max() - targets_dbz[b, t].min(), 1e-6)
            ssim_t = ssim(preds_dbz[b, t], targets_dbz[b, t], data_range=data_range, win_size=5, multichannel=False)
            ssim_values.append(ssim_t)
    ssim_val = np.mean(ssim_values) if ssim_values else 0.0
    preds_bin = (preds_dbz > threshold_dbz).astype(np.uint8)
    targets_bin = (targets_dbz > threshold_dbz).astype(np.uint8)
    cm = confusion_matrix(targets_bin.flatten(), preds_bin.flatten(), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    csi = tp / (tp + fp + fn + 1e-10)
    return {'MAE': mae, 'MSE': mse, 'SSIM': ssim_val, 'CSI': csi}

# === Training loop ===
def train_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        #print(f"Input shape: {inputs.shape}")
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            outputs = model(inputs)
            outputs = outputs.unsqueeze(2)
            #print(f"Outputs shape: {outputs.shape}")
            #print(f"Targets shape: {targets.shape}")
            loss = criterion_mse(outputs, targets)
        if scaler:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            outputs = model(inputs)
            batch_metrics = calculate_metrics(outputs, targets)
            for k in metrics:
                metrics[k] += batch_metrics[k]
    for k in metrics:
        metrics[k] /= len(loader)
    torch.cuda.empty_cache()
    return metrics

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

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("/home/f.demicco/RainPredRNN2/runs/", timestamp)
os.makedirs(log_dir, exist_ok=True)

# Crea le sottocartelle per Train e Validation
train_log_dir = os.path.join(log_dir, "Train")
val_log_dir = os.path.join(log_dir, "Validation")
os.makedirs(train_log_dir, exist_ok=True)
os.makedirs(val_log_dir, exist_ok=True)

# Inizializza i writer per Train e Validation
train_writer = SummaryWriter(log_dir=train_log_dir)
val_writer = SummaryWriter(log_dir=val_log_dir)

# Inizializzazione del modello
torch.cuda.empty_cache()
model = RainNet(input_shape=(PRED_LENGTH, RESIZE_DIM, RESIZE_DIM), mode="regression")
model = model.to(DEVICE)

torch.autograd.set_detect_anomaly(True)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
scaler = torch.amp.GradScaler('cuda', enabled=True)

# metriche
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=1, win_size=5)

# === Main ===
if __name__ == "__main__":
    DATA_PATH = "/home/f.demicco/RainPredRNN2/dataset"
    CHECKPOINT_DIR = "/home/f.demicco/RainPredRNN2/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Creazione dei dataloader
    train_loader, val_loader, test_loader = create_dataloaders(DATA_PATH, BATCH_SIZE, NUM_WORKERS)

    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, scaler)
        val_metrics = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_metrics['MAE'])
        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\tVal MAE: {val_metrics['MAE']:.4f}, SSIM: {val_metrics['SSIM']:.4f}, CSI: {val_metrics['CSI']:.4f}")

        train_writer.add_scalar("Loss", train_loss, epoch)
        val_writer.add_scalar("Loss", val_metrics['MSE'], epoch)

        if val_metrics['MAE'] < best_val_loss:
            best_val_loss = val_metrics['MAE']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))

    # Caricamento del miglior modello
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Test finale
    test_metrics = evaluate(model, test_loader, DEVICE)
    print("Test Results:")
    print(f"\tMAE: {test_metrics['MAE']:.4f}")
    print(f"\tSSIM: {test_metrics['SSIM']:.4f}")
    print(f"\tCSI: {test_metrics['CSI']:.4f}")

    # Salvataggio delle predizioni
    os.makedirs("/home/f.demicco/RainPredRNN2/test_predictions", exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            save_predictions(outputs, f"/home/f.demicco/RainPredRNN2/test_predictions/batch_{i:04d}")
    
    print("Predizioni salvate correttamente")
    train_writer.close()
    val_writer.close()