import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast  # Per mixed-precision
import rasterio
from rasterio.errors import RasterioIOError
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix
from PIL import Image
import torchvision.transforms as transforms
import glob

# === Configurazione base ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
INPUT_LENGTH = 6
PRED_LENGTH = 6
LAMBDA_DECOUPLE = 0.1

# === Inizializzazione seed ===
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# === Dataset ===
class RadarDataset(Dataset):
    # ... (Implementazione precedente del dataset)

# === Modello ===
class SpatiotemporalLSTMCell(nn.Module):
    # ... (Implementazione precedente)
    
class PredRNN_Block(nn.Module):
    # ... (Implementazione precedente)

class UNet_Encoder(nn.Module):
    # ... (Implementazione precedente)

class UNet_Decoder(nn.Module):
    # ... (Implementazione precedente)

class RainPredRNN(nn.Module):
    # ... (Implementazione precedente)

# === Inizializzazione pesi di He ===
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# === Inizializzazione modello con DataParallel e mixed-precision ===
model = RainPredRNN(input_dim=1, num_hidden=64, num_layers=2, filter_size=3)
model.apply(init_weights)  # Inizializzazione di He
model = DataParallel(model).to(DEVICE)

# === Ottimizzatore e scheduler ===
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.MSELoss()

# === Supporto mixed-precision ===
scaler = GradScaler()  # Per l'addestramento a precisione mista

# === DataLoaders ===
def create_dataloaders(data_path):
    # ... (Implementazione precedente)

# === Metriche ===
def calculate_metrics(preds, targets, threshold=0.5):
    # ... (Implementazione precedente)

# === Training loop con mixed-precision ===
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast():  # Abilita mixed-precision
            outputs, decouple_loss = model(inputs, PRED_LENGTH)
            loss = criterion(outputs, targets) + LAMBDA_DECOUPLE * decouple_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    return total_loss / len(loader)

# === Valutazione ===
def evaluate(model, loader, device):
    # ... (Implementazione precedente)

# === Salvataggio predizioni ===
def save_predictions(predictions, output_dir="outputs"):
    # ... (Implementazione precedente)

# === Main ===
if __name__ == "__main__":
    # Configurazione percorsi
    DATA_PATH = "/percorso/dataset"
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Creazione dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(DATA_PATH)
    
    # Training con supporto mixed-precision
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_metrics['MAE'])
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val MAE: {val_metrics['MAE']:.4f}, SSIM: {val_metrics['SSIM']:.4f}, CSI: {val_metrics['CSI']:.4f}")
        
        if val_metrics['MAE'] < best_val_loss:
            best_val_loss = val_metrics['MAE']
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    
    # Test finale
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth")))
    test_metrics = evaluate(model, test_loader, DEVICE)
    print("Test Results:")
    print(f"MAE: {test_metrics['MAE']:.4f}")
    print(f"SSIM: {test_metrics['SSIM']:.4f}")
    print(f"CSI: {test_metrics['CSI']:.4f}")
    
    # Salvataggio predizioni test
    os.makedirs("./test_predictions", exist_ok=True)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            outputs, _ = model(inputs, PRED_LENGTH)
            save_predictions(outputs, f"./test_predictions/batch_{i:04d}")