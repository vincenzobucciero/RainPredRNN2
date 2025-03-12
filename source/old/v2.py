import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    def __init__(self, data_path, input_length=6, pred_length=6, is_train=True):
        self.input_length = input_length
        self.pred_length = pred_length
        self.seq_length = input_length + pred_length
        self.files = sorted(glob.glob(os.path.join(data_path, '**/*.tiff'), recursive=True))
        self.is_train = is_train
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.file_validity = {}
        self.valid_indices = []

        for start_idx in range(len(self.files) - self.seq_length + 1):
            valid = True
            for i in range(self.seq_length):
                file = self.files[start_idx + i]
                if file not in self.file_validity:
                    try:
                        with rasterio.open(file) as src:
                            valid = src.count > 0 and src.shape == (150, 150)
                    except:
                        valid = False
                    self.file_validity[file] = valid
                if not self.file_validity[file]:
                    valid = False
                    break
            if valid:
                self.valid_indices.append(start_idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        images = []
        for i in range(self.seq_length):
            file = self.files[start + i]
            with rasterio.open(file) as src:
                img = src.read(1).astype(np.float32)
                img = (img / 40.0).clip(0, 1)  # Normalizzazione
                img = Image.fromarray(img)
                img = self.transform(img)
                images.append(img)
        
        inputs = torch.stack(images[:self.input_length])
        targets = torch.stack(images[self.input_length:])
        return inputs, targets

# === Modello ===
class SpatiotemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, filter_size):
        super().__init__()
        self.num_hidden = num_hidden
        padding = filter_size // 2
        
        # Gates per cella convenzionale (C)
        self.conv_xc = nn.Conv2d(in_channel, num_hidden*4, filter_size, padding=padding)
        self.conv_hc = nn.Conv2d(num_hidden, num_hidden*4, filter_size, padding=padding)
        self.conv_cc = nn.Conv2d(num_hidden, num_hidden*3, kernel_size=1)
        
        # Gates per cella spaziotemporale (M)
        self.conv_xm = nn.Conv2d(in_channel, num_hidden*4, filter_size, padding=padding)
        self.conv_hm = nn.Conv2d(num_hidden, num_hidden*4, filter_size, padding=padding)
        self.conv_mm = nn.Conv2d(num_hidden, num_hidden*4, kernel_size=1)
        
        # Decoupling 1x1 convolutions
        self.conv_c_decouple = nn.Conv2d(num_hidden, num_hidden, kernel_size=1)
        self.conv_m_decouple = nn.Conv2d(num_hidden, num_hidden, kernel_size=1)

    def forward(self, x, h_c, c_c, m_c):
        # Processo cella C
        i_c, f_c, g_c, o_c = torch.split(
            self.conv_xc(x) + self.conv_hc(h_c),
            self.num_hidden, dim=1)
        
        c = torch.sigmoid(f_c) * c_c + torch.sigmoid(i_c) * torch.tanh(g_c)
        h_c_new = torch.sigmoid(o_c) * torch.tanh(c)
        
        # Processo cella M
        i_m, f_m, g_m, o_m = torch.split(
            self.conv_xm(x) + self.conv_hm(h_c) + self.conv_mm(m_c),
            self.num_hidden, dim=1)
        
        m = torch.sigmoid(f_m) * m_c + torch.sigmoid(i_m) * torch.tanh(g_m)
        h_m_new = torch.sigmoid(o_m) * torch.tanh(m)
        
        # Decoupling
        c_decouple = self.conv_c_decouple(c)
        m_decouple = self.conv_m_decouple(m)
        
        h_new = h_c_new + h_m_new
        return h_new, c_decouple, m_decouple

class PredRNN_Block(nn.Module):
    def __init__(self, num_layers, num_hidden, filter_size):
        super().__init__()
        self.layers = num_layers
        self.st_cells = nn.ModuleList()
        
        for i in range(num_layers):
            in_channel = num_hidden if i > 0 else 1
            self.st_cells.append(SpatiotemporalLSTMCell(in_channel, num_hidden, filter_size))

    def forward(self, inputs, hidden_states):
        new_hidden = []
        new_cell = []
        new_memory = []
        decouple_loss = 0
        
        for l in range(self.layers):
            h_prev = hidden_states[0][l]
            c_prev = hidden_states[1][l]
            m_prev = hidden_states[2][l] if l > 0 else None
            
            if l > 0:
                m_prev = new_memory[l-1]
            
            h_new, c_new, m_new = self.st_cells[l](inputs, h_prev, c_prev, m_prev)
            decouple_loss += torch.mean(torch.abs(c_new - m_new))
            
            new_hidden.append(h_new)
            new_cell.append(c_new)
            new_memory.append(m_new)
            inputs = h_new
            
        return inputs, [new_hidden, new_cell, new_memory], decouple_loss

class UNet_Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.enc1 = self.contract_block(in_channels, 64, 3, 1)
        self.enc2 = self.contract_block(64, 128, 3, 1)
        self.pool = nn.MaxPool2d(2)
        
    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x_pooled1 = self.pool(x1)
        x2 = self.enc2(x_pooled1)
        return x2, x1

class UNet_Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        
    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, skip):
        x = self.upconv1(x)
        x = torch.cat([x, skip], dim=1)
        x = self.dec1(x)
        x = self.upconv2(x)
        return torch.sigmoid(x)

class RainPredRNN(nn.Module):
    def __init__(self, input_dim=1, num_hidden=64, num_layers=2, filter_size=3):
        super().__init__()
        self.encoder = UNet_Encoder(input_dim)
        self.decoder = UNet_Decoder(input_dim)
        self.rnn_block = PredRNN_Block(num_layers, num_hidden, filter_size)
        self.num_layers = num_layers
        self.num_hidden = num_hidden

    def forward(self, input_sequence, pred_length):
        batch_size, seq_len, _, h, w = input_sequence.size()
        h_t, c_t, m_t = [], [], []
        
        for l in range(self.num_layers):
            h_t.append(torch.zeros(batch_size, self.num_hidden, h//4, w//4).to(input_sequence.device))
            c_t.append(torch.zeros(batch_size, self.num_hidden, h//4, w//4).to(input_sequence.device))
            m_t.append(torch.zeros(batch_size, self.num_hidden, h//4, w//4).to(input_sequence.device))
        
        encoder_features = []
        for t in range(seq_len):
            enc_out, skip = self.encoder(input_sequence[:, t])
            encoder_features.append((enc_out, skip))
        
        predictions = []
        total_decouple_loss = 0
        for t in range(seq_len + pred_length):
            if t < seq_len:
                x, skip = encoder_features[t]
            else:
                x = torch.zeros_like(enc_out)
                skip = torch.zeros_like(skip)

            # if t < seq_len:
            #     x, skip = encoder_features[t]  # Usa i dati reali nei primi frame
            # else:
            #     x = predictions[-1] if predictions else torch.zeros_like(enc_out)  
            #     skip = skip if predictions else torch.zeros_like(skip)
            
            rnn_out, (h_t, c_t, m_t), decouple_loss = self.rnn_block(x, [h_t, c_t, m_t])
            total_decouple_loss += decouple_loss
            
            if t >= seq_len:
                pred = self.decoder(rnn_out, skip)
                predictions.append(pred)
        
        return torch.stack(predictions, dim=1), total_decouple_loss

# === Inizializzazione modello ===
model = RainPredRNN(input_dim=1, num_hidden=64, num_layers=2, filter_size=3)
model = DataParallel(model).to(DEVICE)

# === Ottimizzatore e loss ===
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.MSELoss()

# === DataLoaders ===
def create_dataloaders(data_path):
    train_dataset = RadarDataset(os.path.join(data_path, 'train'), input_length=INPUT_LENGTH, pred_length=PRED_LENGTH)
    val_dataset = RadarDataset(os.path.join(data_path, 'val'), input_length=INPUT_LENGTH, pred_length=PRED_LENGTH, is_train=False)
    test_dataset = RadarDataset(os.path.join(data_path, 'test'), input_length=INPUT_LENGTH, pred_length=PRED_LENGTH, is_train=False)
    
    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True),
        DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
        DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    )

# === Metriche ===
def calculate_metrics(preds, targets, threshold=0.5):
    preds = preds.cpu().numpy().squeeze()
    targets = targets.cpu().numpy().squeeze()
    
    mae = np.mean(np.abs(preds - targets))
    ssim_val = ssim(preds, targets, data_range=targets.max()-targets.min())
    
    preds_bin = (preds > threshold).astype(np.uint8)
    targets_bin = (targets > threshold).astype(np.uint8)
    
    tn, fp, fn, tp = confusion_matrix(targets_bin.flatten(), preds_bin.flatten()).ravel()
    csi = tp / (tp + fp + fn + 1e-10)
    
    return {'MAE': mae, 'SSIM': ssim_val, 'CSI': csi}

# === Training loop ===
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs, decouple_loss = model(inputs, PRED_LENGTH)
        loss = criterion(outputs, targets) + LAMBDA_DECOUPLE * decouple_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

# === Valutazione ===
def evaluate(model, loader, device):
    model.eval()
    metrics = {'MAE': 0, 'SSIM': 0, 'CSI': 0}
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs, PRED_LENGTH)
            batch_metrics = calculate_metrics(outputs, targets)
            for k in metrics:
                metrics[k] += batch_metrics[k]
    for k in metrics:
        metrics[k] /= len(loader)
    return metrics

# === Salvataggio predizioni ===
def save_predictions(predictions, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    preds = predictions.squeeze().cpu().numpy()
    
    for idx, seq in enumerate(preds):
        for t in range(seq.shape[0]):
            frame = (seq[t] * 40.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(frame)
            filename = os.path.join(output_dir, f"pred_{idx:04d}_t{t+1}.tiff")
            img.save(filename)

# === Main ===
if __name__ == "__main__":
    # Configurazione percorsi
    DATA_PATH = "/percorso/dataset"
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Creazione dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(DATA_PATH)
    
    # Training
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