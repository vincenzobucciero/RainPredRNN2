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

# Normalizzazione delle immagini
def normalize_image(img):
    # 1. Converti in dBZ (riflettività)
    img = 10 * np.log10(img + 1e-8)  # Aggiungi epsilon per evitare log(0)
    
    # 2. Normalizza nel range fisso [0, 70] dBZ (valori tipici per radar meteorologici)
    min_dbz, max_dbz = 0, 70
    img = np.clip(img, min_dbz, max_dbz)  # Limita ai valori fisici del dataset
    img = (img - min_dbz) / (max_dbz - min_dbz)  # Scala tra 0 e 1
    
    # 3. Gestisci valori non fisici
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return img

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
                img = normalize_image(img) #img = (img / 40.0).clip(0, 1)  # Normalizzazione
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
        # ===== Calcolo gate per cella C =====
        # Equazione (2) del paper per C
        xc_comb = self.conv_xc(x)
        hc_comb = self.conv_hc(h_c)
        cc_comb = self.conv_cc(c_c)  # Per i gate f_c e i_c
        
        # Split dei gate
        i_c, f_c, g_c, o_c = torch.split(xc_comb + hc_comb, self.num_hidden, dim=1)
        cc_i, cc_f, cc_o = torch.split(cc_comb, self.num_hidden, dim=1)
        
        # Calcolo delta C (it ⊙ gt)
        delta_c = torch.sigmoid(i_c) * torch.tanh(g_c)
        
        # Aggiornamento cella C
        c_new = torch.sigmoid(f_c + cc_f) * c_c + delta_c
        
        # ===== Calcolo gate per cella M =====
        xm_comb = self.conv_xm(x)
        hm_comb = self.conv_hm(h_c)
        mm_comb = self.conv_mm(m_c)
        
        # Split dei gate
        i_m, f_m, g_m, o_m = torch.split(xm_comb + hm_comb + mm_comb, self.num_hidden, dim=1)
        
        # Calcolo delta M (it' ⊙ gt')
        delta_m = torch.sigmoid(i_m) * torch.tanh(g_m)
        
        # Aggiornamento cella M
        m_new = torch.sigmoid(f_m) * m_c + delta_m
        
        # ===== Applicazione decoupling loss =====
        delta_c_conv = self.conv_c_decouple(delta_c)
        delta_m_conv = self.conv_m_decouple(delta_m)
        decouple_loss = torch.mean(torch.abs(delta_c_conv - delta_m_conv))
        
        # ===== Calcolo hidden states =====
        h_c_new = torch.sigmoid(o_c) * torch.tanh(c_new)
        h_m_new = torch.sigmoid(o_m) * torch.tanh(m_new)
        h_new = h_c_new + h_m_new  # Combinazione lineare
        
        return h_new, c_new, m_new, decouple_loss

class PredRNN_Block(nn.Module):
    def __init__(self, num_layers, num_hidden, filter_size):
        super().__init__()
        self.layers = num_layers
        self.st_cells = nn.ModuleList()
        # Primo layer: accetta 128 canali (output dell'encoder)
        self.st_cells.append(SpatiotemporalLSTMCell(128, num_hidden, filter_size))
        for i in range(1, num_layers):
            self.st_cells.append(SpatiotemporalLSTMCell(num_hidden, num_hidden, filter_size))
        # Fusione bottom-up
        self.conv_fuse = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1)

    def forward(self, inputs, hidden_states):
        new_hidden = []
        new_cell = []
        new_memory = []
        total_decouple_loss = 0
        
        # ===== FASE TOP-DOWN (layer 0 → layer N) =====
        for l in range(self.layers):
            h_prev = hidden_states[0][l]
            c_prev = hidden_states[1][l]
            # Prendi M dal layer precedente (l-1) nella stessa timestep
            m_prev = hidden_states[2][l-1] if l > 0 else None
            h_new, c_new, m_new, layer_loss = self.st_cells[l](inputs, h_prev, c_prev, m_prev)
            total_decouple_loss += layer_loss
            new_hidden.append(h_new)
            new_cell.append(c_new)
            new_memory.append(m_new)
            inputs = h_new  # Output del layer corrente diventa input del successivo
        
        # ===== FASE BOTTOM-UP (layer N-1 → layer 0) =====
        for l in reversed(range(self.layers-1)):
            # Prendi M dal layer inferiore (l+1) e fondono con M corrente
            m_lower = new_memory[l+1]
            m_current = new_memory[l]
            fused = self.conv_fuse(torch.cat([m_current, m_lower], dim=1))
            new_memory[l] = fused  # Aggiorna M del layer corrente
        
        return inputs, [new_hidden, new_cell, new_memory], total_decouple_loss

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

    def forward(self, input_sequence, pred_length, teacher_forcing=False):
        batch_size, seq_len, _, h, w = input_sequence.size()
        h_t, c_t, m_t = [], [], []
        
        for l in range(self.num_layers):
            h_t.append(torch.zeros(batch_size, self.num_hidden, h//2, w//2).to(input_sequence.device))
            c_t.append(torch.zeros(batch_size, self.num_hidden, h//2, w//2).to(input_sequence.device))
            m_t.append(torch.zeros(batch_size, self.num_hidden, h//2, w//2).to(input_sequence.device))
        
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
                # Usa teacher forcing durante l'addestramento
                if teacher_forcing and self.training:
                    x, skip = encoder_features[t]  # Usa dati reali
                else:
                    # Usa la previsione precedente
                    prev_pred = predictions[-1] if predictions else torch.zeros_like(targets)
                    enc_out, _ = self.encoder(prev_pred)
                    x = enc_out
                    skip = encoder_features[-1][1]  # Skip dell'ultimo input reale

            # if t < seq_len:
            #     x, skip = encoder_features[t]  # Usa i dati reali nei primi frame
            # else:
            #     skip = skip if predictions else torch.zeros_like(skip)
            
            hidden_states = [h_t, c_t, m_t]  # Stati prima del blocco RNN
            rnn_out, hidden_states, decouple_loss = self.rnn_block(x, hidden_states)
            h_t, c_t, m_t = hidden_states  # Aggiorna gli stati dopo il blocco
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
def calculate_metrics(preds, targets, threshold_dbz=15):
    # Converti tensori in numpy
    preds = preds.cpu().numpy().squeeze()
    targets = targets.cpu().numpy().squeeze()
    
    # Denormalizza usando il range fisso del paper (0-70 dBZ)
    targets_dbz = targets * 70.0
    preds_dbz = preds * 70.0
    
    # Calcola MAE su dBZ
    mae = np.mean(np.abs(preds_dbz - targets_dbz))
    
    # Calcola SSIM con data_range=70 dBZ
    ssim_val = ssim(
        preds_dbz, 
        targets_dbz, 
        data_range=70.0,
        win_size=3,
        multichannel=False
    )
    
    # Binarizza con threshold fisico (es. 15 dBZ)
    preds_bin = (preds_dbz > threshold_dbz).astype(np.uint8)
    targets_bin = (targets_dbz > threshold_dbz).astype(np.uint8)
    
    # Calcola CSI
    tn, fp, fn, tp = confusion_matrix(targets_bin.flatten(), preds_bin.flatten()).ravel()
    csi = tp / (tp + fp + fn + 1e-10)
    
    return {
        'MAE': mae,
        'SSIM': ssim_val,
        'CSI': csi
    }

# === Training loop ===
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs, decouple_loss = model(inputs, PRED_LENGTH, teacher_forcing=True)
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