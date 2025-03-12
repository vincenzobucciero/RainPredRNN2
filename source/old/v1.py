import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
import rasterio
from rasterio.errors import RasterioIOError
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim

# === Inizializzazione seed per riproducibilità ===
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
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.valid_indices = []
        self.file_validity = {}
        
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
        super(SpatiotemporalLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        
        # Gates for conventional cell (C)
        self.conv_xc = nn.Conv2d(in_channel, num_hidden*4, kernel_size=filter_size, padding=self.padding)
        self.conv_hc = nn.Conv2d(num_hidden, num_hidden*4, kernel_size=filter_size, padding=self.padding)
        self.conv_cc = nn.Conv2d(num_hidden, num_hidden*3, kernel_size=1, padding=0)
        
        # Gates for spatiotemporal cell (M)
        self.conv_xm = nn.Conv2d(in_channel, num_hidden*4, kernel_size=filter_size, padding=self.padding)
        self.conv_hm = nn.Conv2d(num_hidden, num_hidden*4, kernel_size=filter_size, padding=self.padding)
        self.conv_mm = nn.Conv2d(num_hidden, num_hidden*4, kernel_size=1, padding=0)
        
        # Decoupling 1x1 convolutions
        self.conv_c_decouple = nn.Conv2d(num_hidden, num_hidden, kernel_size=1, padding=0)
        self.conv_m_decouple = nn.Conv2d(num_hidden, num_hidden, kernel_size=1, padding=0)

    def forward(self, x, h_c, c_c, m_c):
        # Process conventional cell (C)
        i_c, f_c, g_c, o_c = torch.split(
            self.conv_xc(x) + self.conv_hc(h_c),
            self.num_hidden, dim=1)
        
        c_prev = c_c
        c = F.sigmoid(f_c) * c_prev + F.sigmoid(i_c) * torch.tanh(g_c)
        h_c_new = F.sigmoid(o_c) * torch.tanh(c)
        
        # Process spatiotemporal cell (M)
        i_m, f_m, g_m, o_m = torch.split(
            self.conv_xm(x) + self.conv_hm(h_c) + self.conv_mm(m_c),
            self.num_hidden, dim=1)
        
        m = F.sigmoid(f_m) * m_c + F.sigmoid(i_m) * torch.tanh(g_m)
        h_m_new = F.sigmoid(o_m) * torch.tanh(m)
        
        # Decoupling projections
        c_decouple = self.conv_c_decouple(c)
        m_decouple = self.conv_m_decouple(m)
        
        # Combine outputs
        h_new = h_c_new + h_m_new
        return h_new, c_decouple, m_decouple

class PredRNN_Block(nn.Module):
    def __init__(self, num_layers, num_hidden, filter_size):
        super(PredRNN_Block, self).__init__()
        self.layers = num_layers
        self.st_cells = nn.ModuleList()
        self.num_hidden = num_hidden
        
        for i in range(num_layers):
            in_channel = num_hidden if i > 0 else 1  # Input layer uses original channels
            cell = SpatiotemporalLSTMCell(in_channel, num_hidden, filter_size)
            self.st_cells.append(cell)

    def forward(self, inputs, hidden_states):
        new_hidden = []
        new_cell = []
        new_memory = []
        decouple_loss = 0
        
        for l in range(self.layers):
            h_prev = hidden_states[0][l]
            c_prev = hidden_states[1][l]
            m_prev = hidden_states[2][l] if l > 0 else None
            
            # Flusso zig-zag tra i layer
            if l > 0:
                m_below = new_memory[l-1]
                m_prev = m_below
                
            h_new, c_new, m_new = self.st_cells[l](inputs, h_prev, c_prev, m_prev)
            
            # Aggiorna loss di decoupling
            decouple_loss += torch.mean(torch.abs(c_new - m_new))
            
            inputs = h_new
            new_hidden.append(h_new)
            new_cell.append(c_new)
            new_memory.append(m_new)
        
        return inputs, [new_hidden, new_cell, new_memory], decouple_loss

class UNet_Encoder(nn.Module):
    def __init__(self, in_channels):
        super(UNet_Encoder, self).__init__()
        self.enc1 = self.contract_block(in_channels, 64, 3, 1)
        self.enc2 = self.contract_block(64, 128, 3, 1)
        self.pool = nn.MaxPool2d(2)
        
    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return block

    def forward(self, x):
        # Salva le feature maps per skip connections
        x1 = self.enc1(x)
        x_pooled1 = self.pool(x1)
        x2 = self.enc2(x_pooled1)
        return x2, x1  # Restituisce l'output e la feature map per skip

class UNet_Decoder(nn.Module):
    def __init__(self, out_channels):
        super(UNet_Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        
    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return block

    def forward(self, x, skip):
        x = self.upconv1(x)
        x = torch.cat([x, skip], dim=1)
        x = self.dec1(x)
        x = self.upconv2(x)
        return torch.sigmoid(x)

class RainPredRNN(nn.Module):
    def __init__(self, input_dim=1, num_hidden=64, num_layers=2, filter_size=3):
        super(RainPredRNN, self).__init__()
        self.encoder = UNet_Encoder(input_dim)
        self.decoder = UNet_Decoder(input_dim)
        self.rnn_block = PredRNN_Block(num_layers, num_hidden, filter_size)
        self.num_layers = num_layers
        self.num_hidden = num_hidden

    def forward(self, input_sequence, pred_length, lambda_decouple=0.1):
        batch_size, seq_len, _, h, w = input_sequence.size()
        h = h // 4  # Dopo l'encoder UNet
        w = w // 4
        
        # Inizializza stati nascosti
        h_t = []
        c_t = []
        m_t = []
        for l in range(self.num_layers):
            h_t.append(torch.zeros(batch_size, self.num_hidden, h, w).to(input_sequence.device))
            c_t.append(torch.zeros(batch_size, self.num_hidden, h, w).to(input_sequence.device))
            m_t.append(torch.zeros(batch_size, self.num_hidden, h, w).to(input_sequence.device))
        
        total_loss = 0
        predictions = []
        encoder_features = []
        
        # Codifica input
        for t in range(seq_len):
            enc_out, skip = self.encoder(input_sequence[:, t])
            encoder_features.append((enc_out, skip))
        
        # Processo RNN
        for t in range(seq_len + pred_length):
            if t < seq_len:
                x, skip = encoder_features[t]
            else:
                x = torch.zeros_like(enc_out).to(input_sequence.device)
                skip = torch.zeros_like(skip).to(input_sequence.device)
                
            # if t < seq_len:
            #     x, skip = encoder_features[t]  # Usa i dati reali nei primi frame
            # else:
            #     x = predictions[-1] if predictions else torch.zeros_like(enc_out).to(input_sequence.device)  
            #     skip = skip if predictions else torch.zeros_like(skip).to(input_sequence.device)

            # per evitare conversioni ridondanti
            # x = predictions[-1] if predictions else torch.zeros_like(enc_out, device=enc_out.device)
            # skip = skip if predictions else torch.zeros_like(skip, device=skip.device)
            
            rnn_out, (h_t, c_t, m_t), decouple_loss = self.rnn_block(x, [h_t, c_t, m_t])
            total_loss += lambda_decouple * decouple_loss
            
            if t >= seq_len:
                dec_out = self.decoder(rnn_out, skip)
                predictions.append(dec_out)
        
        return torch.stack(predictions, dim=1), total_loss

# === Inizializzazione pesi ===
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# === Configurazione multi-GPU ===
def get_device():
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

# === Inizializzazione modello ===
model = RainPredRNN(input_dim=1, num_hidden=64, num_layers=2, filter_size=3)
model.apply(init_weights)
model = nn.DataParallel(model).to(device)

# === Ottimizzatore e scheduler ===
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.MSELoss()

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
def calculate_metrics(preds, targets, threshold=0.5):
    preds = preds.cpu().numpy().squeeze()
    targets = targets.cpu().numpy().squeeze()
    
    mae = np.mean(np.abs(preds - targets))
    
    ssim_val = ssim(
        preds, targets,
        data_range=targets.max() - targets.min(),
        multichannel=True
    )
    
    preds_bin = (preds > threshold).astype(np.uint8)
    targets_bin = (targets > threshold).astype(np.uint8)
    
    tn, fp, fn, tp = confusion_matrix(
        targets_bin.flatten(),
        preds_bin.flatten()
    ).ravel()
    
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
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs, decouple_loss = model(inputs, pred_length=6)
        loss = criterion(outputs, targets) + 0.1 * decouple_loss
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
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, _ = model(inputs, pred_length=6)
            batch_metrics = calculate_metrics(outputs, targets)
            for k in metrics:
                metrics[k] += batch_metrics[k]
    for k in metrics:
        metrics[k] /= len(loader)
    return metrics

# === Main execution ===
if __name__ == "__main__":
    # Configurazione
    data_path = "/path/to/dataset"
    batch_size = 8
    num_epochs = 100
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Creazione data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path,
        batch_size=batch_size,
        num_workers=8
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics['MAE'])
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val MAE: {val_metrics['MAE']:.4f}, SSIM: {val_metrics['SSIM']:.4f}, CSI: {val_metrics['CSI']:.4f}")
        
        # Salvataggio checkpoint
        if val_metrics['MAE'] < best_val_loss:
            best_val_loss = val_metrics['MAE']
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
    
    # Valutazione finale
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pth")))
    test_metrics = evaluate(model, test_loader, device)
    print("Test Results:")
    print(f"MAE: {test_metrics['MAE']:.4f}")
    print(f"SSIM: {test_metrics['SSIM']:.4f}")
    print(f"CSI: {test_metrics['CSI']:.4f}")