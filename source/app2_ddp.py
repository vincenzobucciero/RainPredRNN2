import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast  # Per mixed-precision
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import rasterio
from rasterio.errors import RasterioIOError
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix
from PIL import Image
import torchvision.transforms as transforms
import glob

# === Funzioni per il setup distribuito ===
def setup_distributed():
    if dist.is_initialized():  # Controlla se il gruppo è già inizializzato
        dist.destroy_process_group()
        
    dist_url = "env://"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method=dist_url, 
                            world_size=world_size, rank=rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def get_ddp_model(model, local_rank):
    model = model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])
    return ddp_model

# === Configurazione multi-GPU ===
def get_device():
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Configurazione base ===
local_rank = setup_distributed()
DEVICE = torch.device(f"cuda:{local_rank}")
NUM_WORKERS = 4
BATCH_SIZE = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
INPUT_LENGTH = 6
PRED_LENGTH = 6
LAMBDA_DECOUPLE = 0.01

# Normalizzazione delle immagini
def normalize_image(img):
    # Verifica che i valori siano validi
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0, None)  # Assicurati che non ci siano valori negativi
    
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
def set_seed(seed=15):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

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
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        # Cache per tenere traccia della validità dei file
        self.file_validity = {}
        
        # Calcolo finestre valide
        self.valid_indices = []
        self.total_possible_windows = max(0, len(self.files) - self.seq_length + 1)
        
        for start_idx in range(self.total_possible_windows):
            window_valid = True
            for i in range(self.seq_length):
                file = self.files[start_idx + i]
                
                # Verifica validità del file solo se non già controllato
                if file not in self.file_validity:
                    try:
                        with rasterio.open(file) as src:
                            valid = src.count > 0
                    except RasterioIOError:
                        valid = False
                        print(f"File non valido: {file}", flush=True)
                    self.file_validity[file] = valid
                
                if not self.file_validity[file]:
                    window_valid = False
                    break  # Interrompe il ciclo alla prima occorrenza di file non valido
                
            if window_valid:
                self.valid_indices.append(start_idx)
        
        # Statistiche finali
        self.total_files = len(self.files)
        self.invalid_files = sum(1 for valid in self.file_validity.values() if not valid)
        self.valid_windows = len(self.valid_indices)
        self.invalid_windows = self.total_possible_windows - self.valid_windows
        
        print(f"\nStatistiche Dataset:", flush=True)
        print(f"1. File totali: {self.total_files}", flush=True)
        print(f"2. File non validi: {self.invalid_files}", flush=True)
        print(f"3. Finestre totali possibili: {self.total_possible_windows}", flush=True)
        print(f"4. Finestre valide: {self.valid_windows}", flush=True)
        print(f"5. Finestre non valide: {self.invalid_windows}", flush=True)
        print(" ===================================================== ", flush=True)

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
    def __init__(self, input_dim, hidden_dim, filter_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Convoluzioni per i gate combinati (input + stato nascosto)
        padding = filter_size // 2
        self.conv_x = nn.Conv2d(input_dim, hidden_dim * 7, 
                               kernel_size=filter_size, padding=padding)
        self.conv_h = nn.Conv2d(hidden_dim, hidden_dim * 7, 
                               kernel_size=filter_size, padding=padding)
        
        # Convoluzioni per memoria spaziale (C) e temporale (M)
        self.conv_c = nn.Conv2d(hidden_dim, hidden_dim * 3, 
                               kernel_size=1)
        self.conv_m = nn.Conv2d(hidden_dim, hidden_dim * 3, 
                               kernel_size=1)
        
        # Convoluzione 1x1 per fondere C e M nell'output
        self.conv_1x1 = nn.Conv2d(hidden_dim * 2, hidden_dim, 
                                 kernel_size=1)
        
        # Convoluzione per la decoupling loss
        self.conv_decouple = nn.Conv2d(hidden_dim, hidden_dim, 
                                       kernel_size=1)

    def forward(self, x, h_prev, c_prev, m_prev, m_upper=None):
        # Aggiorna memoria temporale con flusso bottom-up
        if m_upper is not None:
            m_prev = m_prev + m_upper  # Combinazione con memoria del livello superiore

        # Assicurati che le dimensioni spaziali di x e h_prev siano coerenti
        if x.size(2) != h_prev.size(2) or x.size(3) != h_prev.size(3):
            x = F.interpolate(x, size=(h_prev.size(2), h_prev.size(3)), mode='bilinear', align_corners=False)

        # Calcolo gate combinati
        combined = self.conv_x(x) + self.conv_h(h_prev)
        i_c, i_m, f_c, f_m, g_c, g_m, o = torch.split(
            combined, self.hidden_dim, dim=1)

        # Aggiornamento memoria spaziale (C)
        c_conv = self.conv_c(c_prev)
        f_c_c, i_c_c, o_c = torch.split(c_conv, self.hidden_dim, dim=1)
        delta_c = torch.sigmoid(i_c + i_c_c) * torch.tanh(g_c)
        c_new = torch.sigmoid(f_c + f_c_c) * c_prev + delta_c

        # Aggiornamento memoria temporale (M)
        m_conv = self.conv_m(m_prev)
        f_m_m, i_m_m, o_m = torch.split(m_conv, self.hidden_dim, dim=1)
        delta_m = torch.sigmoid(i_m + i_m_m) * torch.tanh(g_m)
        m_new = torch.sigmoid(f_m + f_m_m) * m_prev + delta_m

        # === Modifiche chiave ===
        # 1. Fusione C e M con convoluzione 1x1
        combined_states = torch.cat([c_new, m_new], dim=1)
        fused_states = self.conv_1x1(combined_states)
        
        # 2. Calcolo stato nascosto finale
        h_new = torch.sigmoid(o) * torch.tanh(fused_states)
        
        # 3. Decoupling loss con convoluzione condivisa
        delta_c_decoupled = self.conv_decouple(delta_c)
        delta_m_decoupled = self.conv_decouple(delta_m)
        decouple_loss = torch.mean(torch.abs(delta_c_decoupled - delta_m_decoupled))
        
        return h_new, c_new, m_new, decouple_loss

class PredRNN_Block(nn.Module):
    def __init__(self, num_layers, num_hidden, filter_size):
        super().__init__()
        self.cells = nn.ModuleList()
        for _ in range(num_layers):
            self.cells.append(SpatiotemporalLSTMCell(
                input_dim=num_hidden,
                hidden_dim=num_hidden,
                filter_size=filter_size
            ))
        self.num_layers = num_layers

    def forward(self, input_sequence, h_t, c_t, m_t):
        seq_len = input_sequence.size(1)
        output_inner = []
        total_decouple_loss = 0.0

        for t in range(seq_len):
            ##########################
            # Fase 1: Propagazione bottom-up (da layer alto a basso)
            ##########################
            for l in reversed(range(self.num_layers)):
                # Input per il layer corrente
                if l == 0:
                    input_current = input_sequence[:, t]  # Input diretto per il primo layer
                else:
                    input_current = h_t[l-1]  # Output del layer precedente
                
                # Memoria del layer superiore (se esiste)
                m_upper = m_t[l+1] if l < self.num_layers - 1 else None
                
                # Aggiorna gli stati del layer
                h_new, c_new, m_new, cell_loss = self.cells[l](
                    input_current,
                    h_t[l],
                    c_t[l],
                    m_t[l],
                    m_upper
                )
                
                # Aggiorna gli stati persistenti
                h_t[l] = h_new
                c_t[l] = c_new
                m_t[l] = m_new
                total_decouple_loss += cell_loss  # Accumula la loss

            ##########################
            # Fase 2: Propagazione top-down (da layer basso ad alto)
            ##########################
            for l in range(1, self.num_layers):
                input_current = h_t[l-1]  # Output del layer precedente
                m_upper = m_t[l-1]  # Memoria del layer inferiore (aggiornata nella fase 1)
                
                # Aggiorna gli stati del layer
                h_new, c_new, m_new, cell_loss = self.cells[l](
                    input_current,
                    h_t[l],
                    c_t[l],
                    m_t[l],
                    m_upper
                )
                
                # Aggiorna gli stati persistenti
                h_t[l] = h_new
                c_t[l] = c_new
                m_t[l] = m_new
                total_decouple_loss += cell_loss  # Accumula la loss

            # Salva l'output dell'ultimo layer
            output_inner.append(h_t[-1])
        
        total_decouple_loss /= (seq_len * self.num_layers)

        return torch.stack(output_inner, dim=1), h_t, c_t, m_t, total_decouple_loss

class UNet_Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.enc1 = self.contract_block(in_channels, 32, 3, 1)
        self.enc2 = self.contract_block(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.InstanceNorm2d(out_channels),
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
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.expand_block(64, 32, 3, 1)
        self.upconv2 = nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2)
        
        # Aggiungiamo un layer di convoluzione per allineare i canali
        self.skip_conv = nn.Conv2d(32, 32, kernel_size=1)

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, skip):
        # Prima di concatenare, allineiamo i canali della skip connection
        skip = self.skip_conv(skip)
        
        x = self.upconv1(x)
        
        # Se le dimensioni non corrispondono, usiamo l'interpolazione
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.dec1(x)
        x = self.upconv2(x)
        return torch.sigmoid(x)

class RainPredRNN(nn.Module):
    def __init__(self, input_dim=1, num_hidden=64, num_layers=3, filter_size=3):
        super().__init__()
        self.encoder = UNet_Encoder(input_dim)
        self.decoder = UNet_Decoder(input_dim)
        self.rnn_block = PredRNN_Block(num_layers, num_hidden, filter_size)
        self.num_layers = num_layers
        self.num_hidden = num_hidden

    def forward(self, input_sequence, pred_length, teacher_forcing=False):
        # Rimuovi il parametro 'targets' e la logica del teacher forcing
        batch_size, seq_len, _, h, w = input_sequence.size()
        device = input_sequence.device
        
        # Codifica degli input e salvataggio delle skip connections
        encoder_skips = []
        encoder_outputs = []
        for t in range(seq_len):
            enc_out, skip = self.encoder(input_sequence[:, t])
            encoder_outputs.append(enc_out)
            encoder_skips.append(skip)
        
        predictions = []
        total_decouple_loss = 0.0
        
        # Inizializza gli stati persistenti
        h_t = [torch.zeros(batch_size, self.num_hidden, h//2, w//2).to(device) 
            for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch_size, self.num_hidden, h//2, w//2).to(device) 
            for _ in range(self.num_layers)]
        m_t = [torch.zeros(batch_size, self.num_hidden, h//2, w//2).to(device) 
            for _ in range(self.num_layers)]
        
        for t in range(seq_len + pred_length):
            if t < seq_len:
                # Fase di input: usa gli encoder_skips del timestep corrente
                x = encoder_outputs[t]  # enc_out è l'output dell'encoder per il timestep t
            else:
                # Usa SEMPRE la precedente predizione (niente teacher forcing)
                prev_pred = predictions[-1] if predictions else input_sequence[:, -1]
                x, skip = self.encoder(prev_pred)
                
                # Aggiorna le skip connections SOLO durante il training
                if self.training:
                    encoder_skips.append(skip)
                
                # Usa la skip connection corrente (dalla predizione)
                current_skip = skip  # <--- Correzione critica
            
            # Esegui il passaggio attraverso il blocco RNN
            rnn_out, h_t, c_t, m_t, decouple_loss = self.rnn_block(
                x.unsqueeze(1), h_t, c_t, m_t
            )
            total_decouple_loss += decouple_loss
            
            # Genera la predizione per i timestep futuri
            if t >= seq_len:
                pred = self.decoder(rnn_out.squeeze(1), current_skip)
                predictions.append(pred)
        
        return torch.stack(predictions, dim=1), total_decouple_loss

# === Inizializzazione pesi ===
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

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
        
        # Aggiungi il contesto autocast
        with autocast():
            outputs, decouple_loss = model(inputs, PRED_LENGTH, teacher_forcing=False)
            loss = criterion(outputs, targets) + LAMBDA_DECOUPLE * (decouple_loss / (INPUT_LENGTH + PRED_LENGTH))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        # Liberare memoria periodicamente
        torch.cuda.empty_cache()

    return total_loss / len(loader)

# === Valutazione ===
def evaluate(model, loader, device):
    model.eval()
    metrics = {'MAE': 0, 'SSIM': 0, 'CSI': 0}
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs, PRED_LENGTH, teacher_forcing=False)
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
    preds = predictions.squeeze().cpu().numpy()
    
    for idx, seq in enumerate(preds):
        for t in range(seq.shape[0]):
            frame = (seq[t] * 70.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(frame)
            filename = os.path.join(output_dir, f"pred_{idx:04d}_t{t+1}.tiff")
            img.save(filename)

# === Inizializzazione modello ===
torch.cuda.empty_cache()
model = RainPredRNN(input_dim=1, num_hidden=64, num_layers=2, filter_size=3)
# model.apply(init_weights)
model = DataParallel(model).to(DEVICE)

# === Ottimizzatore e loss ===
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.MSELoss()

# === Supporto mixed-precision ===
scaler = GradScaler() # Per l'addestramento a precisione mista

# === Modifica del main ===
if __name__ == "__main__":
    # Setup distribuito
    local_rank = setup_distributed()
    DEVICE = torch.device(f"cuda:{local_rank}")
    
    # Configurazione percorsi
    DATA_PATH = "/home/f.demicco/RainPredRNN2/dataset"
    CHECKPOINT_DIR = "/home/f.demicco/RainPredRNN2/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Creazione dataloaders con DistributedSampler
    train_dataset = RadarDataset(os.path.join(DATA_PATH, 'train'), is_train=True)
    val_dataset = RadarDataset(os.path.join(DATA_PATH, 'val'), is_train=False)
    test_dataset = RadarDataset(os.path.join(DATA_PATH, 'test'), is_train=False)

    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=local_rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=local_rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Non shuffle qui quando si usa DistributedSampler
        num_workers=NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        sampler=val_sampler
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Inizializzazione modello con DDP
    model = RainPredRNN(input_dim=1, num_hidden=64, num_layers=2, filter_size=3)
    model = get_ddp_model(model, local_rank)

    # Ottimizzatore e loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    # Supporto mixed-precision
    scaler = GradScaler()

    # Training
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)  # Richiesto per DistributedSampler
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        val_metrics = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_metrics['MAE'])

        if local_rank == 0:  # Solo il processo principale stampa e salva
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}", flush=True)
            print(f"\tTrain Loss: {train_loss:.4f}", flush=True)
            print(f"\tVal MAE: {val_metrics['MAE']:.4f}, SSIM: {val_metrics['SSIM']:.4f}, CSI: {val_metrics['CSI']:.4f}", flush=True)

            if val_metrics['MAE'] < best_val_loss:
                best_val_loss = val_metrics['MAE']
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))

    # Test finale
    if local_rank == 0:  # Solo il processo principale esegue il test
        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth")))
        test_metrics = evaluate(model, test_loader, DEVICE)
        print("Test Results:", flush=True)
        print(f"\tMAE: {test_metrics['MAE']:.4f}", flush=True)
        print(f"\tSSIM: {test_metrics['SSIM']:.4f}", flush=True)
        print(f"\tCSI: {test_metrics['CSI']:.4f}", flush=True)

        # Salvataggio predizioni test
        os.makedirs("/home/f.demicco/RainPredRNN2/test_predictions", exist_ok=True)
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(DEVICE)
                outputs, _ = model(inputs, PRED_LENGTH)
                save_predictions(outputs, f"/home/f.demicco/RainPredRNN2/test_predictions/batch_{i:04d}")