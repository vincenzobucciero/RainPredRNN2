
# RainPredRNN2
Radar-nowcasting with a U-Net encoder/decoder + temporal Transformer, trained on sequences of TIFF radar frames.
The model takes INPUT_LENGTH past frames and predicts PRED_LENGTH future frames. It includes data normalization, on-the-fly augmentation, and evaluation metrics (MAE, MSE, SmoothL1, SSIM, CSI).

*Highlights*
  - Input/output: single-channel radar .tiff images
  - U-Net encoder/decoder for spatial features
  - Temporal Transformer for sequence modeling
  - TensorBoard logging
  - Mixed precision (AMP) on CUDA, optional gradient accumulation (easy to enable)
  - Works on Linux, macOS (CPU/MPS), Windows (CPU/CUDA)

## 1) Project Structure
| Path | Purpose |
|---|---|
| `source/app8.py` | Main training & evaluation script |
| `source/…` | Model, dataloaders, utils |
| `dataset_campania/train/**/*.tiff` | Training radar frames (TIFF) |
| `dataset_campania/val/**/*.tiff` | Validation frames |
| `dataset_campania/test/**/*.tiff` | Test frames |
| `checkpoints/` | Saved models (created at runtime)
| `runs/Train`, `runs/Validation` | TensorBoard logs |
| `test_predictions/` | Saved predictions (created at runtime) |

Dataset expectation: three splits (train/, val/, test/) containing chronologically ordered **.tiff** frames.
The loader scans recursively (**/*.tiff).

## 2) Model Overview
- **Encoder**: U-Net downsampling path (Conv-BN-ReLU + MaxPool).
- **Temporal block**: Transformer operating on patch embeddings across time.
- **Decoder**: U-Net upsampling path with skip connections.
- **Normalization**: radar values clipped to [0,70] dBZ → scaled to [0,1].
- **Augmentation (train only)**: flips + random affine (via torchio).
- **Metrics**: MAE, MSE, SmoothL1, SSIM, CSI (threshold in dBZ).
- **Logging**: per-epoch train loss + val metrics to TensorBoard.
- **Outputs**: predicted frames written to test_predictions/… as .tiff.

## 3) Requirements
    > Recommended: use Conda (conda-forge) for GDAL/rasterio stability.
### Core Python deps (Python 3.10–3.12)**:
- *torch, torchvision, numpy, scikit-image*
- *scikit-learn, pytorch-msssim, einops*
- *rasterio (depends on GDAL)*
- *Pillow, tensorboard, torchio*
- 3.1 Quick install (Conda – works on Linux/macOS/Windows)

### Create env
```
conda create -n rainpredrnn2 python=3.11 -y
conda activate rainpredrnn2
```
## Install PyTorch
## Linux/Windows CUDA (choose version from pytorch.org if needed)
CPU-only: replace with 
```
pytorch torchvision cpuonly -c pytorch
```

```
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Rasterio + deps (via conda-forge: safest)
```
conda install -c conda-forge rasterio gdal -y
```

### The rest via pip (or conda if you prefer)
```
pip install numpy scikit-image scikit-learn pytorch-msssim einops pillow tensorboard torchio
```
### 3.2 macOS (Apple Silicon)

Install PyTorch with MPS support (CPU/MPS):
```
conda create -n rainpredrnn2 python=3.11 -y
conda activate rainpredrnn2
conda install pytorch torchvision -c pytorch -y       # this enables MPS on macOS
conda install -c conda-forge rasterio gdal -y
pip install numpy scikit-image scikit-learn pytorch-msssim einops pillow tensorboard torchio
```
You can use "mps" as device (see §6 “Configuration”).

### 3.3 Windows notes
Prefer Conda: 
```
conda-forge provides compatible GDAL/rasterio builds.
```
For CUDA, install the matching PyTorch build (see pytorch.org “Get Started”).

## 4) Quick Start
Clone the repo and open in VS Code or terminal.
Prepare your dataset:
dataset_campania/

├─ train/... .tiff

├─ val/...   .tiff

└─ test/...  .tiff

Edit the dataset path in app8.py (bottom of file):

### macOS (local)
```
DATA_PATH = os.path.abspath("/Users/<you>/Desktop/RainPredRNN2/dataset_campania")
```

### Linux cluster / Ubuntu
```
DATA_PATH = os.path.abspath("/home/<you>/projects/RainPredRNN2/dataset_campania")
```
Run training:
```
cd source
python app8.py
```
Monitor logs with TensorBoard:
```
tensorboard --logdir runs
```

### open the shown URL in your browser
Inspect predictions:
Saved under 
*test_predictions/batch_XXXX/pred_t*.tiff (and ground truth under batch_real_XXXX/)*.

## 5) Configuration (the important knobs)
At the top of app8.py:
```
DEVICE         = get_device()   # see §6 for device setup
NUM_WORKERS    = 0              # DataLoader workers
BATCH_SIZE     = 2
LEARNING_RATE  = 1e-3
NUM_EPOCHS     = 50
INPUT_LENGTH   = 6
PRED_LENGTH    = 6
LAMBDA_DECOUPLE = 0.001         # kept for future loss components
```
Other relevant bits:
- Normalization is handled in **normalize_image()**.
- Augmentation is defined in **get_augmentation_transforms()** (train only).
- Logging dirs: **runs/<timestamp>/Train** and **runs/<timestamp>/Validation**.
- Checkpoints: **checkpoints/best_model.pth**.

## 6) Devices & Precision
### 6.1 CUDA (NVIDIA GPU)
Set:
```
DEVICE = "cuda"
amp_enabled = True
pin_memory = True
NUM_WORKERS = 8
BATCH_SIZE = 4  # increase gradually if you have >16GB VRAM
```
Optional (reduces fragmentation / OOMs):
```
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```
If you still hit CUDA OOM, lower BATCH_SIZE (e.g., 2) and/or use gradient accumulation (see §9).

### 6.2 macOS (Apple Silicon)
Use MPS if available:
```
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
```
AMP (cuda autocast/GradScaler) is not used on MPS. Keep operations in FP32 or try torch.set_float32_matmul_precision("high").

### 6.3 CPU (any OS)
Use:
```
DEVICE = "cpu"
NUM_WORKERS = max(1, os.cpu_count() // 3)
BATCH_SIZE  = 2..8  # depending on RAM
```
Disable any CUDA-specific code paths.
Note: The bundled **get_device()** had a corner case. Make sure multi-GPU returns *"cuda"*, not *"gpu"*. 
A safe version:
```
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

## 7) Running on a SLURM Cluster
Request a GPU node (example: 1 GPU for 4 hours):
```
salloc -p gpu --gres=gpu:1 -c 8 --mem=32G --time=04:00:00
module load anaconda cuda/12.x  # if your cluster uses modules
conda activate rainpredrnn2
cd ~/projects/RainPredRNN2/source
python app8.py
```

Check what’s available:
```
sinfo -o "%P %N %G %c %m %O"     # shows partition, node, GRES (GPU), CPUs, memory
scontrol show node <node_name>   # detailed node info
```

## 8) Outputs & Logging
- **Checkpoints**: checkpoints/best_model.pth
- **TensorBoard logs**: runs/<timestamp>/Train, runs/<timestamp>/Validation
- **Predictions**: test_predictions/batch_0000/pred_t1.tiff, … and ground truth under batch_real_0000/

To visualize logs:
```
tensorboard --logdir runs
```
## 9) Performance Tips & OOM Survival Kit
Start conservative on GPU:
```
BATCH_SIZE = 4
amp_enabled = True  # only on CUDA
```
If you hit CUDA out of memory:

Reduce BATCH_SIZE (e.g., 2).

Enable gradient accumulation to emulate a larger effective batch:
```
ACC_STEPS = 8  # eff. batch = BATCH_SIZE * ACC_STEPS
```
### In the train loop:
```
loss = loss / ACC_STEPS
scaler.scale(loss).backward()
if (step + 1) % ACC_STEPS == 0:
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```
Set allocator flags:
```
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
```
Consider channels_last for convs:
```
model = model.to(memory_format=torch.channels_last)
inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
```
Activation checkpointing (advanced): wrap heavy blocks with torch.utils.
checkpoint.checkpoint().

## 10) Troubleshooting
### 10.1 rasterio / GDAL install errors
Prefer Conda (conda-forge) for gdal/rasterio.

On Windows, using pip wheels for rasterio often requires matching GDAL binaries—again, Conda is easiest.

### 10.2 GitHub rejects push (>100MB single file)
GitHub hard-limits files **>100MB**. Either:
Ignore large raw data in .gitignore, or Use Git LFS:
```
git lfs install
git lfs track "*.tiff"
git add .gitattributes
git add path/to/needed.tiff
git commit -m "Track TIFFs with LFS"
git push
```

### 10.3 VS Code “Sync Changes” from a remote SSH
Configure SSH keys on the remote or use HTTPS + Personal Access Token.

Errors like ECONNREFUSED ... vscode-git-*.sock mean the VS Code credential helper isn’t available remotely → set up SSH on the remote and switch origin to git@github.com:....

### 10.4 macOS (Apple Silicon) slow training
Use MPS device if available; otherwise CPU.

Reduce image size (e.g., transforms to (224, 224) are already set), or lower BATCH_SIZE.

## 11) Reproducibility
```
def set_seed(seed=15):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
```
For strict determinism set ```deterministic=True``` and ```benchmark=False```, but expect slower training.

## 12) License
MIT License.


## 14) Contact
Issues / questions: open a GitHub issue or contact the maintainer.

### ✅ Final checklist before you run
- [ ] Dataset placed under dataset_campania/{train,val,test}
- [ ] DATA_PATH in app8.py points to the correct folder
- [ ] Conda env active; python -c "import torch, rasterio; print(torch.__version__)" works
- [ ] Correct device set (CUDA / MPS / CPU)
- [ ] If on CUDA: try BATCH_SIZE=4, AMP on, allocator flags set tensorboard --logdir runs ready to monitor training
