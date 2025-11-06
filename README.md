
# RainPredRNN2

## Overview
RainPredRNN2 is an advanced deep learning system for radar-based precipitation nowcasting, combining a U-Net encoder/decoder architecture with a temporal Transformer. The model processes sequences of radar reflectivity images (TIFF format) to predict future precipitation patterns.

### Key Features
- **Advanced Architecture**: 
  - U-Net encoder/decoder for robust spatial feature extraction
  - Temporal Transformer for sophisticated sequence modeling
  - Hybrid design optimized for meteorological predictions
- **Data Processing**:
  - Input/Output: Single-channel radar TIFF images
  - Automated data normalization and augmentation
  - Comprehensive evaluation metrics (MAE, MSE, SSIM, CSI)
- **Technical Features**:
  - Mixed precision training (AMP) on CUDA
  - Flexible gradient accumulation
  - Cross-platform support (Linux, macOS, Windows)
  - Real-time TensorBoard logging
  - Extensive performance optimization options

## 1. Project Structure
```
RainPredRNN2/
├── source/
│   ├── app9.py              # Main training script
│   ├── infer.py             # Inference script
│   ├── extract_tiff.py      # Data extraction utility
│   └── checkpoints/         # Model weights
├── dataset_campania/        # Dataset directory
│   ├── train/              # Training data
│   ├── val/                # Validation data
│   └── test/               # Test data
├── runs/                    # TensorBoard logs
└── test_predictions/        # Model outputs
```

### Key Components
| Component | Description |
|-----------|-------------|
| `app9.py` | Primary training and evaluation script with advanced configuration options |
| `infer.py` | Standalone inference script for model deployment |
| `extract_tiff.py` | Utility for preparing input data sequences |
| `dataset_campania/` | Structured dataset directory containing chronologically ordered TIFF frames |
| `checkpoints/` | Storage for model weights and training states |
| `runs/` | TensorBoard logging directory for monitoring training progress |
| `test_predictions/` | Output directory for model predictions |

### Dataset Structure
The dataset should be organized in three splits (train/val/test), each containing chronologically ordered TIFF frames. The data loader recursively scans for `*.tiff` files in each directory.

## 2. Model Architecture

### Network Components
- **Encoder Network**
  - U-Net downsampling path with Conv-BN-ReLU blocks
  - MaxPool operations for spatial feature hierarchy
  - Efficient feature extraction from radar images

- **Temporal Processing**
  - Transformer architecture for sequence modeling
  - Patch-wise embeddings for temporal attention
  - Advanced temporal dependency learning

- **Decoder Network**
  - U-Net upsampling path with skip connections
  - Feature refinement through progressive upsampling
  - High-resolution output reconstruction

### Data Processing Pipeline
- **Normalization**
  - Radar values clipped to [0,70] dBZ range
  - Linear scaling to [0,1] for stable training
  - Automated handling of missing/invalid values

- **Augmentation** (Training Only)
  - Spatial flips and random affine transformations
  - Implemented via torchio for efficiency
  - Configurable augmentation parameters

### Evaluation System
- **Core Metrics**
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - SSIM (Structural Similarity Index)
  - CSI (Critical Success Index) with configurable dBZ threshold

- **Monitoring**
  - Real-time TensorBoard logging
  - Per-epoch training loss tracking
  - Validation metrics visualization
  - Model prediction samples

## 3. Installation and Setup

### Prerequisites
> **Recommended**: Use Conda (conda-forge) for consistent GDAL/rasterio installation

### Dependencies
- **Python Version**: 3.10-3.12
- **Core Libraries**:
  - PyTorch & torchvision
  - NumPy & scikit-image
  - scikit-learn
  - pytorch-msssim & einops
  - rasterio (GDAL-dependent)
  - Pillow, tensorboard, torchio

### Installation Guide

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

## 10. Model Inference

### Using infer.py
The `infer.py` script provides a streamlined way to generate predictions using a trained model.

#### Prerequisites
1. Trained model checkpoint (`best_model.pth`)
2. Sequence of 18 input TIFF files
3. Properly configured Python environment

#### Configuration
```python
# Key Parameters in infer.py
CKPT_PATH  = "/path/to/checkpoints/best_model.pth"  # Model weights
INPUT_DIR  = "/path/to/extracted_tiff"              # Input frames
OUTPUT_DIR = "./pred_out"                           # Prediction output
REPEATS    = 50                                     # Performance test iterations
USE_AMP    = True                                   # Mixed precision inference
RESIZE_HW  = (224, 224)                            # Input dimensions
```

#### Input Preparation
1. **Extract Input Sequence**
   ```bash
   # Extract 18 consecutive frames
   python extract_tiff.py --src "/path/to/data/test" --mode random --clear-dst
   ```

2. **Available Extraction Modes**
   - `--mode start`: Sequential frames from start
   - `--mode random`: Random starting point
   - `--mode next`: Sliding window across dataset

#### Running Inference
```bash
# Basic inference
python infer.py

# Using SLURM (if available)
sbatch infer.sbatch
```

#### Output Structure
1. **Predictions**
   - 6 TIFF files for future timestamps
   - Named according to timestamp progression
   - Located in `pred_out/` directory

2. **Performance Metrics**
   ```
   === Inference 18→6 (batch=1) ===
   Ripetizioni       : 50
   Tempo medio (s)   : X.XXXX
   Dev std (s)       : X.XXXX
   P50 (s)          : X.XXXX
   P95 (s)          : X.XXXX
   Predizioni/s      : XX.XX
   ```

### Advanced Usage

#### Batch Processing
To process multiple sequences:
1. Create a list of input directories
2. Modify `extract_tiff.py` mode for sequential extraction
3. Run inference in a loop or parallel processes

#### Performance Optimization
- Enable AMP (`USE_AMP = True`) for CUDA devices
- Adjust `RESIZE_HW` based on memory constraints
- Use `WARMUP` iterations for stable timing measurements

#### Output Analysis
- Use standard GIS tools to visualize TIFF outputs
- Compare predictions with actual observations
- Analyze temporal consistency of predictions

## 11) Troubleshooting
### 11.1 rasterio / GDAL install errors
Prefer Conda (conda-forge) for gdal/rasterio.

On Windows, using pip wheels for rasterio often requires matching GDAL binaries—again, Conda is easiest.

### 11.2 GitHub rejects push (>100MB single file)
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

### 11.3 VS Code “Sync Changes” from a remote SSH
Configure SSH keys on the remote or use HTTPS + Personal Access Token.

Errors like ECONNREFUSED ... vscode-git-*.sock mean the VS Code credential helper isn’t available remotely → set up SSH on the remote and switch origin to git@github.com:....

### 11.4 macOS (Apple Silicon) slow training
Use MPS device if available; otherwise CPU.

Reduce image size (e.g., transforms to (224, 224) are already set), or lower BATCH_SIZE.

## 12) Reproducibility
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


## 13) Contact
Issues / questions: open a GitHub issue or contact the maintainer.

### ✅ Final checklist before you run
- [ ] Dataset placed under dataset_campania/{train,val,test}
- [ ] DATA_PATH in app8.py points to the correct folder
- [ ] Conda env active; python -c "import torch, rasterio; print(torch.__version__)" works
- [ ] Correct device set (CUDA / MPS / CPU)
- [ ] If on CUDA: try BATCH_SIZE=4, AMP on, allocator flags set tensorboard --logdir runs ready to monitor training
