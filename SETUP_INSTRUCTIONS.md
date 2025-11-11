# Setup Instructions for Google Colab

## 📦 Overview

This ControlNet training project is designed to run on Google Colab by importing modular scripts from your Google Drive. This approach keeps the code organized and maintainable.

## 🎯 Architecture

The project uses a **modular architecture**:

```
controlnet_light/              ← Upload this entire folder to Google Drive
├── train_controlnet.py        ← Core training logic
├── prepare_dataset.py          ← Dataset preparation utilities
├── inference.py                ← Model testing/inference
├── config_template.py          ← Configuration presets
├── colab_train.py              ← Orchestration script (imports everything)
└── ...other files
```

**`colab_train.py`** is the entry point that:
1. Mounts Google Drive
2. Adds the repository to Python path
3. Imports from other modules
4. Orchestrates the entire training process

## 🚀 Setup Steps

### Step 1: Upload Repository to Google Drive

1. Download or clone this repository to your local machine
2. Upload the **entire `controlnet_light` folder** to Google Drive
3. Recommended location: `/MyDrive/controlnet_light/`

Your Google Drive structure should look like:
```
MyDrive/
└── controlnet_light/
    ├── train_controlnet.py
    ├── prepare_dataset.py
    ├── colab_train.py
    ├── inference.py
    ├── config_template.py
    ├── requirements.txt
    └── ...
```

### Step 2: Open Google Colab

1. Go to https://colab.research.google.com/
2. Create a new notebook or upload `colab_train.py`
3. Change runtime to **GPU**:
   - Menu: `Runtime` → `Change runtime type`
   - Hardware accelerator: `GPU`
   - GPU type: `T4` (free) or `A100` (Colab Pro)
   - Click `Save`

### Step 3: Configure Paths

Edit `colab_train.py` and modify these key lines:

```python
# Line ~52: Repository path in Google Drive
REPO_PATH = "/content/drive/MyDrive/controlnet_light"

# Lines ~95-97: Output directories
config.output_dir = "/content/drive/MyDrive/AML/controlnet_trained"
config.dataset_base = "/content/drive/MyDrive/AML/dataset"
config.checkpoint_dir = "/content/drive/MyDrive/AML/checkpoints"

# Lines ~100-101: Dataset sources (if using Sketchy dataset)
SKETCHY_PHOTO_ROOT = "/root/.cache/kagglehub/datasets/.../photo"
SKETCHY_CSV_PATH = "/content/drive/Shareddrives/AML/path_caption_pairs.csv"
```

### Step 4: Run Training

In Colab, run:

```python
!python colab_train.py
```

Or paste the entire contents of `colab_train.py` into cells and run them sequentially.

### Step 5: Monitor Progress

The script will:
- ✅ Mount Google Drive
- ✅ Install dependencies
- ✅ Verify repository structure
- ✅ Prepare dataset (if needed)
- ✅ Start training
- ✅ Save checkpoints automatically
- ✅ Display progress and loss metrics

## 📊 What Happens During Training

1. **Repository Verification**: Checks that all required files exist
2. **Configuration**: Loads `TrainingConfig` from `train_controlnet.py`
3. **Dataset Preparation**: Uses `prepare_dataset.py` to organize images
4. **Training**: Calls `train()` function from `train_controlnet.py`
5. **Checkpointing**: Saves model every N steps to Google Drive
6. **Completion**: Final model saved to `config.output_dir`

## 🔧 Customization

### Quick Configuration Changes

Edit these in `colab_train.py`:

```python
# Training duration
config.max_train_steps = 10000  # More steps = better quality

# Batch size (reduce if OOM)
config.train_batch_size = 4  # Try 2 or 1 if out of memory

# Learning rate
config.learning_rate = 1e-5  # Lower = more stable, Higher = faster convergence

# Resolution
config.resolution = 512  # 256, 384, 512, or 768
```

### Use Pre-configured Presets

Instead of `TrainingConfig()`, use a preset:

```python
from config_template import LowMemoryConfig, HighQualityConfig

# For free Colab T4
config = LowMemoryConfig()

# OR for high-end GPUs
config = HighQualityConfig()

# Then customize as needed
config.output_dir = "your/path"
```

## 🧪 Testing Your Model

After training, test your model:

```python
# Import inference utilities from the repository
from inference import load_model, generate_image

# Load your trained model
pipe = load_model('/content/drive/MyDrive/AML/controlnet_trained')

# Generate an image
generate_image(
    pipe=pipe,
    input_image_path='sketch.jpg',
    prompt='a realistic photo of a cat',
    output_path='result.png',
    condition_type='scribble'
)
```

## 📁 File Locations After Training

```
Google Drive/
├── MyDrive/
│   ├── controlnet_light/           ← Your repository (source code)
│   │   ├── train_controlnet.py
│   │   ├── colab_train.py
│   │   └── ...
│   │
│   └── AML/
│       ├── controlnet_trained/     ← Final trained model
│       │   ├── config.json
│       │   ├── diffusion_pytorch_model.bin
│       │   └── training_config.json
│       │
│       ├── checkpoints/            ← Training checkpoints
│       │   ├── checkpoint-1000/
│       │   ├── checkpoint-2000/
│       │   └── ...
│       │
│       └── dataset/                ← Training data
│           ├── images/
│           └── captions.csv
```

## ⚠️ Important Notes

### Repository Must Be in Drive

The entire repository **must** be uploaded to Google Drive. The `colab_train.py` script imports from:
- `train_controlnet.py` (training logic)
- `prepare_dataset.py` (dataset utilities)
- `config_template.py` (configuration)
- `inference.py` (for testing)

### Don't Modify Core Files in Colab

Edit configurations in `colab_train.py`, but avoid modifying the core modules (`train_controlnet.py`, etc.) directly in Colab. Make changes to files in Google Drive instead.

### Checkpoints Are Your Friend

Training can take hours. Checkpoints are saved automatically to Google Drive every N steps. If Colab disconnects, you can resume:

```python
config.resume_from_checkpoint = "/content/drive/MyDrive/AML/checkpoints/checkpoint-5000"
```

## 🆘 Troubleshooting

### "Repository not found" Error

**Problem**: `REPO_PATH` is incorrect

**Solution**: 
1. Verify folder exists in Drive at specified path
2. Check spelling and capitalization
3. Make sure you uploaded the **entire folder**, not just individual files

### "Missing required files" Error

**Problem**: Repository is incomplete

**Solution**: 
1. Re-upload the entire `controlnet_light` folder
2. Verify all files are present: `train_controlnet.py`, `prepare_dataset.py`, etc.

### "No training data found" Error

**Problem**: Dataset not prepared or path incorrect

**Solution**:
1. Check `config.dataset_base` path
2. Verify `images/` folder and `captions.csv` exist
3. Run dataset preparation section first

### Out of Memory

**Problem**: GPU memory exhausted

**Solution**:
```python
config.train_batch_size = 1  # Reduce to minimum
config.gradient_accumulation_steps = 16  # Increase to compensate
config.gradient_checkpointing = True  # Enable
config.use_8bit_adam = True  # Enable
```

## ✅ Verification Checklist

Before starting training:

- [ ] Entire `controlnet_light` repository uploaded to Google Drive
- [ ] Colab runtime set to GPU
- [ ] `REPO_PATH` points to correct location
- [ ] Output directories configured
- [ ] Dataset prepared or source paths set
- [ ] At least 5GB free space in Google Drive
- [ ] Configuration reviewed and customized

## 🎓 Next Steps

1. Read [QUICKSTART.md](QUICKSTART.md) for quick setup
2. See [README.md](README.md) for detailed documentation
3. Check [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for architecture
4. Review [config_template.py](config_template.py) for preset configurations

---

**Ready to train?** Follow the steps above and run `!python colab_train.py` in Google Colab! 🚀

