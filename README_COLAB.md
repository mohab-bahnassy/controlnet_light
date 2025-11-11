# Running ControlNet Training on Google Colab

## 🎯 Quick Start

This project trains a ControlNet model by importing modular scripts from your Google Drive.

### Prerequisites

- Google Account with Google Drive access
- Google Colab (free or Pro)
- 5-20GB free space on Google Drive

### Setup (5 minutes)

1. **Upload this repository to Google Drive**
   ```
   /MyDrive/controlnet_light/
   ```

2. **Open `colab_train.py` in Google Colab**

3. **Set GPU runtime**
   - Runtime → Change runtime type → GPU

4. **Edit configuration** (lines 52, 95-97)
   ```python
   REPO_PATH = "/content/drive/MyDrive/controlnet_light"
   config.output_dir = "/content/drive/MyDrive/AML/controlnet_trained"
   ```

5. **Run the script**
   ```python
   !python colab_train.py
   ```

## 📂 Repository Structure

```
controlnet_light/                  ← Upload to Google Drive
│
├── colab_train.py                 ← START HERE (orchestration)
│   │
│   ├──> Imports from:
│   │    ├── train_controlnet.py  ← Training logic
│   │    ├── prepare_dataset.py   ← Dataset preparation
│   │    ├── config_template.py   ← Configuration presets
│   │    └── inference.py         ← Model testing
│   │
│   └──> Saves to Google Drive:
│        ├── controlnet_trained/  ← Final model
│        ├── checkpoints/         ← Intermediate saves
│        └── dataset/             ← Training data
│
├── train_controlnet.py            ← Core training implementation
├── prepare_dataset.py             ← Dataset utilities
├── config_template.py             ← Configuration classes
├── inference.py                   ← Inference utilities
│
├── QUICKSTART.md                  ← Quick start guide
├── README.md                      ← Full documentation
├── SETUP_INSTRUCTIONS.md          ← Detailed setup
└── PROJECT_OVERVIEW.md            ← Architecture overview
```

## 🔄 Workflow

```
┌─────────────────────────────────────────┐
│  1. Upload Repository to Google Drive   │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  2. Open colab_train.py in Colab        │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  3. Script mounts Drive & verifies repo │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  4. Imports train_controlnet.py         │
│     Imports prepare_dataset.py          │
│     Imports config_template.py          │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  5. Prepares dataset (if needed)        │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  6. Starts training                     │
│     - Saves checkpoints to Drive        │
│     - Displays progress                 │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  7. Final model saved to Drive          │
└─────────────────────────────────────────┘
```

## ⚙️ Configuration

### Basic (Edit in colab_train.py)

```python
# Repository location in Google Drive
REPO_PATH = "/content/drive/MyDrive/controlnet_light"

# Output directories
config.output_dir = "/content/drive/MyDrive/AML/controlnet_trained"
config.dataset_base = "/content/drive/MyDrive/AML/dataset"
config.checkpoint_dir = "/content/drive/MyDrive/AML/checkpoints"

# Training parameters
config.train_batch_size = 4          # Reduce if OOM
config.max_train_steps = 10000       # More = better quality
config.learning_rate = 1e-5          # Adjust for stability
config.condition_type = "scribble"   # or "canny"
```

### Advanced (Use Presets)

```python
from config_template import LowMemoryConfig, HighQualityConfig

# For free Colab
config = LowMemoryConfig()

# For Colab Pro / A100
config = HighQualityConfig()

# Customize
config.output_dir = "your/path"
config.max_train_steps = 15000
```

## 📊 What to Expect

### Timeline

- **Setup**: 5 minutes
- **Dependency installation**: 3-5 minutes
- **Dataset preparation**: 10-30 minutes (for 5000 images)
- **Training**: 2-6 hours (depends on steps and GPU)

### Resource Usage

- **GPU Memory**: 10-15GB (T4 can handle it)
- **Drive Space**: 5-20GB
- **Training Steps**: 10000 = ~3-4 hours on T4

### Checkpoints

Saved every 1000 steps to Google Drive:
```
checkpoints/
├── checkpoint-1000/
├── checkpoint-2000/
├── checkpoint-3000/
└── ...
```

Can resume if disconnected:
```python
config.resume_from_checkpoint = "/path/to/checkpoint-5000"
```

## 🧪 Testing Your Model

After training:

```python
from inference import load_model, generate_image

# Load trained model
pipe = load_model('/content/drive/MyDrive/AML/controlnet_trained')

# Generate image
generate_image(
    pipe=pipe,
    input_image_path='sketch.jpg',
    prompt='a realistic photo of a cat',
    output_path='result.png'
)
```

## 🔧 Troubleshooting

### "Repository not found"

**Problem**: Path incorrect or repository not uploaded

**Fix**:
1. Verify repository is at `/MyDrive/controlnet_light/`
2. Check `REPO_PATH` matches exactly
3. Ensure all files are present

### "Out of Memory"

**Problem**: GPU memory exhausted

**Fix**:
```python
config.train_batch_size = 1
config.gradient_accumulation_steps = 16
config.gradient_checkpointing = True
```

### "No training data found"

**Problem**: Dataset not prepared

**Fix**:
1. Verify dataset at `config.dataset_base`
2. Check `images/` folder exists
3. Verify `captions.csv` present

### Colab Disconnects

**Problem**: Session timeout

**Fix**:
- Checkpoints auto-save to Drive
- Resume with: `config.resume_from_checkpoint = "/path"`
- Consider Colab Pro for longer sessions

## 📈 Optimization Tips

### For Faster Training

```python
config.enable_xformers = True
config.mixed_precision = "fp16"
config.train_batch_size = 8  # If enough memory
```

### For Better Quality

```python
config.max_train_steps = 20000
config.learning_rate = 5e-6  # Lower, more stable
# Use more diverse training data (10000+ images)
```

### For Limited Memory

```python
from config_template import LowMemoryConfig
config = LowMemoryConfig()
```

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)** - Detailed setup guide
- **[README.md](README.md)** - Full documentation
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Architecture
- **[CHANGES.md](CHANGES.md)** - What changed in this version

## 🎓 Examples

### Minimal Example

```python
# After uploading repo to Drive and opening Colab

!python /content/drive/MyDrive/controlnet_light/colab_train.py
```

### Custom Configuration

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/controlnet_light')

from train_controlnet import TrainingConfig, train

config = TrainingConfig()
config.output_dir = "/content/drive/MyDrive/my_model"
config.dataset_base = "/content/drive/MyDrive/my_data"
config.max_train_steps = 5000

train(config)
```

### Using Presets

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/controlnet_light')

from config_template import LowMemoryConfig
from train_controlnet import train

config = LowMemoryConfig()
config.output_dir = "/content/drive/MyDrive/my_model"

train(config)
```

## ✅ Checklist

Before starting:

- [ ] Repository uploaded to Google Drive
- [ ] Colab runtime set to GPU
- [ ] REPO_PATH configured correctly
- [ ] Output paths set
- [ ] Dataset prepared or source data available
- [ ] 5+ GB free in Google Drive

## 🚀 Ready to Train!

1. Upload repository: ✓
2. Open `colab_train.py`: ✓
3. Set GPU runtime: ✓
4. Edit paths: ✓
5. Run: `!python colab_train.py`

**That's it!** Your ControlNet model will train and save automatically to Google Drive.

---

**Need help?** Check [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) for detailed guidance.

**Questions?** Review [README.md](README.md) for comprehensive documentation.

