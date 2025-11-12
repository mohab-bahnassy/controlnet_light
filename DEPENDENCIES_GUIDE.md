# Dependencies Installation Guide

## 📦 What Gets Installed

When you run `colab_train.py`, **all required dependencies are automatically installed BEFORE** any imports from your repository modules. Here's the complete list:

## Installation Order in colab_train.py

### CELL 1: Mount Google Drive ✓
```python
from google.colab import drive
drive.mount('/content/drive')
```

### CELL 2: Install ALL Dependencies ✓ (Lines 38-72)

This happens **BEFORE** importing from `train_controlnet.py`, `prepare_dataset.py`, etc.

#### 1. Core ML Libraries (Line 55)
```bash
pip install torch torchvision numpy
```
- **torch** - PyTorch (deep learning framework)
- **torchvision** - Computer vision utilities
- **numpy** - Numerical computing

#### 2. Diffusers & HuggingFace (Line 60)
```bash
pip install --upgrade diffusers transformers accelerate huggingface_hub
```
- **diffusers** (latest >=0.27.0) - Stable Diffusion and ControlNet library
- **transformers** (latest >=4.40.0) - CLIP text encoder and tokenizer  
- **accelerate** (latest >=0.27.0) - Multi-GPU training utilities
- **huggingface_hub** (latest >=0.20.0) - HuggingFace Hub client

**Note:** Uses **latest versions** to avoid `cached_download` import error that occurred with older versions.

#### 3. ControlNet & Image Processing (Line 64)
```bash
pip install controlnet_aux opencv-python Pillow
```
- **controlnet_aux** - ControlNet auxiliary models (HED detector, etc.)
- **opencv-python** - Image processing (Canny edge detection, etc.)
- **Pillow** - Image I/O and manipulation

#### 4. Training Utilities (Lines 68-69)
```bash
pip install xformers datasets pandas tensorboard tqdm
pip install bitsandbytes kagglehub
```
- **xformers** - Memory-efficient attention (speeds up training)
- **datasets** - Dataset utilities
- **pandas** - CSV handling and data manipulation
- **tensorboard** - Training visualization
- **tqdm** - Progress bars
- **bitsandbytes** - 8-bit optimizer (saves GPU memory)
- **kagglehub** - Download datasets from Kaggle

### CELL 3: Import from Repository ✓ (Line 101+)

Only **AFTER** all dependencies are installed, we import:
```python
from train_controlnet import TrainingConfig, train
from prepare_dataset import prepare_from_sketchy_dataset
# ... etc
```

## ✅ Complete Dependency List

### Required by `train_controlnet.py`
- ✅ torch
- ✅ torch.nn.functional
- ✅ torch.utils.data (Dataset, DataLoader)
- ✅ torchvision.transforms
- ✅ PIL (Pillow)
- ✅ numpy
- ✅ pandas
- ✅ tqdm
- ✅ accelerate (Accelerator, set_seed)
- ✅ diffusers (StableDiffusionControlNetPipeline, ControlNetModel, etc.)
- ✅ diffusers.optimization (get_scheduler)
- ✅ transformers (CLIPTextModel, CLIPTokenizer)
- ✅ controlnet_aux (HEDdetector)
- ✅ cv2 (opencv-python)

### Required by `prepare_dataset.py`
- ✅ pandas
- ✅ PIL (Pillow)
- ✅ pathlib (standard library - no install needed)
- ✅ os (standard library)
- ✅ shutil (standard library)

### Required by `inference.py`
- ✅ torch
- ✅ PIL (Pillow)
- ✅ diffusers
- ✅ controlnet_aux
- ✅ cv2 (opencv-python)
- ✅ numpy

### Required by dataset preparation
- ✅ kagglehub
- ✅ pandas
- ✅ tqdm
- ✅ PIL (Pillow)

## 🔍 Verification

You can verify all dependencies are installed by running:

```python
# After running colab_train.py installation section, run this:

import torch
import torchvision
import numpy
import pandas
import PIL
import cv2
import transformers
import diffusers
import accelerate
import controlnet_aux
import tqdm
import kagglehub

print("✅ All dependencies successfully imported!")
print(f"PyTorch version: {torch.__version__}")
print(f"Diffusers version: {diffusers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 🚨 Common Issues

### Issue: "ModuleNotFoundError: No module named 'X'"

**Solution:** This shouldn't happen if running `colab_train.py` as a single cell. If it does:

1. Make sure you're running the **entire** `colab_train.py` file, not parts of it
2. Check that CELL 2 (installation) completes before CELL 3 (imports)
3. Manually install missing package:
   ```python
   !pip install [package_name]
   ```

### Issue: "ImportError: cannot import name 'cached_download'"

**Solution:** Fixed by using **latest versions** (line 60). Modern `diffusers` (>=0.27.0) doesn't use `cached_download`.

If you still get this error:
1. Restart Colab runtime (Runtime → Restart runtime)
2. Run the script again
3. The fresh installation will work correctly

### Issue: Installation fails for a specific package

**Solution:** Most packages are optional except core ones. You can skip problematic packages:
- **xformers** - Optional (training will be slower but works)
- **bitsandbytes** - Optional (can use regular Adam optimizer)

**Core packages (REQUIRED):**
- torch, diffusers, transformers, accelerate, controlnet_aux

## 📋 Installation Summary

When you run `colab_train.py` as a single cell:

1. ✅ **Lines 1-36**: Header and Drive mount
2. ✅ **Lines 38-72**: Install ALL dependencies
3. ✅ **Lines 74-89**: Verify repository exists
4. ✅ **Lines 91-101**: Add repo to Python path
5. ✅ **Line 101+**: Import from repository (train_controlnet, etc.)
6. ✅ **Rest of script**: Configuration, dataset prep, training

**Everything is installed before any imports!**

## 🎯 Key Points

✅ **All dependencies are installed automatically**
✅ **Installation happens BEFORE repository imports**
✅ **Version compatibility is handled** (huggingface_hub<0.20)
✅ **No manual installation needed**
✅ **Works when run as single cell in Colab**

## 🔧 Manual Installation (If Needed)

If for some reason you need to install dependencies manually:

```python
# Run this in a Colab cell BEFORE running colab_train.py

!pip install --upgrade pip
!pip install torch torchvision numpy
!pip install 'huggingface_hub<0.20' 'diffusers==0.25.0' 'transformers==4.36.0' 'accelerate==0.25.0'
!pip install controlnet_aux opencv-python Pillow
!pip install xformers datasets pandas tensorboard tqdm
!pip install bitsandbytes kagglehub
```

But **this is not necessary** - `colab_train.py` does this automatically!

## 📊 Estimated Installation Time

- **First run**: 3-5 minutes (downloading and installing packages)
- **Subsequent runs**: ~30 seconds (most packages cached)

## ✨ Summary

**You don't need to install anything manually!**

Just run `colab_train.py` and it will:
1. Install all dependencies with correct versions
2. Import from your repository modules
3. Download the dataset from Kaggle
4. Train the model
5. Save everything to Google Drive

All dependencies are handled automatically! 🚀

