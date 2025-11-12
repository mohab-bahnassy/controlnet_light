# Troubleshooting Guide

## Common Issues and Solutions

### 1. ImportError: cannot import name 'cached_download' from 'huggingface_hub'

**Error Message:**
```
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```

**Cause:** You're using an old version of `diffusers` (0.25.0 or older) that requires `cached_download`, which was removed from newer versions of `huggingface_hub`.

**Solution (UPDATED - Now uses latest versions):**

The script now automatically installs the **latest compatible versions** which don't have this issue:

```bash
pip install --upgrade diffusers transformers accelerate huggingface_hub
```

**In colab_train.py:** This is now fixed automatically (line 60). The script installs modern versions that work together.

**If you still encounter this error:**

1. **Restart Colab runtime** (Runtime → Restart runtime)
2. **Run the script again** - it will install fresh versions
3. **Or manually upgrade**:
   ```python
   !pip install -q --upgrade pip
   !pip install -q --upgrade diffusers transformers accelerate huggingface_hub
   ```

**Why this works:** Newer versions of `diffusers` (>=0.27.0) don't use `cached_download` at all, so they work with any version of `huggingface_hub`.

---

### 2. Out of Memory (OOM) Errors

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solution:** Reduce memory usage:

```python
# In colab_train.py configuration section
config.train_batch_size = 1  # Reduce from 4 to 1
config.gradient_accumulation_steps = 16  # Increase to compensate
config.gradient_checkpointing = True
config.use_8bit_adam = True
config.enable_xformers = True
config.resolution = 256  # Reduce from 512 if still OOM
```

---

### 3. Repository Not Found Error

**Error Message:**
```
FileNotFoundError: Repository not found at /content/drive/MyDrive/controlnet_light
```

**Solution:** 
1. Verify the repository is uploaded to Google Drive
2. Check the path in `colab_train.py` line 52:
   ```python
   REPO_PATH = "/content/drive/MyDrive/controlnet_light"  # Update this path
   ```
3. Mount Google Drive first:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

---

### 4. Sketches Directory Not Found

**Error Message:**
```
FileNotFoundError: Sketches not found at ...
```

**Solution:** The script tries multiple paths automatically. If it still fails:

```python
# Check what's available in the dataset
import os
dataset_path = "/root/.cache/kagglehub/datasets/dhananjayapaliwal/fulldataset/versions/3"
base = os.path.join(dataset_path, "temp_extraction/256x256")
print("Available directories:")
for item in os.listdir(base):
    print(f"  - {item}")

# Manually set the correct path in colab_train.py (line ~192)
# sketches_root = os.path.join(dataset_path, "YOUR_CORRECT_PATH_HERE")
```

---

### 5. Kaggle Dataset Download Fails

**Error Message:**
```
Exception: Dataset download failed
```

**Solution:**

**Option 1: Authenticate Kaggle (Recommended)**
```python
# Upload your kaggle.json to Colab
from google.colab import files
files.upload()  # Upload kaggle.json

# Move to correct location
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

**Option 2: Manual Download**
1. Download dataset from Kaggle manually
2. Upload to Google Drive
3. Set `DOWNLOAD_DATASET = False` in colab_train.py
4. Update paths to point to your uploaded dataset

---

### 6. Training Loss is NaN or Very High

**Error Message:**
```
Step 100: loss=nan
```

**Solution:**

```python
# Reduce learning rate
config.learning_rate = 5e-6  # Lower from 1e-5

# Enable gradient clipping (already enabled by default)
config.max_grad_norm = 1.0

# Use mixed precision
config.mixed_precision = "fp16"

# Check dataset quality
# Verify images and captions are correct in:
# /content/drive/MyDrive/AML/dataset/captions.csv
```

---

### 7. xFormers Not Available

**Warning Message:**
```
⚠ xFormers not available
```

**Solution:**

```python
# Install xformers (usually works on Colab)
!pip install xformers

# If installation fails, disable in config:
config.enable_xformers = False
```

**Note:** Training will be slower but will still work.

---

### 8. Colab Session Disconnects

**Problem:** Long training interrupted by Colab timeout

**Solution:**

1. **Use checkpoints** (automatic every 1000 steps):
   ```python
   # Resume from checkpoint
   config.resume_from_checkpoint = "/content/drive/MyDrive/AML/checkpoints/checkpoint-5000"
   ```

2. **Prevent disconnection:**
   - Use Colab Pro for longer sessions
   - Run JavaScript in console to keep session alive:
     ```javascript
     function ClickConnect(){
       console.log("Working");
       document.querySelector("colab-connect-button").click()
     }
     setInterval(ClickConnect, 60000)
     ```

3. **Save more frequently:**
   ```python
   config.checkpointing_steps = 500  # Save every 500 steps instead of 1000
   ```

---

### 9. Dataset CSV Format Error

**Error Message:**
```
KeyError: 'image_filename' or 'caption'
```

**Solution:** CSV must have these columns:
```csv
image_filename,caption
sketch_000000.png,"a sketch of an airplane"
sketch_000001.png,"a sketch of a cat"
```

**Fix:** Re-run dataset preparation:
```python
# Delete existing dataset
!rm -rf /content/drive/MyDrive/AML/dataset

# Run colab_train.py again - it will recreate the dataset
```

---

### 10. Model Quality is Poor

**Problem:** Generated images look bad or don't follow sketches

**Solutions:**

1. **Train longer:**
   ```python
   config.max_train_steps = 20000  # Increase from 10000
   ```

2. **Use more data:**
   ```python
   IMAGES_PER_CLASS = 100  # Increase from 50
   ```

3. **Adjust learning rate:**
   ```python
   config.learning_rate = 5e-6  # Lower for stability
   ```

4. **Check training loss:**
   - Should decrease over time
   - Should be < 0.1 by step 10000
   - If stuck or increasing, restart with lower learning rate

---

### 11. GPU Not Being Used

**Problem:** Training is very slow

**Solution:**

1. **Check GPU is enabled:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
   ```

2. **Enable GPU in Colab:**
   - Runtime → Change runtime type → Hardware accelerator → GPU

3. **Verify in training:**
   - Look for "Using device: cuda" in output
   - If it says "cpu", restart runtime and enable GPU

---

### 12. Dependencies Installation Fails

**Error Message:**
```
ERROR: Failed building wheel for [package]
```

**Solution:**

```python
# Try installing dependencies one by one
!pip install --upgrade pip
!pip install 'huggingface_hub<0.20'
!pip install 'diffusers==0.25.0'
!pip install 'transformers==4.36.0'
!pip install 'accelerate==0.25.0'
!pip install controlnet_aux
!pip install xformers
!pip install bitsandbytes
!pip install kagglehub

# If specific package fails, skip it and continue
# Some packages (like xformers) are optional
```

---

### 13. Google Drive Space Full

**Error Message:**
```
OSError: [Errno 28] No space left on device
```

**Solution:**

1. **Check space:**
   ```python
   !df -h /content/drive
   ```

2. **Free up space:**
   - Delete old checkpoints
   - Reduce `IMAGES_PER_CLASS`
   - Clear Colab cache: `!rm -rf ~/.cache/*`

3. **Reduce checkpoint frequency:**
   ```python
   config.checkpointing_steps = 2000  # Save less frequently
   ```

---

## Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Import error | `!pip install 'huggingface_hub<0.20' 'diffusers==0.25.0'` |
| Out of memory | `config.train_batch_size = 1` |
| Repo not found | Check `REPO_PATH` in line 52 |
| No GPU | Runtime → Change runtime type → GPU |
| Dataset error | Delete dataset folder and re-run |
| Colab timeout | Use checkpoints, Colab Pro |
| Poor quality | Train longer, use more data |
| Slow training | Enable xformers, GPU |

---

## Getting Help

If your issue isn't listed here:

1. **Check error message carefully** - Often contains the solution
2. **Review configuration** - Verify all paths are correct
3. **Check GPU is enabled** - Most common issue
4. **Try with smaller dataset first** - `IMAGES_PER_CLASS = 25`
5. **Check Google Drive space** - Need 5-20GB free
6. **Read full error traceback** - Shows exactly where it failed

## Useful Debug Commands

```python
# Check GPU
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Check paths
import os
print(f"Repo exists: {os.path.exists('/content/drive/MyDrive/controlnet_light')}")
print(f"Dataset exists: {os.path.exists('/content/drive/MyDrive/AML/dataset')}")

# Check dataset
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/AML/dataset/captions.csv')
print(f"Dataset size: {len(df)}")
print(df.head())

# Check Drive space
!df -h /content/drive

# Check installed versions
!pip list | grep -E "diffusers|transformers|accelerate|huggingface_hub"
```

---

## Still Having Issues?

1. Review [QUICKSTART.md](QUICKSTART.md) for step-by-step instructions
2. Check [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) for detailed setup
3. See [SKETCHY_DATASET_GUIDE.md](SKETCHY_DATASET_GUIDE.md) for dataset issues
4. Read [README.md](README.md) for comprehensive documentation

**Most issues are fixed by:**
- Ensuring GPU runtime is enabled
- Using compatible package versions (huggingface_hub<0.20)
- Checking all paths are correct
- Having sufficient Google Drive space

