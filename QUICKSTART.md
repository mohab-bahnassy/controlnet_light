# ControlNet Training - Quick Start Guide

> **📌 Important**: This project is designed to run on Google Colab. Upload the entire `controlnet_light` repository to your Google Drive before starting.

> **🎨 Dataset**: Automatically downloads the **Sketchy Images Dataset** from Kaggle and uses **sketch images** to train ControlNet for sketch-to-image generation.

## 🚀 Fastest Way to Get Started (Google Colab)

### Method 1: Using the Colab Orchestration Script (Easiest)

1. **Upload Repository to Google Drive**:
   - Download or clone this entire `controlnet_light` repository
   - Upload the folder to your Google Drive (e.g., `/MyDrive/controlnet_light/`)

2. **Open Google Colab**: Go to https://colab.research.google.com/

3. **Change Runtime to GPU**:
   - Click: `Runtime` → `Change runtime type`
   - Select: `GPU` (T4 or better)
   - Click: `Save`

4. **Upload and modify `colab_train.py`**:
   - Upload `colab_train.py` to Colab, or
   - Create a new cell and paste the contents of `colab_train.py`
   
5. **Edit the configuration** (modify these lines in the script):
   ```python
   # Line 52: Set your repo path in Google Drive
   REPO_PATH = "/content/drive/MyDrive/controlnet_light"
   
   # Lines 95-97: Set your output paths
   config.output_dir = "/content/drive/MyDrive/AML/controlnet_trained"
   config.dataset_base = "/content/drive/MyDrive/AML/dataset"
   config.checkpoint_dir = "/content/drive/MyDrive/AML/checkpoints"
   
   # Lines 100-102: Dataset settings (Sketchy dataset)
   DOWNLOAD_DATASET = True     # Auto-download from Kaggle
   IMAGES_PER_CLASS = 50       # 50 sketches per class (recommended)
   USE_TRAIN_SPLIT = True      # Use training split
   ```

6. **Run the script**:
   ```python
   !python colab_train.py
   ```
   Or simply run all cells if you pasted it into the notebook

7. **The script will automatically**:
   - Mount Google Drive ✓
   - Download Sketchy dataset from Kaggle (~5-10 min) ✓
   - Extract sketch images (not photos!) ✓
   - Generate captions from class names ✓
   - Train ControlNet on sketches ✓
   - Save checkpoints to Drive ✓

8. **Wait for training** (2-6 hours depending on dataset size)

9. **Your trained model will be in Google Drive** at the path you specified!
   - Model converts sketches → realistic images
   - Trained on Sketchy dataset classes (airplane, cat, car, etc.)

---

### Method 2: Using Individual Scripts (More Control)

This method assumes you've uploaded the repository to Google Drive.

#### Step 1: Mount Drive and Setup Path

```python
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/controlnet_light')
```

#### Step 2: Prepare Your Dataset

```python
from prepare_dataset import prepare_from_sketchy_dataset

prepare_from_sketchy_dataset(
    photo_root="/path/to/photos",
    csv_path="/path/to/captions.csv",
    output_base_dir="/content/drive/MyDrive/AML/dataset",
    max_images=5000
)
```

#### Step 3: Configure Training

```python
from train_controlnet import TrainingConfig

config = TrainingConfig()
config.output_dir = "/content/drive/MyDrive/AML/controlnet_trained"
config.dataset_base = "/content/drive/MyDrive/AML/dataset"
config.max_train_steps = 10000
```

#### Step 4: Train

```python
from train_controlnet import train

train(config)
```

---

## 📁 Dataset Structure (Automatically Created)

The script automatically downloads and prepares the Sketchy dataset:

```
dataset/
├── images/
│   ├── sketch_000000.png     ← Sketch from "airplane" class
│   ├── sketch_000001.png     ← Sketch from "airplane" class
│   ├── sketch_000002.png     ← Sketch from "cat" class
│   └── ... (IMAGES_PER_CLASS × number of classes)
│
└── captions.csv
```

Example `captions.csv` (auto-generated):
```csv
image_filename,caption,original_path,class
sketch_000000.png,"a sketch of an airplane",/path/to/original,airplane
sketch_000001.png,"a sketch of an airplane",/path/to/original,airplane
sketch_000002.png,"a sketch of a cat",/path/to/original,cat
```

**Note**: You don't need to create this manually! The script does everything automatically.

---

## 🎯 Using Your Trained Model

After training completes:

```python
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import HEDdetector

# Load your trained model
controlnet = ControlNetModel.from_pretrained(
    "/content/drive/MyDrive/AML/controlnet_trained",
    torch_dtype=torch.float16
)

# Create pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.to("cuda")

# Generate conditioning image
hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
input_image = Image.open("your_sketch.jpg")
control_image = hed_detector(input_image, scribble=True)

# Generate result
result = pipe(
    prompt="a realistic photo of a cat",
    image=control_image,
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]

result.save("output.png")
```

---

## ⚡ Quick Tips

### For Faster Training
- Use Colab Pro (A100 or V100 GPU)
- Enable all optimizations in config
- Start with smaller dataset (1000-2000 images)

### For Better Quality
- Train longer (20000+ steps)
- Use more diverse training data
- Use descriptive captions
- Fine-tune from pretrained ControlNet

### For Limited Memory (Free Colab)
```python
config.train_batch_size = 2
config.gradient_accumulation_steps = 8
config.use_8bit_adam = True
config.gradient_checkpointing = True
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce batch size to 1-2 |
| Training too slow | Enable xformers, use GPU runtime |
| No data found | Check paths in configuration |
| Poor results | Train longer, use more data |
| Colab disconnects | Checkpoints save automatically to Drive |

---

## 📊 Recommended Settings by Use Case

### Quick Test (1 hour)
```python
max_train_steps = 2000
train_batch_size = 4
dataset_size = 1000 images
```

### Standard Training (3-4 hours)
```python
max_train_steps = 10000
train_batch_size = 4
dataset_size = 3000-5000 images
```

### High Quality (6+ hours)
```python
max_train_steps = 20000+
train_batch_size = 4-8
dataset_size = 10000+ images
```

---

## 📞 Need Help?

1. Check the full [README.md](README.md) for detailed documentation
2. Make sure you're using GPU runtime in Colab
3. Verify your dataset structure matches the required format
4. Check that paths in configuration are correct

---

## 🎨 Sketchy Dataset Configuration

The script automatically uses the Sketchy Images Dataset from Kaggle. You can customize:

### Dataset Size

| IMAGES_PER_CLASS | Total Images | Training Time | Use Case |
|------------------|--------------|---------------|----------|
| 25 | ~1,250 | 1-2 hours | Quick test |
| 50 | ~2,500 | 3-4 hours | **Recommended** |
| 100 | ~5,000 | 6-8 hours | High quality |
| 200 | ~10,000 | 12-16 hours | Best results |

### Configuration Options

```python
# In colab_train.py (lines 100-102)

# Quick test
IMAGES_PER_CLASS = 25

# Recommended for good results
IMAGES_PER_CLASS = 50

# High quality training
IMAGES_PER_CLASS = 100

# Use training or test split
USE_TRAIN_SPLIT = True  # or False for test split

# Download dataset or use cached
DOWNLOAD_DATASET = True  # False if already downloaded
```

### What the Model Learns

Your trained ControlNet will:
- ✅ Convert sketch drawings → realistic images
- ✅ Understand object structures from sketches
- ✅ Generate images based on text prompts
- ✅ Work with classes from Sketchy dataset (airplane, cat, car, dog, etc.)

**See [SKETCHY_DATASET_GUIDE.md](SKETCHY_DATASET_GUIDE.md) for detailed information.**

---

## ✅ Checklist Before Training

- [ ] Google Colab runtime set to GPU
- [ ] Google Drive mounted
- [ ] Repository uploaded to Drive
- [ ] Configuration paths are correct
- [ ] `IMAGES_PER_CLASS` set appropriately
- [ ] Sufficient time for training (2+ hours)

**Ready? Run the training script and monitor the progress!** 🚀

The dataset will be automatically downloaded and prepared from Kaggle!

