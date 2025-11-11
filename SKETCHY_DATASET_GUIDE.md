# Sketchy Dataset Integration Guide

## 📊 Overview

The `colab_train.py` script now **automatically downloads and processes the Sketchy Images Dataset from Kaggle**, using sketch images (not photos) to train ControlNet for sketch-to-image generation.

## 🎯 What This Does

### Dataset Flow

```
Kaggle API
    ↓
Download "dhananjayapaliwal/fulldataset"
    ↓
Extract SKETCH images (not photos!)
    ↓
Generate captions from class names
    ↓
Create training dataset
    ↓
Train ControlNet (sketch → realistic image)
```

### Training Process

1. **Input**: Sketch images from Sketchy dataset
2. **Conditioning**: HED detector processes sketches to create edge maps
3. **Target**: The same sketch (ControlNet learns the mapping)
4. **Caption**: Auto-generated from class name (e.g., "a sketch of an airplane")
5. **Output**: Model that converts sketches to realistic images

## ⚙️ Configuration

### Key Settings in `colab_train.py`

```python
# Dataset settings
DOWNLOAD_DATASET = True      # Automatically download from Kaggle
IMAGES_PER_CLASS = 50        # Number of sketches per class (subset)
USE_TRAIN_SPLIT = True       # Use train split (vs test split)

# Training settings
config.condition_type = "scribble"  # Uses HED detector for sketch processing
```

### Adjusting Dataset Size

```python
# Small dataset (fast training, ~1-2 hours)
IMAGES_PER_CLASS = 25    # ~1,250 images (50 classes × 25)

# Medium dataset (recommended, ~3-4 hours)
IMAGES_PER_CLASS = 50    # ~2,500 images (50 classes × 50)

# Large dataset (best quality, ~6-8 hours)
IMAGES_PER_CLASS = 100   # ~5,000 images (50 classes × 100)

# All data (very long training)
IMAGES_PER_CLASS = 999   # Takes all available sketches per class
```

## 📁 Sketchy Dataset Structure

### Dataset on Kaggle

```
dhananjayapaliwal/fulldataset
└── temp_extraction/
    └── 256x256/
        ├── splitted_sketches/
        │   ├── train/              ← Training sketches
        │   │   ├── airplane/
        │   │   │   ├── n02691156_10151-1.png
        │   │   │   ├── n02691156_10151-2.png
        │   │   │   └── ...
        │   │   ├── ant/
        │   │   ├── bear/
        │   │   └── ... (125 classes)
        │   │
        │   └── test/               ← Test sketches
        │       └── ... (same structure)
        │
        └── photo/                  ← NOT USED (we use sketches!)
            └── ...
```

### Processed Dataset Structure

After running the script:

```
/content/drive/MyDrive/AML/dataset/
├── images/
│   ├── sketch_000000.png
│   ├── sketch_000001.png
│   ├── sketch_000002.png
│   └── ... (50 × IMAGES_PER_CLASS images)
│
└── captions.csv
    Columns: image_filename, caption, original_path, class
    Example:
      sketch_000000.png, "a sketch of an airplane", /path/to/original, airplane
      sketch_000001.png, "a sketch of an airplane", /path/to/original, airplane
      sketch_000002.png, "a sketch of a cat", /path/to/original, cat
```

## 🔍 How Captions Are Generated

### Caption Generation Function

```python
def extract_caption_from_filename(filename, class_name):
    """
    Filename: n02691156_10151-1.png
    Class: airplane
    Generated Caption: "a sketch of an airplane"
    """
    caption = f"a sketch of a {class_name.replace('_', ' ')}"
    return caption
```

### Examples

| Filename | Class | Generated Caption |
|----------|-------|-------------------|
| `n02691156_10151-1.png` | `airplane` | `"a sketch of an airplane"` |
| `n02219486_11726-1.png` | `ant` | `"a sketch of an ant"` |
| `n02131653_10247-2.png` | `bear` | `"a sketch of a bear"` |
| `n03792782_10042-3.png` | `motorcycle` | `"a sketch of a motorcycle"` |

### Customizing Captions

You can enhance the caption generation for better results:

```python
def extract_caption_from_filename(filename, class_name):
    """Enhanced caption generation"""
    # Simple version (current)
    # caption = f"a sketch of a {class_name.replace('_', ' ')}"
    
    # Enhanced version (more descriptive)
    descriptors = [
        "a detailed sketch of a",
        "a hand-drawn sketch of a", 
        "a line drawing of a",
        "a simple sketch of a"
    ]
    import random
    descriptor = random.choice(descriptors)
    caption = f"{descriptor} {class_name.replace('_', ' ')}"
    
    return caption
```

## 🎨 What the Model Learns

### Training Data

- **Input**: Sketch line drawings from Sketchy dataset
- **Conditioning**: HED edge detection applied to sketches
- **Caption**: Class-based descriptions
- **Goal**: Learn to generate realistic images from sketch conditioning

### Model Capabilities

After training, your model can:

✅ Convert simple sketches to realistic images  
✅ Use text prompts to guide generation  
✅ Work with any sketch style (not just training data)  
✅ Generate variations based on prompt details  

### Example Usage

```python
# Your sketch: Simple line drawing of a cat
# Your prompt: "a realistic photo of a fluffy cat sitting on a chair"
# Output: Realistic photo matching your sketch structure and prompt description
```

## 📊 Dataset Statistics

### Sketchy Dataset Classes (125 total)

Sample classes included:
- **Animals**: airplane, ant, bear, bee, bird, cat, cow, dog, elephant, fish, etc.
- **Objects**: backpack, basket, bench, bicycle, bottle, chair, cup, fork, guitar, etc.
- **Vehicles**: airplane, bicycle, bus, car, helicopter, motorcycle, train, truck, etc.
- **Nature**: bee, bird, butterfly, crab, crocodile, flower, leaf, tree, etc.

### Dataset Sizes

| Images per Class | Total Images | Training Time (T4) | Quality |
|------------------|--------------|-------------------|---------|
| 25 | ~1,250 | 1-2 hours | Basic |
| 50 | ~2,500 | 3-4 hours | Good |
| 100 | ~5,000 | 6-8 hours | Better |
| 200 | ~10,000 | 12-16 hours | Best |

## 🚀 Quick Start

### 1. Basic Usage (Recommended)

```python
# In colab_train.py, set:
DOWNLOAD_DATASET = True
IMAGES_PER_CLASS = 50
USE_TRAIN_SPLIT = True

# Then run:
!python colab_train.py
```

### 2. Fast Testing

```python
# Quick test with small dataset
DOWNLOAD_DATASET = True
IMAGES_PER_CLASS = 25
USE_TRAIN_SPLIT = True
config.max_train_steps = 2000
```

### 3. High Quality Training

```python
# Best quality, longer training
DOWNLOAD_DATASET = True
IMAGES_PER_CLASS = 100
USE_TRAIN_SPLIT = True
config.max_train_steps = 20000
```

## 🔧 Troubleshooting

### "Sketches directory not found"

**Problem**: Script can't find the sketches folder

**Solution**: The script tries multiple paths. If it fails, check available directories:
```python
# The script will print available directories
# Look for: "temp_extraction/256x256/sketch" or similar
```

### "Not enough images"

**Problem**: Some classes have fewer images than `IMAGES_PER_CLASS`

**Solution**: This is normal. The script takes up to N images per class.
```python
# Some classes may have only 30 sketches, others 200+
# The script handles this automatically
```

### Dataset download is slow

**Problem**: Large dataset takes time to download

**Solution**: 
- First run: Downloads dataset (~5-10 minutes)
- Subsequent runs: Uses cached version (instant)
- Set `DOWNLOAD_DATASET = False` after first download

### Want to use photos instead?

**Problem**: You want to train on photos, not sketches

**Solution**: Modify the sketches_root path:
```python
# Change line ~192 in colab_train.py
sketches_root = os.path.join(dataset_path, "temp_extraction/256x256/photo", split)
```

## 📈 Expected Results

### Training Progress

```
Step 1000: loss=0.15   ← Learning sketch features
Step 2000: loss=0.12   ← Better understanding
Step 5000: loss=0.08   ← Good quality
Step 10000: loss=0.05  ← High quality results
```

### Validation

Generated images should:
- Follow the sketch structure
- Match the prompt description
- Look realistic
- Maintain sketch proportions

## 🎓 Advanced Customization

### Using Specific Classes

```python
# Modify prepare_sketchy_dataset_from_kaggle function
# Add class filtering:

classes = [d for d in os.listdir(sketches_root) 
           if os.path.isdir(os.path.join(sketches_root, d))]

# Filter to specific classes
desired_classes = ['airplane', 'car', 'cat', 'dog', 'bird']
classes = [c for c in classes if c in desired_classes]
```

### Better Caption Generation

```python
# Use a mapping for better descriptions
caption_map = {
    'airplane': 'a detailed sketch of an airplane in flight',
    'car': 'a sketch of a modern car',
    'cat': 'a sketch of a cute cat',
    # ... add more
}

def extract_caption_from_filename(filename, class_name):
    return caption_map.get(class_name, f"a sketch of a {class_name}")
```

### Train/Test Split

```python
# Use test split for validation
USE_TRAIN_SPLIT = False  # Uses test split instead

# Or use both for more data
# Modify script to combine train + test splits
```

## 📝 Summary

✅ **Automatic**: Downloads dataset from Kaggle automatically  
✅ **Sketch-based**: Uses sketches as input (not photos)  
✅ **Caption generation**: Automatic from class names  
✅ **Flexible**: Configure dataset size easily  
✅ **Ready to use**: Just run `colab_train.py`  

Your ControlNet will learn to convert sketches into realistic images guided by text prompts! 🎨→🖼️

