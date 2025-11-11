# ControlNet Lightweight Training

Train a lightweight ControlNet model on Google Colab with your own dataset.

## 📋 Overview

This project provides scripts to:
1. **Prepare datasets** for ControlNet training
2. **Train a lightweight ControlNet model** from scratch or fine-tune existing models
3. **Save trained models** to Google Drive for later use

## 🚀 Quick Start (Google Colab)

### Step 1: Upload Scripts to Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `train_controlnet.py` and `prepare_dataset.py` to your Colab session
3. Or clone this repository:

```python
!git clone https://github.com/yourusername/controlnet_light.git
%cd controlnet_light
```

### Step 2: Prepare Your Dataset

#### Option A: Use the Sketchy Dataset (from your notebook)

```python
# Run the dataset preparation script
!python prepare_dataset.py

# Or do it manually in a cell:
from prepare_dataset import prepare_from_sketchy_dataset

prepare_from_sketchy_dataset(
    photo_root="/content/drive/Shareddrives/AML/Sketchy/Rendered Images/256x256/photo",
    csv_path="/content/drive/Shareddrives/AML/path_caption_pairs.csv",
    output_base_dir="/content/drive/MyDrive/AML/dataset",
    max_images=5000  # Start with 5000 images for faster training
)
```

#### Option B: Use Your Own Dataset

```python
from prepare_dataset import prepare_from_existing_csv

prepare_from_existing_csv(
    existing_csv_path="/path/to/your/captions.csv",
    image_path_column="image_path",  # Column name with image paths
    caption_column="caption",         # Column name with captions
    output_base_dir="/content/drive/MyDrive/AML/dataset",
    max_images=5000
)
```

Your prepared dataset will have this structure:
```
dataset/
├── images/           # Training images
├── conditions/       # Optional: Pre-generated conditioning images
└── captions.csv      # Image filenames and captions
```

### Step 3: Configure Training

Edit the configuration in `train_controlnet.py` or override in your notebook:

```python
from train_controlnet import TrainingConfig, train

# Modify configuration
config = TrainingConfig()

# Essential paths
config.output_dir = "/content/drive/MyDrive/AML/controlnet_trained"
config.dataset_base = "/content/drive/MyDrive/AML/dataset"
config.checkpoint_dir = "/content/drive/MyDrive/AML/checkpoints"

# Training hyperparameters
config.resolution = 512
config.train_batch_size = 4  # Reduce if out of memory
config.learning_rate = 1e-5
config.max_train_steps = 10000  # Adjust based on dataset size
config.checkpointing_steps = 1000
config.validation_steps = 500

# Optimization for limited memory
config.gradient_accumulation_steps = 4
config.gradient_checkpointing = True
config.use_8bit_adam = True
config.enable_xformers = True

# Conditioning type
config.condition_type = "scribble"  # Options: "scribble", "canny"

# Start training
train(config)
```

### Step 4: Train the Model

Simply run the training script:

```python
!python train_controlnet.py
```

Or run in a notebook cell for more control:

```python
from train_controlnet import train, TrainingConfig

config = TrainingConfig()
# ... configure as needed ...

train(config)
```

## 🎛️ Configuration Options

### Model Configuration

- `pretrained_model_name`: Base Stable Diffusion model (default: "runwayml/stable-diffusion-v1-5")
- `controlnet_model_name`: Starting ControlNet checkpoint (default: "lllyasviel/sd-controlnet-scribble")
- `resolution`: Training resolution (default: 512)

### Training Hyperparameters

- `train_batch_size`: Batch size per GPU (default: 4)
- `gradient_accumulation_steps`: Accumulate gradients for effective larger batch (default: 4)
- `learning_rate`: Learning rate (default: 1e-5)
- `max_train_steps`: Total training steps (default: 10000)
- `checkpointing_steps`: Save checkpoint every N steps (default: 1000)
- `validation_steps`: Run validation every N steps (default: 500)

### Memory Optimization

- `use_8bit_adam`: Use 8-bit Adam optimizer (saves memory)
- `enable_xformers`: Enable memory-efficient attention
- `gradient_checkpointing`: Trade compute for memory
- `mixed_precision`: "fp16" or "bf16" for faster training

### Conditioning Types

- `"scribble"`: HED-based scribble conditioning (default)
- `"canny"`: Canny edge detection conditioning

## 📊 Training Tips

### For Limited Memory (Free Colab)

```python
config.train_batch_size = 2
config.gradient_accumulation_steps = 8
config.gradient_checkpointing = True
config.use_8bit_adam = True
config.enable_xformers = True
config.mixed_precision = "fp16"
```

### For Colab Pro / A100

```python
config.train_batch_size = 8
config.gradient_accumulation_steps = 2
config.resolution = 512
config.mixed_precision = "bf16"  # Better quality on newer GPUs
```

### Dataset Size Recommendations

- **Small dataset (< 1000 images)**: 5000-10000 steps
- **Medium dataset (1000-5000 images)**: 10000-20000 steps
- **Large dataset (> 5000 images)**: 20000+ steps

### Learning Rate Guidelines

- **Fine-tuning pretrained ControlNet**: 1e-5 to 5e-5
- **Training from scratch**: 1e-4 to 5e-4

## 📁 Output Structure

After training, you'll have:

```
/content/drive/MyDrive/AML/
├── controlnet_trained/           # Final model
│   ├── config.json
│   ├── diffusion_pytorch_model.bin
│   └── training_config.json
├── checkpoints/                  # Intermediate checkpoints
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   └── ...
└── dataset/                      # Your training data
    ├── images/
    └── captions.csv
```

## 🔍 Using Your Trained Model

After training, load your model for inference:

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

# Load your trained ControlNet
controlnet = ControlNetModel.from_pretrained(
    "/content/drive/MyDrive/AML/controlnet_trained",
    torch_dtype=torch.float16
)

# Create pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.to("cuda")

# Generate image
from PIL import Image
from controlnet_aux import HEDdetector

# Load conditioning image
hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
input_image = Image.open("your_sketch.jpg")
control_image = hed_detector(input_image, scribble=True)

# Generate
result = pipe(
    prompt="a realistic photo of a cat",
    image=control_image,
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]

result.show()
```

## 🐛 Troubleshooting

### Out of Memory Errors

1. Reduce `train_batch_size` to 2 or 1
2. Increase `gradient_accumulation_steps` to compensate
3. Enable all memory optimizations:
   - `gradient_checkpointing = True`
   - `use_8bit_adam = True`
   - `enable_xformers = True`
4. Reduce `resolution` to 256 or 384

### "No training data found" Error

- Verify images are in `{dataset_base}/images/`
- Verify `captions.csv` exists at `{dataset_base}/captions.csv`
- Check CSV has columns: `image_filename`, `caption`

### Training is Very Slow

- Enable xformers: `config.enable_xformers = True`
- Use mixed precision: `config.mixed_precision = "fp16"`
- Increase batch size if you have memory
- Use Colab Pro for faster GPUs

### Model Quality is Poor

- Train for more steps (`max_train_steps`)
- Use a larger dataset (> 5000 images)
- Adjust learning rate (try 5e-6 or 2e-5)
- Ensure captions are descriptive and accurate

## 📝 Advanced: Resume Training

To resume from a checkpoint:

```python
config.resume_from_checkpoint = "/content/drive/MyDrive/AML/checkpoints/checkpoint-5000"
train(config)
```

## 🔗 Resources

- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [ControlNet Paper](https://arxiv.org/abs/2302.05543)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- ControlNet by Lvmin Zhang
- Hugging Face Diffusers team
- Stability AI for Stable Diffusion
