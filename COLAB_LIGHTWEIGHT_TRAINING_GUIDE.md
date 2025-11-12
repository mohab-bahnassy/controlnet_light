# Colab Lightweight Training Guide

## Overview

The `colab_train.py` script has been modified to train **all 4 lightweight ControlNet variants** instead of the standard ControlNet model.

## What Changed

### Before
- Trained standard ControlNet using HuggingFace Diffusers approach
- Single model output
- Used `train_controlnet.py` training function

### After
- Trains **all 4 lightweight ControlNet variants** sequentially
- Uses PyTorch Lightning approach (like `tutorial_train_light.py`)
- Saves each model to separate subdirectories
- Tracks and displays all model paths at the end

## The 4 Lightweight Models

### 1. Light ControlNet
- **Description**: 50% reduced channel width
- **Config**: `./models/cldm_v15_light.yaml`
- **Batch Size**: 8
- **Best For**: Balanced speed/quality (recommended default)
- **Output**: `/content/drive/MyDrive/AML/controlnet_trained/controlnet_light/`

### 2. Tiny ControlNet
- **Description**: 25% channel width
- **Config**: `./models/cldm_v15_tiny.yaml`
- **Batch Size**: 12
- **Best For**: Maximum speed, resource-constrained environments
- **Output**: `/content/drive/MyDrive/AML/controlnet_trained/controlnet_tiny/`

### 3. Efficient ControlNet
- **Description**: Depthwise separable convolutions
- **Config**: `./models/cldm_v15_efficient.yaml`
- **Batch Size**: 8
- **Best For**: Parameter efficiency with depthwise separable convolutions
- **Output**: `/content/drive/MyDrive/AML/controlnet_trained/controlnet_efficient/`

### 4. SimpleCNN ControlNet
- **Description**: Simple CNN blocks without attention
- **Config**: `./models/cldm_v15_simple_cnn.yaml`
- **Batch Size**: 10
- **Best For**: Maximum efficiency, simple control tasks
- **Output**: `/content/drive/MyDrive/AML/controlnet_trained/controlnet_simple_cnn/`

## Key Configuration Variables

Located in **CELL 4** of `colab_train.py`:

```python
# Modify these for your setup
BASE_OUTPUT_DIR = "/content/drive/MyDrive/AML/controlnet_trained"
DATASET_BASE = "/content/drive/MyDrive/AML/dataset"

# Dataset settings
DOWNLOAD_DATASET = True
IMAGES_PER_CLASS = 50
USE_TRAIN_SPLIT = True

# Training settings
MAX_EPOCHS = 5  # Number of epochs per model
LEARNING_RATE = 1e-5
LOGGER_FREQ = 300

# Batch sizes (optimized for each model)
BATCH_SIZES = {
    'light': 8,
    'tiny': 12,
    'efficient': 8,
    'simple_cnn': 10
}

# Initial checkpoint (required)
RESUME_PATH = './models/control_sd15_ini.ckpt'
```

## Output Structure

After training completes, you'll have:

```
/content/drive/MyDrive/AML/controlnet_trained/
├── controlnet_light/
│   ├── light_final.ckpt
│   └── [training logs]
├── controlnet_tiny/
│   ├── tiny_final.ckpt
│   └── [training logs]
├── controlnet_efficient/
│   ├── efficient_final.ckpt
│   └── [training logs]
└── controlnet_simple_cnn/
    ├── simple_cnn_final.ckpt
    └── [training logs]
```

## Trained Model Paths Dictionary

At the end of training, the script stores all model paths in `TRAINED_MODEL_PATHS`:

```python
TRAINED_MODEL_PATHS = {
    'light': {
        'path': '/content/drive/.../controlnet_light/light_final.ckpt',
        'description': 'Light (50% channels)',
        'parameters': 45234567,  # Example count
        'batch_size': 8
    },
    'tiny': {
        'path': '/content/drive/.../controlnet_tiny/tiny_final.ckpt',
        'description': 'Tiny (25% channels)',
        'parameters': 12345678,
        'batch_size': 12
    },
    # ... etc for efficient and simple_cnn
}
```

## Training Flow

1. **CELL 1**: Mount Google Drive
2. **CELL 2**: Install dependencies (commented out - uncomment if needed)
3. **CELL 3**: Setup repository path
4. **CELL 4**: Configure training parameters
5. **CELL 5**: Download and prepare Sketchy dataset from Kaggle
6. **CELL 6**: Train all 4 lightweight models sequentially
7. **CELL 7**: Display summary and testing instructions

## Sequential Training

The script trains each model one after another:

```
Training Light → Save → Training Tiny → Save → Training Efficient → Save → Training SimpleCNN → Save
```

If any model fails:
- Error is caught and logged
- Training continues with next model
- Final summary shows which models succeeded

## Error Handling

- If a config file is missing, that model is skipped
- If training is interrupted (KeyboardInterrupt), it moves to next model
- If any error occurs, it's logged and training continues
- Final summary shows which models completed successfully

## How to Use

### 1. Upload Repository to Google Drive
```
/content/drive/MyDrive/controlnet_light/
```

### 2. Open in Google Colab
- Upload `colab_train.py` to Colab
- Or open directly from Drive

### 3. Set GPU Runtime
- Runtime → Change runtime type → GPU

### 4. Modify Configuration (CELL 4)
- Set output directories
- Adjust `IMAGES_PER_CLASS` (50 recommended for testing)
- Adjust `MAX_EPOCHS` (5 is a good start)

### 5. Run All Cells
- The script will:
  - Mount Drive
  - Setup paths
  - Download dataset
  - Train all 4 models
  - Display results

## Training Time Estimates

Assuming:
- Dataset: 50 images per class, ~125 classes = ~6,250 images
- GPU: T4 (free Colab tier)
- MAX_EPOCHS: 5

Approximate training times per model:
- **Light**: ~2-3 hours
- **Tiny**: ~1-2 hours
- **Efficient**: ~2-3 hours
- **SimpleCNN**: ~1.5-2 hours

**Total**: ~7-10 hours for all 4 models

## Benefits of This Approach

1. **Compare Models**: Train all variants in one session
2. **Automatic Tracking**: All paths saved in `TRAINED_MODEL_PATHS`
3. **Optimized Batch Sizes**: Each model uses appropriate batch size
4. **Error Resilient**: One failure doesn't stop the others
5. **Organized Output**: Each model in separate directory
6. **Resource Efficient**: Lightweight models use less VRAM

## Testing Your Models

After training, use this code to test any model:

```python
from cldm.model import create_model, load_state_dict

# Choose a model
model_path = TRAINED_MODEL_PATHS['light']['path']
config_path = './models/cldm_v15_light.yaml'

# Load model
model = create_model(config_path)
model.load_state_dict(load_state_dict(model_path, location='cpu'))
model.eval()
model = model.cuda()

# Run inference (add your inference code)
```

## Troubleshooting

### "Config file not found"
- Ensure all YAML files exist in `./models/` directory
- Check `cldm_v15_light.yaml`, `cldm_v15_tiny.yaml`, etc.

### "Resume checkpoint not found"
- Download `control_sd15_ini.ckpt` from ControlNet repo
- Place in `./models/` directory

### Out of Memory
- Reduce batch sizes in `BATCH_SIZES` dict
- Reduce `IMAGES_PER_CLASS`
- Use smaller image resolution

### Training too slow
- Reduce `MAX_EPOCHS`
- Reduce `IMAGES_PER_CLASS`
- Use only 1-2 models instead of all 4

## Customization

### Train only specific models
Comment out unwanted models in CELL 4:

```python
LIGHTWEIGHT_MODELS = {
    'light': { ... },
    # 'tiny': { ... },  # Commented out
    # 'efficient': { ... },  # Commented out
    # 'simple_cnn': { ... },  # Commented out
}
```

### Change model order
Reorder the dictionary entries in `LIGHTWEIGHT_MODELS`

### Custom architectures
Add new entries to `LIGHTWEIGHT_MODELS` with your own YAML configs

## Accessing Model Paths

At the end of training (CELL 7), the script prints all model paths:

```
SAVED MODEL SUMMARY
===========================================================
Base output directory: /content/drive/MyDrive/AML/controlnet_trained
Dataset location: /content/drive/MyDrive/AML/dataset
Dataset info: /content/drive/MyDrive/AML/dataset/captions.csv

All model checkpoints:
  light: /content/drive/.../controlnet_light/light_final.ckpt
  tiny: /content/drive/.../controlnet_tiny/tiny_final.ckpt
  efficient: /content/drive/.../controlnet_efficient/efficient_final.ckpt
  simple_cnn: /content/drive/.../controlnet_simple_cnn/simple_cnn_final.ckpt
```

You can copy these paths for inference or further fine-tuning.

## Parameter Comparison

The script automatically counts and displays parameters for each model:

```
Model Performance Characteristics:

LIGHT: Light (50% channels)
  Parameters: ~45M
  Trained with batch size: 8
  Best for: Balanced speed/quality, recommended default

TINY: Tiny (25% channels)
  Parameters: ~12M
  Trained with batch size: 12
  Best for: Maximum speed, resource-constrained environments

EFFICIENT: Efficient (Depthwise Separable)
  Parameters: ~35M
  Trained with batch size: 8
  Best for: Good efficiency with depthwise separable convolutions

SIMPLE_CNN: SimpleCNN (No Attention)
  Parameters: ~25M
  Trained with batch size: 10
  Best for: Maximum efficiency, simple control tasks
```

## Notes

- All models share the same dataset
- Dataset is prepared once, used by all models
- Models train sequentially (not parallel) to avoid memory issues
- Each model gets its own output directory
- Progress is displayed for each model separately
- Final summary compares all trained models

## Support

For issues or questions:
1. Check the training logs in each model's directory
2. Review error messages in Colab output
3. Verify all paths are correct
4. Ensure GPU runtime is enabled
5. Check that all required files exist in repository

