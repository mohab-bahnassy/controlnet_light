# -*- coding: utf-8 -*-
"""
ControlNet Training - Colab Orchestration Script with Sketchy Dataset
This script:
  1. Downloads the Sketchy Images Dataset from Kaggle
  2. Uses SKETCH images (not photos) as training input
  3. Extracts captions from class names
  4. Trains ControlNet on sketch-to-image generation
  5. Saves model to Google Drive

Dataset: Sketches → ControlNet → Generated Images
Captions: Automatically generated from class names (e.g., "a sketch of an airplane")

To use:
1. Upload the entire controlnet_light folder to Google Drive (e.g., /content/drive/MyDrive/controlnet_light/)
2. Open this script in Google Colab
3. Change runtime to GPU (Runtime → Change runtime type → GPU)
4. Modify REPO_PATH and configuration below (especially IMAGES_PER_CLASS)
5. Run the script
6. Dataset will be automatically downloaded from Kaggle
7. Model will be trained and saved to Google Drive

Note: This uses sketches as input, not photos. The ControlNet will learn to convert
sketches into realistic images based on the caption descriptions.
"""

# ============================================================
# CELL 1: Mount Google Drive
# ============================================================
print("="*60)
print("CONTROLNET TRAINING - GOOGLE COLAB")
print("="*60)
print("\nMounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')
print("✓ Google Drive mounted!")

# ============================================================
# CELL 2: Install Dependencies
# ============================================================
print("\n" + "="*60)
print("INSTALLING DEPENDENCIES")
print("="*60)
print("This may take a few minutes...")

import os
import sys

# Install from requirements.txt if available, otherwise install manually
os.system("pip install -q diffusers==0.25.0 transformers==4.36.0 accelerate==0.25.0")
os.system("pip install -q controlnet_aux xformers datasets pandas torchvision opencv-python Pillow tensorboard")
os.system("pip install -q bitsandbytes kagglehub")
print("✓ All dependencies installed!")

# ============================================================
# CELL 3: Setup Repository Path
# ============================================================
print("\n" + "="*60)
print("REPOSITORY SETUP")
print("="*60)

# === MODIFY THIS PATH ===
# Path to where you uploaded the controlnet_light repository in Google Drive
REPO_PATH = "/content/drive/MyDrive/controlnet_light"

# Add repository to Python path
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
    print(f"✓ Added {REPO_PATH} to Python path")

# Verify repository exists
if not os.path.exists(REPO_PATH):
    print(f"\n❌ ERROR: Repository not found at {REPO_PATH}")
    print("Please upload the controlnet_light folder to your Google Drive")
    print("Expected structure:")
    print("  /content/drive/MyDrive/controlnet_light/")
    print("    ├── train_controlnet.py")
    print("    ├── prepare_dataset.py")
    print("    ├── config_template.py")
    print("    └── ...")
    raise FileNotFoundError(f"Repository not found at {REPO_PATH}")

# Verify key files exist
required_files = ['train_controlnet.py', 'prepare_dataset.py', 'config_template.py']
missing_files = [f for f in required_files if not os.path.exists(os.path.join(REPO_PATH, f))]
if missing_files:
    print(f"\n❌ ERROR: Missing required files: {missing_files}")
    print(f"Please ensure the complete repository is at {REPO_PATH}")
    raise FileNotFoundError(f"Missing files: {missing_files}")

print("✓ Repository found and verified")

# ============================================================
# CELL 4: Configuration - MODIFY THESE PATHS
# ============================================================
print("\n" + "="*60)
print("CONFIGURATION")
print("="*60)

# Import configuration module
from train_controlnet import TrainingConfig

# Create configuration instance
config = TrainingConfig()

# === MODIFY THESE PATHS FOR YOUR SETUP ===
config.output_dir = "/content/drive/MyDrive/AML/controlnet_trained"
config.dataset_base = "/content/drive/MyDrive/AML/dataset"
config.checkpoint_dir = "/content/drive/MyDrive/AML/checkpoints"

# Dataset preparation settings (Sketchy dataset from Kaggle)
DOWNLOAD_DATASET = True  # Set to False if already downloaded
IMAGES_PER_CLASS = 50    # Number of images to use from each class (subset)
USE_TRAIN_SPLIT = True   # True for train split, False for test split

# === TRAINING HYPERPARAMETERS ===
config.resolution = 512
config.train_batch_size = 4  # Reduce to 2 or 1 if OOM
config.gradient_accumulation_steps = 4
config.learning_rate = 1e-5
config.max_train_steps = 10000
config.checkpointing_steps = 1000
config.validation_steps = 500

# === OPTIMIZATION ===
config.use_8bit_adam = True
config.enable_xformers = True
config.gradient_checkpointing = True
config.mixed_precision = "fp16"

# === MODEL & CONDITIONING ===
config.pretrained_model_name = "runwayml/stable-diffusion-v1-5"
config.controlnet_model_name = "lllyasviel/sd-controlnet-scribble"
config.condition_type = "scribble"  # "scribble" or "canny"

# === OTHER ===
config.seed = 42

# Display configuration
print(f"\nConfiguration Summary:")
print(f"  Output: {config.output_dir}")
print(f"  Dataset: {config.dataset_base}")
print(f"  Resolution: {config.resolution}")
print(f"  Batch size: {config.train_batch_size}")
print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
print(f"  Effective batch size: {config.train_batch_size * config.gradient_accumulation_steps}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Max steps: {config.max_train_steps}")
print(f"  Condition type: {config.condition_type}")
print("="*60)

# ============================================================
# CELL 5: Download and Prepare Sketchy Dataset from Kaggle
# ============================================================
print("\n" + "="*60)
print("DATASET PREPARATION - SKETCHY DATASET")
print("="*60)

import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

def extract_caption_from_filename(filename, class_name):
    """
    Extract caption from Sketchy dataset filename.
    Filenames are like: n02691156_10151-1.png
    We use the class name as the caption.
    """
    # You can make this more sophisticated, but simple captions work well
    # The class name itself is a good description
    caption = f"a sketch of a {class_name.replace('_', ' ')}"
    return caption

def prepare_sketchy_dataset_from_kaggle(
    output_base,
    images_per_class=50,
    use_train=True,
    download=True
):
    """
    Download and prepare Sketchy dataset from Kaggle.
    Uses sketch images (not photos) as training data.
    Captions are generated from class names.
    """
    
    # Download dataset from Kaggle
    if download:
        print("Downloading Sketchy dataset from Kaggle...")
        print("This may take several minutes...")
        import kagglehub
        dataset_path = kagglehub.dataset_download("dhananjayapaliwal/fulldataset")
        print(f"✓ Dataset downloaded to: {dataset_path}")
    else:
        # Assume it's already downloaded
        dataset_path = "/root/.cache/kagglehub/datasets/dhananjayapaliwal/fulldataset/versions/3"
        if not os.path.exists(dataset_path):
            print("⚠ Dataset not found, downloading...")
            import kagglehub
            dataset_path = kagglehub.dataset_download("dhananjayapaliwal/fulldataset")
    
    # Construct path to sketches (not photos!)
    split = "train" if use_train else "test"
    sketches_root = os.path.join(dataset_path, "temp_extraction/256x256/splitted_sketches", split)
    
    # Alternative paths to try
    if not os.path.exists(sketches_root):
        sketches_root = os.path.join(dataset_path, f"temp_extraction/256x256/sketch/{split}")
    if not os.path.exists(sketches_root):
        # Try without split
        sketches_root = os.path.join(dataset_path, "temp_extraction/256x256/sketch")
    
    print(f"Looking for sketches at: {sketches_root}")
    
    if not os.path.exists(sketches_root):
        print(f"❌ Error: Sketches directory not found!")
        print(f"Expected at: {sketches_root}")
        print("Available directories:")
        base = os.path.join(dataset_path, "temp_extraction/256x256")
        if os.path.exists(base):
            print(f"  Contents of {base}:")
            for item in os.listdir(base):
                print(f"    - {item}")
        raise FileNotFoundError(f"Sketches not found at {sketches_root}")
    
    # Create output directories
    output_base = Path(output_base)
    images_dir = output_base / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Sketches directory found: {sketches_root}")
    print(f"Preparing dataset with {images_per_class} images per class...")
    
    # Collect classes (subdirectories in sketches_root)
    classes = [d for d in os.listdir(sketches_root) 
               if os.path.isdir(os.path.join(sketches_root, d))]
    
    print(f"Found {len(classes)} classes")
    print(f"Sample classes: {classes[:5]}")
    
    # Prepare data
    csv_data = []
    total_count = 0
    
    for class_name in tqdm(classes, desc="Processing classes"):
        class_dir = os.path.join(sketches_root, class_name)
        
        # Get all sketch files in this class
        sketch_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Take subset
        sketch_files = sketch_files[:images_per_class]
        
        # Process each sketch
        for sketch_file in sketch_files:
            try:
                sketch_path = os.path.join(class_dir, sketch_file)
                
                # Generate caption from class name
                caption = extract_caption_from_filename(sketch_file, class_name)
                
                # Copy sketch to dataset
                output_filename = f"sketch_{total_count:06d}.png"
                output_path = images_dir / output_filename
                
                # Load and save sketch
                img = Image.open(sketch_path).convert("RGB")
                img.save(output_path, quality=95)
                
                # Add to CSV data
                csv_data.append({
                    "image_filename": output_filename,
                    "caption": caption,
                    "original_path": sketch_path,
                    "class": class_name
                })
                
                total_count += 1
                
            except Exception as e:
                print(f"\nError processing {sketch_path}: {e}")
                continue
    
    # Save CSV
    csv_path = output_base / "captions.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Dataset prepared successfully!")
    print(f"  Total images: {len(csv_data)}")
    print(f"  Classes: {len(classes)}")
    print(f"  Images per class: ~{len(csv_data) // len(classes)}")
    print(f"  Location: {output_base}")
    print(f"  CSV: {csv_path}")
    print(f"{'='*60}")
    
    return csv_path

# Check if dataset already exists
if not os.path.exists(os.path.join(config.dataset_base, "captions.csv")):
    print("Dataset not found. Preparing dataset from Kaggle...")
    
    # Download and prepare Sketchy dataset
    prepare_sketchy_dataset_from_kaggle(
        output_base=config.dataset_base,
        images_per_class=IMAGES_PER_CLASS,
        use_train=USE_TRAIN_SPLIT,
        download=DOWNLOAD_DATASET
    )
else:
    print(f"✓ Dataset already exists at: {config.dataset_base}")
    print(f"✓ Captions CSV found")
    
    # Verify dataset integrity
    import pandas as pd
    captions_csv = os.path.join(config.dataset_base, "captions.csv")
    df = pd.read_csv(captions_csv)
    images_dir = os.path.join(config.dataset_base, "images")
    
    if os.path.exists(images_dir):
        num_images = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✓ Found {num_images} images in dataset")
        print(f"✓ Found {len(df)} entries in captions CSV")
        
        # Show sample
        print(f"\nSample entries:")
        for idx in range(min(3, len(df))):
            print(f"  - {df.iloc[idx]['image_filename']}: {df.iloc[idx]['caption']}")
    else:
        print("⚠ Warning: images directory not found!")
        print(f"Expected at: {images_dir}")

# ============================================================
# CELL 6: Start Training!
# ============================================================
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

# Import the training function
from train_controlnet import train

# Run training with the configured settings
print("\nInitiating ControlNet training on Sketchy dataset...")
print("=" * 60)
print("Training Process:")
print("  1. Sketch images are loaded")
print("  2. HED detector generates edge maps from sketches")
print("  3. ControlNet learns to condition image generation on these edges")
print("  4. Model learns to generate realistic images from sketch conditioning")
print("=" * 60)
print("\nThis will take several hours depending on your settings.")
print("Progress will be displayed below.")
print("Checkpoints will be automatically saved to Google Drive.")
print("="*60 + "\n")

try:
    # Start training
    train(config)
    
    # Training completed successfully
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"✓ Final model saved to: {config.output_dir}")
    print(f"✓ Checkpoints saved to: {config.checkpoint_dir}")
    print("\n🎉 Training finished! Your ControlNet model is saved to Google Drive.")
    print(f"\nModel location: {config.output_dir}")
    print("\n" + "="*60)
    print("What your model does:")
    print("  ✓ Takes sketch images as input")
    print("  ✓ Generates realistic images from sketches")
    print("  ✓ Uses text prompts to guide the generation")
    print("  ✓ Trained on Sketchy dataset classes")
    print("="*60)
    print("\nYou can now use this model for inference!")
    print(f"\nTo test your model, run:")
    print(f"  from inference import load_model, generate_image")
    print(f"  pipe = load_model('{config.output_dir}')")
    print(f"  generate_image(pipe, 'sketch.jpg', 'a realistic photo of [object]', 'output.png')")
    
except KeyboardInterrupt:
    print("\n" + "="*60)
    print("TRAINING INTERRUPTED")
    print("="*60)
    print("⚠ Training was interrupted by user")
    print(f"✓ Checkpoints saved to: {config.checkpoint_dir}")
    print("\nYou can resume training by setting:")
    print(f"  config.resume_from_checkpoint = '/path/to/checkpoint'")
    
except Exception as e:
    print("\n" + "="*60)
    print("TRAINING ERROR")
    print("="*60)
    print(f"❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
    print("\nPlease check:")
    print("  1. Dataset is properly formatted")
    print("  2. Google Drive has sufficient space")
    print("  3. GPU runtime is enabled")
    print("  4. All paths in configuration are correct")

# ============================================================
# CELL 7: Optional - Test Your Model
# ============================================================
print("\n" + "="*60)
print("OPTIONAL: TEST YOUR TRAINED MODEL")
print("="*60)
print("\nYour model converts sketches to realistic images!")
print("\nTo test your model with a sketch, you can run:")
print("\n```python")
print("# Import inference utilities")
print("from inference import load_model, generate_image")
print("")
print("# Load your trained model")
print(f"pipe = load_model('{config.output_dir}')")
print("")
print("# Generate a realistic image from a sketch")
print("# The sketch should be a simple line drawing")
print("generate_image(")
print("    pipe=pipe,")
print("    input_image_path='path/to/your_sketch.jpg',  # Your sketch file")
print("    prompt='a realistic photo of an airplane',    # Describe what the sketch represents")
print("    output_path='generated_image.png',")
print(f"    condition_type='{config.condition_type}'")
print(")")
print("```")
print("\nNote: The model was trained on sketches from the Sketchy dataset.")
print("      It works best with simple line drawings of objects it was trained on.")
print(f"      Training classes included in the dataset (check the CSV for full list)")
print("\n" + "="*60)
print("Training script completed!")
print(f"Dataset: Sketches from Kaggle (Sketchy Images Dataset)")
print(f"Training samples: Check {config.dataset_base}/captions.csv")
print(f"Model: {config.output_dir}")
print("="*60)
