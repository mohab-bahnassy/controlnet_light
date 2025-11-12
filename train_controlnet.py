# -*- coding: utf-8 -*-
"""
ControlNet Lightweight Training Script for Google Colab
Train a lightweight ControlNet model from scratch or fine-tune existing one
"""

import os
import sys
import json
import argparse
from pathlib import Path

# ============================================================
# 1. Setup and Mount Google Drive
# ============================================================
print("=" * 60)
print("STEP 1: Mounting Google Drive")
print("=" * 60)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Google Drive mounted successfully")
    IN_COLAB = True
except:
    print("⚠ Not running in Colab, skipping Drive mount")
    IN_COLAB = False

# ============================================================
# 2. Install Dependencies
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Installing Dependencies")
print("=" * 60)

dependencies = [
    "diffusers==0.25.0",
    "transformers==4.36.0",
    "accelerate==0.25.0",
    "controlnet_aux",
    "xformers",
    "datasets",
    "pandas",
    "torchvision",
    "opencv-python",
    "Pillow",
    "torch>=2.0.0",
    "tensorboard"
]

print("Installing packages...")
for dep in dependencies:
    print(f"  - {dep}")
    os.system(f"pip install -q {dep}")

print("✓ All dependencies installed")

# ============================================================
# 3. Imports
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Importing Libraries")
print("=" * 60)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from controlnet_aux import HEDdetector
import cv2

print("✓ Libraries imported successfully")

# ============================================================
# 4. Configuration
# ============================================================
class TrainingConfig:
    """Training configuration"""
    
    # Paths (modify these for your setup)
    if IN_COLAB:
        output_dir = "/content/drive/MyDrive/AML/controlnet_trained"
        dataset_base = "/content/drive/MyDrive/AML/dataset"
        checkpoint_dir = "/content/drive/MyDrive/AML/checkpoints"
    else:
        output_dir = "./controlnet_trained"
        dataset_base = "./dataset"
        checkpoint_dir = "./checkpoints"
    
    # Model configuration
    pretrained_model_name = "runwayml/stable-diffusion-v1-5"
    controlnet_model_name = "lllyasviel/sd-controlnet-scribble"  # Starting point
    
    # Training hyperparameters
    resolution = 512
    train_batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 1e-5
    lr_scheduler = "constant"
    lr_warmup_steps = 500
    max_train_steps = 10000
    checkpointing_steps = 1000
    validation_steps = 500
    
    # Optimization
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    
    # Mixed precision
    mixed_precision = "fp16"  # or "bf16" if supported
    
    # Lightweight modifications
    use_8bit_adam = True
    enable_xformers = True
    gradient_checkpointing = True
    
    # Seed
    seed = 42
    
    # Dataset
    condition_type = "scribble"  # scribble, canny, hed, etc.
    
    # Resume training
    resume_from_checkpoint = None

config = TrainingConfig()

# Create directories
os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)
os.makedirs(config.dataset_base, exist_ok=True)

print("\n" + "=" * 60)
print("STEP 4: Configuration")
print("=" * 60)
print(f"Output directory: {config.output_dir}")
print(f"Dataset directory: {config.dataset_base}")
print(f"Resolution: {config.resolution}")
print(f"Batch size: {config.train_batch_size}")
print(f"Learning rate: {config.learning_rate}")
print(f"Max train steps: {config.max_train_steps}")
print(f"Mixed precision: {config.mixed_precision}")

# ============================================================
# 5. Dataset Class
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Setting up Dataset")
print("=" * 60)

class ControlNetDataset(Dataset):
    """
    Dataset for ControlNet training
    Expected structure:
    dataset_base/
      ├── images/          # Original images
      ├── conditions/      # Conditioning images (scribbles/edges)
      └── captions.csv     # CSV with columns: image_filename, caption
    """
    
    def __init__(
        self,
        dataset_base,
        captions_csv,
        resolution=512,
        condition_type="scribble"
    ):
        self.dataset_base = Path(dataset_base)
        self.images_dir = self.dataset_base / "images"
        self.conditions_dir = self.dataset_base / "conditions"
        self.resolution = resolution
        self.condition_type = condition_type
        
        # Load captions
        if os.path.exists(captions_csv):
            self.captions_df = pd.read_csv(captions_csv)
            print(f"✓ Loaded {len(self.captions_df)} captions from CSV")
        else:
            print(f"⚠ Captions file not found at {captions_csv}")
            self.captions_df = None
        
        # Find all images
        self.image_files = []
        if self.images_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                self.image_files.extend(list(self.images_dir.glob(ext)))
        
        print(f"✓ Found {len(self.image_files)} images")
        
        # Transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.condition_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])
        
        # Initialize HED detector for on-the-fly condition generation
        if condition_type == "scribble":
            print("Loading HED detector for scribble generation...")
            self.hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
            print("✓ HED detector loaded")
        else:
            self.hed_detector = None
    
    def __len__(self):
        return len(self.image_files)
    
    def get_caption(self, image_filename):
        """Get caption for an image"""
        if self.captions_df is not None:
            row = self.captions_df[self.captions_df['image_filename'] == image_filename]
            if not row.empty:
                return row.iloc[0]['caption']
        return "a photo"  # Default caption
    
    def generate_condition(self, image):
        """Generate conditioning image from source image"""
        if self.condition_type == "scribble" and self.hed_detector is not None:
            # Use HED detector to generate scribble
            condition = self.hed_detector(image, scribble=True)
            return condition
        elif self.condition_type == "canny":
            # Canny edge detection
            img_array = np.array(image)
            edges = cv2.Canny(img_array, 100, 200)
            condition = Image.fromarray(edges).convert("RGB")
            return condition
        else:
            # Return grayscale as default
            return image.convert("L").convert("RGB")
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image_filename = image_path.name
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Check if pre-generated condition exists
        condition_path = self.conditions_dir / image_filename
        if condition_path.exists():
            condition_image = Image.open(condition_path).convert("RGB")
        else:
            # Generate condition on the fly
            condition_image = self.generate_condition(image)
        
        # Get caption
        caption = self.get_caption(image_filename)
        
        # Apply transforms
        image = self.image_transforms(image)
        condition_image = self.condition_transforms(condition_image)
        
        return {
            "pixel_values": image,
            "conditioning_pixel_values": condition_image,
            "caption": caption
        }

# ============================================================
# 6. Helper Functions
# ============================================================

def collate_fn(examples):
    """Collate function for DataLoader"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    captions = [example["caption"] for example in examples]
    
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "captions": captions
    }

def save_checkpoint(controlnet, accelerator, step, config):
    """Save training checkpoint"""
    save_path = os.path.join(config.checkpoint_dir, f"checkpoint-{step}")
    accelerator.save_state(save_path)
    
    # Also save just the ControlNet weights
    unwrapped_controlnet = accelerator.unwrap_model(controlnet)
    unwrapped_controlnet.save_pretrained(
        os.path.join(config.output_dir, f"controlnet-step-{step}")
    )
    print(f"✓ Checkpoint saved at step {step}")

def validation_loop(controlnet, vae, text_encoder, tokenizer, noise_scheduler, config, step):
    """Run validation"""
    print(f"\n{'='*60}")
    print(f"Running validation at step {step}")
    print(f"{'='*60}")
    
    # Create pipeline for inference
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        config.pretrained_model_name,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipeline.to("cuda")
    
    # Sample validation
    validation_prompt = "a realistic photo of a cat"
    validation_image_path = os.path.join(config.output_dir, f"validation-step-{step}.png")
    
    # Create a simple edge map for validation
    validation_condition = torch.ones(1, 3, config.resolution, config.resolution) * 0.5
    validation_condition = validation_condition.to("cuda")
    
    with torch.no_grad():
        image = pipeline(
            prompt=validation_prompt,
            image=validation_condition,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]
        
        image.save(validation_image_path)
        print(f"✓ Validation image saved to {validation_image_path}")
    
    del pipeline
    torch.cuda.empty_cache()

# ============================================================
# 7. Training Function
# ============================================================

def train(config):
    """Main training function"""
    
    print("\n" + "=" * 60)
    print("STEP 6: Initializing Training")
    print("=" * 60)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    
    # Set seed
    set_seed(config.seed)
    
    # Load models
    print("Loading models...")
    
    # Load noise scheduler
    # Create scheduler directly to avoid compatibility issues with different diffusers versions
    try:
        noise_scheduler = DDPMScheduler.from_pretrained(
            config.pretrained_model_name, subfolder="scheduler"
        )
    except (RuntimeError, ModuleNotFoundError) as e:
        # Fallback: create scheduler with default SD 1.5 parameters
        print(f"Warning: Could not load scheduler from pretrained ({e})")
        print("Creating scheduler with default Stable Diffusion 1.5 parameters...")
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon"
        )
    
    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name, subfolder="tokenizer"
    )
    
    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(
        config.pretrained_model_name, subfolder="text_encoder"
    )
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name, subfolder="vae"
    )
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name, subfolder="unet"
    )
    
    # Load or initialize ControlNet
    print("Loading ControlNet...")
    try:
        controlnet = ControlNetModel.from_pretrained(config.controlnet_model_name)
        print(f"✓ Loaded ControlNet from {config.controlnet_model_name}")
    except:
        print("⚠ Could not load pretrained ControlNet, initializing from UNet...")
        controlnet = ControlNetModel.from_unet(unet)
        print("✓ Initialized ControlNet from UNet")
    
    # Freeze vae, text_encoder, and unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Only train the ControlNet
    controlnet.train()
    
    # Enable optimizations
    if config.enable_xformers:
        try:
            controlnet.enable_xformers_memory_efficient_attention()
            unet.enable_xformers_memory_efficient_attention()
            print("✓ xFormers memory efficient attention enabled")
        except:
            print("⚠ xFormers not available")
    
    if config.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        print("✓ Gradient checkpointing enabled")
    
    # Setup optimizer
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            print("✓ Using 8-bit AdamW optimizer")
        except:
            optimizer_class = torch.optim.AdamW
            print("⚠ 8-bit Adam not available, using regular AdamW")
    else:
        optimizer_class = torch.optim.AdamW
    
    optimizer = optimizer_class(
        controlnet.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )
    
    # Setup dataset
    print("\nLoading dataset...")
    captions_csv = os.path.join(config.dataset_base, "captions.csv")
    train_dataset = ControlNetDataset(
        dataset_base=config.dataset_base,
        captions_csv=captions_csv,
        resolution=config.resolution,
        condition_type=config.condition_type
    )
    
    if len(train_dataset) == 0:
        print("❌ ERROR: No training data found!")
        print(f"Please ensure images are in: {config.dataset_base}/images/")
        print(f"And captions CSV is at: {captions_csv}")
        return
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
    )
    
    # Prepare everything with accelerator
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move models to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    unet.to(accelerator.device)
    
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader)
    num_train_epochs = config.max_train_steps // num_update_steps_per_epoch
    
    print(f"\n{'='*60}")
    print("Training Configuration:")
    print(f"{'='*60}")
    print(f"  Number of training samples: {len(train_dataset)}")
    print(f"  Number of epochs: {num_train_epochs}")
    print(f"  Batch size per device: {config.train_batch_size}")
    print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Total optimization steps: {config.max_train_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"{'='*60}\n")
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(range(config.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")
    
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample timestep
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=latents.device
                )
                timesteps = timesteps.long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                input_ids = tokenizer(
                    batch["captions"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(accelerator.device)
                
                encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Get ControlNet conditioning
                controlnet_image = batch["conditioning_pixel_values"]
                
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )
                
                # Predict noise with UNet
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backpropagation
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), config.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(**logs)
                
                # Save checkpoint
                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(controlnet, accelerator, global_step, config)
                
                # Run validation
                if global_step % config.validation_steps == 0:
                    if accelerator.is_main_process:
                        validation_loop(
                            accelerator.unwrap_model(controlnet),
                            vae, text_encoder, tokenizer,
                            noise_scheduler, config, global_step
                        )
            
            if global_step >= config.max_train_steps:
                break
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_controlnet = accelerator.unwrap_model(controlnet)
        unwrapped_controlnet.save_pretrained(config.output_dir)
        print(f"\n{'='*60}")
        print(f"✓ Training complete!")
        print(f"✓ Final model saved to: {config.output_dir}")
        print(f"{'='*60}")
        
        # Save training config
        config_path = os.path.join(config.output_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(vars(config), f, indent=2)
        print(f"✓ Training config saved to: {config_path}")

# ============================================================
# 8. Main Execution
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CONTROLNET LIGHTWEIGHT TRAINING")
    print("=" * 60)
    
    try:
        train(config)
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

