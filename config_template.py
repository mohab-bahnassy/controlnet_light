# -*- coding: utf-8 -*-
"""
Configuration Template for ControlNet Training
Copy this file and modify for your specific use case
"""

class TrainingConfig:
    """Training configuration - Modify these values for your setup"""
    
    # ============================================================
    # PATHS - MODIFY THESE FOR YOUR SETUP
    # ============================================================
    
    # Google Colab paths (if using Colab)
    output_dir = "/content/drive/MyDrive/AML/controlnet_trained"
    dataset_base = "/content/drive/MyDrive/AML/dataset"
    checkpoint_dir = "/content/drive/MyDrive/AML/checkpoints"
    
    # Local paths (if running locally)
    # output_dir = "./output/controlnet_trained"
    # dataset_base = "./data/dataset"
    # checkpoint_dir = "./output/checkpoints"
    
    # ============================================================
    # MODEL CONFIGURATION
    # ============================================================
    
    # Base Stable Diffusion model
    pretrained_model_name = "runwayml/stable-diffusion-v1-5"
    
    # Starting ControlNet (set to None to train from scratch)
    controlnet_model_name = "lllyasviel/sd-controlnet-scribble"
    # controlnet_model_name = None  # Train from scratch
    
    # ============================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================
    
    # Image resolution (512 or 768)
    resolution = 512
    
    # Batch size per GPU
    # Reduce to 2 or 1 if you get out-of-memory errors
    train_batch_size = 4
    
    # Gradient accumulation steps
    # Effective batch size = train_batch_size × gradient_accumulation_steps
    gradient_accumulation_steps = 4
    
    # Learning rate
    # Fine-tuning: 1e-5 to 5e-5
    # Training from scratch: 1e-4 to 5e-4
    learning_rate = 1e-5
    
    # Learning rate scheduler
    lr_scheduler = "constant"  # Options: constant, linear, cosine
    lr_warmup_steps = 500
    
    # Training duration
    # Small dataset (<1000 images): 5000-10000 steps
    # Medium dataset (1000-5000): 10000-20000 steps
    # Large dataset (>5000): 20000+ steps
    max_train_steps = 10000
    
    # Checkpointing
    checkpointing_steps = 1000  # Save checkpoint every N steps
    validation_steps = 500      # Run validation every N steps
    
    # ============================================================
    # OPTIMIZATION SETTINGS
    # ============================================================
    
    # Adam optimizer parameters
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    
    # Use 8-bit Adam optimizer (saves memory)
    use_8bit_adam = True
    
    # ============================================================
    # MEMORY OPTIMIZATION
    # ============================================================
    
    # Mixed precision training
    # "fp16" for V100/T4, "bf16" for A100
    mixed_precision = "fp16"
    
    # Enable xFormers memory efficient attention
    enable_xformers = True
    
    # Gradient checkpointing (saves memory at cost of speed)
    gradient_checkpointing = True
    
    # ============================================================
    # DATASET CONFIGURATION
    # ============================================================
    
    # Type of conditioning
    # Options: "scribble", "canny"
    condition_type = "scribble"
    
    # ============================================================
    # REPRODUCIBILITY
    # ============================================================
    
    # Random seed
    seed = 42
    
    # ============================================================
    # ADVANCED OPTIONS
    # ============================================================
    
    # Resume from checkpoint (set to checkpoint path or None)
    resume_from_checkpoint = None
    # resume_from_checkpoint = "/content/drive/MyDrive/AML/checkpoints/checkpoint-5000"


# ============================================================
# PRESET CONFIGURATIONS
# ============================================================

class QuickTestConfig(TrainingConfig):
    """Quick test configuration (1 hour training)"""
    max_train_steps = 2000
    train_batch_size = 4
    checkpointing_steps = 500
    validation_steps = 250


class StandardConfig(TrainingConfig):
    """Standard training configuration (3-4 hours)"""
    max_train_steps = 10000
    train_batch_size = 4
    checkpointing_steps = 1000
    validation_steps = 500


class HighQualityConfig(TrainingConfig):
    """High quality training configuration (6+ hours)"""
    max_train_steps = 20000
    train_batch_size = 4
    checkpointing_steps = 2000
    validation_steps = 1000
    learning_rate = 5e-6  # Lower LR for stability


class LowMemoryConfig(TrainingConfig):
    """Configuration for limited memory (free Colab T4)"""
    train_batch_size = 2
    gradient_accumulation_steps = 8
    use_8bit_adam = True
    gradient_checkpointing = True
    enable_xformers = True
    resolution = 512


class HighMemoryConfig(TrainingConfig):
    """Configuration for high-end GPUs (A100, V100)"""
    train_batch_size = 8
    gradient_accumulation_steps = 2
    resolution = 512
    mixed_precision = "bf16"


# ============================================================
# USAGE EXAMPLES
# ============================================================

if __name__ == "__main__":
    print("Configuration Templates")
    print("=" * 60)
    print("\nAvailable configurations:")
    print("  - TrainingConfig: Default balanced configuration")
    print("  - QuickTestConfig: Fast training for testing (~1 hour)")
    print("  - StandardConfig: Standard training (~3-4 hours)")
    print("  - HighQualityConfig: High quality training (~6+ hours)")
    print("  - LowMemoryConfig: For free Colab T4 GPU")
    print("  - HighMemoryConfig: For A100/V100 GPUs")
    print("\nUsage:")
    print("  from config_template import StandardConfig")
    print("  config = StandardConfig()")
    print("  # Modify as needed")
    print("  config.output_dir = '/your/path'")
    print("  # Then pass to training")
    print("  from train_controlnet import train")
    print("  train(config)")

