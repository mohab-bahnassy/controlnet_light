"""
Training script for lightweight ControlNet variants.

This is a modified version of tutorial_train.py that demonstrates
how to train with the lightweight variants.

Simply change the config_path to switch between variants:
- './models/cldm_v15_light.yaml' - Light (50% channels)
- './models/cldm_v15_tiny.yaml' - Tiny (25% channels)  
- './models/cldm_v15_efficient.yaml' - Efficient (Depthwise Sep Conv)
"""

from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# ============================================================================
# Configuration - Change these to experiment with different variants
# ============================================================================

# Choose your variant:
# - './models/cldm_v15.yaml' - Standard (baseline)
# - './models/cldm_v15_light.yaml' - Light (recommended, 50% channels)
# - './models/cldm_v15_tiny.yaml' - Tiny (25% channels, very fast)
# - './models/cldm_v15_efficient.yaml' - Efficient (depthwise separable)

config_path = './models/cldm_v15_light.yaml'  # Default to Light variant

# Resume from initial checkpoint (same for all variants)
resume_path = './models/control_sd15_ini.ckpt'

# Training hyperparameters
# Note: Lightweight variants can use larger batch sizes!
batch_size = 4  # Increase to 8 for Light, 12 for Tiny
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# ============================================================================


print(f"Training with config: {config_path}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(config_path).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Count parameters for reference
control_params = sum(p.numel() for p in model.control_model.parameters())
print(f"Control model parameters: {control_params:,}")

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
print("Starting training...")
trainer.fit(model, dataloader)

print("Training complete!")


