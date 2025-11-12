# PyTorch Lightning 2.x Compatibility Fix

## Problem
When running training with PyTorch Lightning 2.x, you may encounter these errors:

1. `ModuleNotFoundError: No module named 'pytorch_lightning.utilities.distributed'`
2. `TypeError: LatentDiffusion.on_train_batch_start() missing 1 required positional argument: 'dataloader_idx'`

## Root Cause
PyTorch Lightning 2.x changed the import path for `rank_zero_only` decorator and updated callback method signatures to include `dataloader_idx` parameter.

## Files Fixed

### 1. `ldm/models/diffusion/ddpm.py`
**Line 20** - Updated import statement:

**Before:**
```python
from pytorch_lightning.utilities.distributed import rank_zero_only
```

**After:**
```python
from pytorch_lightning.utilities.rank_zero import rank_zero_only
```

### 2. `cldm/logger.py`
**Line 8** - Updated import statement:

**Before:**
```python
from pytorch_lightning.utilities.distributed import rank_zero_only
```

**After:**
```python
from pytorch_lightning.utilities.rank_zero import rank_zero_only
```

## Method Signatures Already Correct
The `on_train_batch_start` and `on_train_batch_end` methods already had the correct signatures with `dataloader_idx` parameter:

```python
# In ldm/models/diffusion/ddpm.py (line 591)
def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
    # ... implementation

# In cldm/logger.py (line 74)
def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    # ... implementation
```

The error occurred because the outdated import prevented the module from loading, not because the signatures were incorrect.

## Testing
After applying these fixes, you should be able to run training without the `TypeError` or `ModuleNotFoundError`.

## Additional Notes
- These changes are backward-compatible with PyTorch Lightning 1.x for the most part
- If you encounter similar errors in other files, check for the old import path and update it
- The `colab_train.py` file has a temporary `ImageLogger` workaround that can now be removed if desired, as the original logger should work correctly

## Date Fixed
November 12, 2025

