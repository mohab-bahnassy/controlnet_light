# ControlNet Lightweight Training - Project Overview

## 📦 What This Project Does

This project provides a complete pipeline to **train a lightweight ControlNet model** on Google Colab and save it to Google Drive. ControlNet allows you to control image generation using conditioning images (like sketches, edges, or poses).

### Key Features

✅ **Easy to use** - Run on Google Colab with minimal setup  
✅ **Memory efficient** - Optimized for free Colab GPUs  
✅ **Auto-saves to Drive** - Never lose your progress  
✅ **Checkpoint system** - Resume training anytime  
✅ **Flexible dataset** - Works with various image datasets  
✅ **Ready-to-use inference** - Test your model immediately  

---

## 📁 Files in This Project

### Core Scripts

| File | Purpose | When to Use |
|------|---------|-------------|
| `train_controlnet.py` | Main training script (modular) | When you want full control over configuration |
| `colab_train.py` | All-in-one training script | **Easiest way** - Just run one script |
| `prepare_dataset.py` | Dataset preparation utilities | To prepare your data before training |
| `inference.py` | Inference/generation script | After training, to use your model |

### Configuration

| File | Purpose |
|------|---------|
| `config_template.py` | Pre-configured training setups |
| `requirements.txt` | Python dependencies |

### Documentation

| File | Purpose | Start Here? |
|------|---------|-------------|
| `QUICKSTART.md` | **Quick start guide** | ✅ **START HERE** |
| `README.md` | Detailed documentation | For deep dive |
| `PROJECT_OVERVIEW.md` | This file - project overview | For understanding structure |

### Original

| File | Purpose |
|------|---------|
| `controlnet.py` | Your original Colab notebook (converted) |

---

## 🚀 Quick Start - 4 Steps

### 1️⃣ Upload Repository to Google Drive

Upload the entire `controlnet_light` folder to `/content/drive/MyDrive/controlnet_light/`

### 2️⃣ Open Google Colab

Go to https://colab.research.google.com/ and set runtime to GPU

### 3️⃣ Upload and Configure `colab_train.py`

Upload the script to Colab and edit the `REPO_PATH` to point to your repository in Drive

### 4️⃣ Run It!

```python
!python colab_train.py
```

**That's it!** The script will import from the repository and train your model, saving everything to Google Drive.

---

## 📊 Training Workflow

```
┌─────────────────────┐
│  Prepare Dataset    │  ← prepare_dataset.py
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Configure Training │  ← config_template.py
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Train Model       │  ← train_controlnet.py or colab_train.py
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Use Model          │  ← inference.py
└─────────────────────┘
```

---

## 🎯 Which File Should I Use?

### For Training

**Beginner?** → Use `colab_train.py` (orchestrates everything, easiest)
- Imports from existing modules
- Just edit paths and run

**Need customization?** → Directly import `train_controlnet.py` in Colab
- Full control over configuration
- Use `config_template.py` for presets

### For Dataset Preparation

**Have raw images?** → Use `prepare_dataset.py`

**Using Sketchy dataset?** → Already integrated in `colab_train.py`

### For Using Your Model

**Generate single image?** → Use `inference.py --mode single`

**Batch processing?** → Use `inference.py --mode batch`

**Interactive testing?** → Use `inference.py --mode interactive`

---

## 💾 Output Structure

After running training, you'll have:

```
/content/drive/MyDrive/AML/
│
├── controlnet_trained/          ← Your final trained model
│   ├── config.json
│   ├── diffusion_pytorch_model.bin
│   └── training_config.json
│
├── checkpoints/                 ← Intermediate saves
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   └── ...
│
└── dataset/                     ← Your training data
    ├── images/
    │   ├── image_000000.jpg
    │   └── ...
    └── captions.csv
```

---

## ⚙️ System Requirements

### For Training

- **Platform**: Google Colab (recommended) or local machine with GPU
- **GPU**: T4 or better (free Colab provides T4)
- **RAM**: 12GB+ (free Colab provides this)
- **Storage**: 5-20GB on Google Drive
- **Time**: 2-6 hours depending on dataset size

### For Inference

- **GPU**: Any CUDA-capable GPU
- **RAM**: 8GB+ GPU memory
- **Storage**: ~5GB for model

---

## 📚 Documentation Guide

### Read These In Order

1. **QUICKSTART.md** (5 min) - Get started immediately
2. **README.md** (15 min) - Understand all features
3. **config_template.py** (5 min) - Customize training

### When You Need Help

- **Can't start training?** → QUICKSTART.md troubleshooting
- **Out of memory?** → README.md optimization section
- **Poor results?** → README.md training tips
- **Want to customize?** → config_template.py presets

---

## 🔄 Typical Usage Scenarios

### Scenario 1: First Time User

1. Upload entire `controlnet_light` repository to Google Drive
2. Read QUICKSTART.md
3. Upload `colab_train.py` to Colab
4. Edit `REPO_PATH` and output paths in the script
5. Run: `!python colab_train.py`
6. Wait for training to complete
7. Test with `inference.py` (imported from repo)

### Scenario 2: Custom Dataset

1. Upload repository to Google Drive
2. Prepare your images and captions CSV
3. Import and run `prepare_dataset.py` in Colab to organize data
4. Edit `colab_train.py` to point to your dataset
5. Run training with `!python colab_train.py`
6. Test with imported `inference.py`

### Scenario 3: Fine-tuning Existing Model

1. Upload repository to Google Drive
2. In `colab_train.py`, set `config.controlnet_model_name` to pretrained model
3. Lower learning rate (1e-5 or 5e-6)
4. Set shorter training duration (5000-10000 steps)
5. Run training
6. Test improvements with `inference.py`

---

## 🐛 Common Issues & Solutions

| Issue | Solution | File to Check |
|-------|----------|---------------|
| "No training data found" | Verify dataset structure | prepare_dataset.py |
| Out of memory | Reduce batch size | config_template.py → LowMemoryConfig |
| Training too slow | Enable xformers, use GPU | train_controlnet.py line 72-73 |
| Poor image quality | Train longer, more data | config_template.py → HighQualityConfig |
| Model not saving | Check Drive space & paths | train_controlnet.py line 31-33 |

---

## 🎓 Learning Resources

### Understand ControlNet
- [ControlNet Paper](https://arxiv.org/abs/2302.05543)
- [ControlNet Examples](https://huggingface.co/lllyasviel)

### Understand Stable Diffusion
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Stable Diffusion Guide](https://github.com/CompVis/stable-diffusion)

### Training Tips
- [Fine-tuning Diffusion Models](https://huggingface.co/docs/diffusers/training/overview)

---

## 🤝 Contributing & Support

### Found a Bug?
- Check QUICKSTART.md troubleshooting first
- Verify your configuration matches examples
- Check Google Colab runtime is set to GPU

### Want to Improve?
- Feel free to modify scripts for your needs
- Share your configurations in config_template.py
- Document your use case for others

---

## ✅ Pre-Flight Checklist

Before starting training, make sure:

- [ ] Google Colab runtime set to **GPU**
- [ ] Google Drive has **5GB+ free space**
- [ ] Dataset prepared with **images/ folder** and **captions.csv**
- [ ] Paths in configuration are **correct**
- [ ] At least **1000 training images** available
- [ ] You have **2+ hours** for training

---

## 🎉 Success Criteria

Your training is successful when:

✅ Training completes without errors  
✅ Model is saved to Google Drive  
✅ Inference generates reasonable images  
✅ Generated images follow the conditioning  

If any of these fail, check:
- Did training run for enough steps? (10000+ recommended)
- Is your dataset diverse enough? (3000+ images ideal)
- Are captions descriptive?

---

## 📞 Next Steps

1. **Read QUICKSTART.md** to begin
2. **Run training** on a small dataset first (1000 images, 2000 steps)
3. **Test your model** with inference.py
4. **Scale up** if results are good (more data, more steps)
5. **Share your results!**

---

## 📄 License & Credits

- **ControlNet**: Lvmin Zhang et al.
- **Stable Diffusion**: Stability AI
- **Diffusers**: Hugging Face
- **This Project**: Educational and research purposes

---

**Ready to train your ControlNet model?** → Start with **QUICKSTART.md** 🚀

