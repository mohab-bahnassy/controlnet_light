# -*- coding: utf-8 -*-
"""
Dataset Preparation Script for ControlNet Training
Prepares images and captions from various sources
"""

import os
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
from PIL import Image

def prepare_dataset_from_paths(
    image_paths_list,
    captions_list,
    output_base_dir,
    copy_images=True,
    max_images=None
):
    """
    Prepare dataset from lists of image paths and captions
    
    Args:
        image_paths_list: List of image file paths
        captions_list: List of captions (same length as image_paths_list)
        output_base_dir: Where to save the prepared dataset
        copy_images: Whether to copy images or just create CSV
        max_images: Maximum number of images to include (None = all)
    """
    
    output_base = Path(output_base_dir)
    images_dir = output_base / "images"
    conditions_dir = output_base / "conditions"
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing dataset in: {output_base_dir}")
    print(f"Total images: {len(image_paths_list)}")
    
    # Limit dataset size if requested
    if max_images:
        image_paths_list = image_paths_list[:max_images]
        captions_list = captions_list[:max_images]
        print(f"Limited to {max_images} images")
    
    # Prepare data for CSV
    csv_data = []
    
    # Process images
    for idx, (img_path, caption) in enumerate(tqdm(zip(image_paths_list, captions_list), 
                                                     desc="Processing images",
                                                     total=len(image_paths_list))):
        try:
            # Generate output filename
            img_path = Path(img_path)
            output_filename = f"image_{idx:06d}{img_path.suffix}"
            output_path = images_dir / output_filename
            
            if copy_images:
                # Verify image can be opened
                img = Image.open(img_path)
                img.verify()
                
                # Copy or convert image
                img = Image.open(img_path).convert("RGB")
                img.save(output_path, quality=95)
            
            # Add to CSV data
            csv_data.append({
                "image_filename": output_filename,
                "caption": caption,
                "original_path": str(img_path)
            })
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue
    
    # Save CSV
    csv_path = output_base / "captions.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Dataset prepared successfully!")
    print(f"  Images: {len(csv_data)}")
    print(f"  Location: {output_base_dir}")
    print(f"  CSV: {csv_path}")
    
    return csv_path


def prepare_from_existing_csv(
    existing_csv_path,
    image_path_column,
    caption_column,
    output_base_dir,
    copy_images=True,
    max_images=None
):
    """
    Prepare dataset from existing CSV file
    
    Args:
        existing_csv_path: Path to existing CSV
        image_path_column: Name of column containing image paths
        caption_column: Name of column containing captions
        output_base_dir: Where to save prepared dataset
        copy_images: Whether to copy images
        max_images: Maximum number of images
    """
    
    print(f"Loading CSV from: {existing_csv_path}")
    df = pd.read_csv(existing_csv_path)
    
    print(f"Found {len(df)} rows in CSV")
    
    # Extract image paths and captions
    image_paths = df[image_path_column].tolist()
    captions = df[caption_column].tolist()
    
    return prepare_dataset_from_paths(
        image_paths,
        captions,
        output_base_dir,
        copy_images=copy_images,
        max_images=max_images
    )


def prepare_from_sketchy_dataset(
    photo_root,
    csv_path,
    output_base_dir,
    split="train",
    max_images=5000
):
    """
    Prepare dataset from the Sketchy dataset structure used in the original notebook
    
    Args:
        photo_root: Root directory containing photos
        csv_path: Path to captions CSV
        output_base_dir: Output directory
        split: "train" or "test"
        max_images: Maximum images to include
    """
    
    print(f"Preparing Sketchy dataset from: {photo_root}")
    
    # Load captions
    df = pd.read_csv(csv_path)
    
    # Build captions dictionary
    captions_dict = {}
    for _, row in df.iterrows():
        file_path = row['image_path']
        caption = row['caption']
        parts = file_path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            parent = parts[-2]
            filename = os.path.splitext(parts[-1])[0]
            filename_main = filename.split('-')[0]
            key = f"{parent}/{filename_main}"
            captions_dict[key] = caption
    
    print(f"Loaded {len(captions_dict)} captions")
    
    # Collect image paths
    image_paths = []
    captions_list = []
    
    for root, _, files in os.walk(photo_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                
                # Extract key
                parts = full_path.replace("\\", "/").split("/")
                if len(parts) >= 2:
                    parent = parts[-2]
                    filename = os.path.splitext(parts[-1])[0]
                    filename_main = filename.split('-')[0]
                    key = f"{parent}/{filename_main}"
                    
                    if key in captions_dict:
                        image_paths.append(full_path)
                        captions_list.append(captions_dict[key])
                        
                        if len(image_paths) >= max_images:
                            break
        
        if len(image_paths) >= max_images:
            break
    
    print(f"Found {len(image_paths)} images with captions")
    
    return prepare_dataset_from_paths(
        image_paths,
        captions_list,
        output_base_dir,
        copy_images=True,
        max_images=max_images
    )


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DATASET PREPARATION FOR CONTROLNET TRAINING")
    print("=" * 60)
    
    # Example 1: Prepare from Sketchy dataset (like in the original notebook)
    # Uncomment and modify paths as needed:
    
    # prepare_from_sketchy_dataset(
    #     photo_root="/content/drive/Shareddrives/AML/Sketchy/Rendered Images/256x256/photo",
    #     csv_path="/content/drive/Shareddrives/AML/path_caption_pairs.csv",
    #     output_base_dir="/content/drive/MyDrive/AML/dataset",
    #     max_images=5000
    # )
    
    # Example 2: Prepare from existing CSV
    # prepare_from_existing_csv(
    #     existing_csv_path="/path/to/your/captions.csv",
    #     image_path_column="image_path",
    #     caption_column="caption",
    #     output_base_dir="/content/drive/MyDrive/AML/dataset",
    #     max_images=5000
    # )
    
    # Example 3: Prepare from lists
    # image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
    # captions = ["a cat", "a dog"]
    # prepare_dataset_from_paths(
    #     image_paths,
    #     captions,
    #     output_base_dir="/content/drive/MyDrive/AML/dataset"
    # )
    
    print("\n⚠ Please uncomment and configure one of the examples above!")
    print("Then run this script to prepare your dataset.")

