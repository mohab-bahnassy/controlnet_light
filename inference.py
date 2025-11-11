# -*- coding: utf-8 -*-
"""
ControlNet Inference Script
Use your trained ControlNet model to generate images
"""

import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import HEDdetector
import argparse
import os

def load_model(controlnet_path, base_model="runwayml/stable-diffusion-v1-5"):
    """Load the trained ControlNet model"""
    print(f"Loading ControlNet from: {controlnet_path}")
    
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path,
        torch_dtype=torch.float16
    )
    
    print(f"Loading base model: {base_model}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    
    # Optimizations
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✓ xFormers enabled")
    except:
        print("⚠ xFormers not available")
    
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    print("✓ Model loaded successfully")
    return pipe

def generate_image(
    pipe,
    input_image_path,
    prompt,
    output_path="output.png",
    condition_type="scribble",
    negative_prompt="low quality, deformed, extra limbs, blurry, bad lighting",
    num_steps=25,
    guidance_scale=9.0,
    controlnet_scale=1.0,
    seed=None
):
    """Generate an image using the ControlNet model"""
    
    print(f"\nGenerating image...")
    print(f"Input: {input_image_path}")
    print(f"Prompt: {prompt}")
    
    # Load input image
    input_image = Image.open(input_image_path).convert("RGB").resize((512, 512))
    
    # Generate conditioning image
    print(f"Generating {condition_type} conditioning...")
    if condition_type == "scribble":
        hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
        control_image = hed_detector(input_image, scribble=True)
    elif condition_type == "canny":
        import cv2
        import numpy as np
        img_array = np.array(input_image)
        edges = cv2.Canny(img_array, 100, 200)
        control_image = Image.fromarray(edges).convert("RGB")
    else:
        control_image = input_image.convert("L").convert("RGB")
    
    # Set seed if provided
    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"Using seed: {seed}")
    
    # Generate
    print("Running inference...")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        controlnet_conditioning_scale=controlnet_scale,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    
    # Save result
    result.save(output_path)
    print(f"✓ Image saved to: {output_path}")
    
    # Also save conditioning image for reference
    control_output = output_path.replace(".png", "_condition.png")
    control_image.save(control_output)
    print(f"✓ Conditioning image saved to: {control_output}")
    
    return result, control_image

def batch_generate(
    pipe,
    input_dir,
    output_dir,
    prompt,
    condition_type="scribble",
    **kwargs
):
    """Generate images for all images in a directory"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    print(f"\nFound {len(image_files)} images to process")
    
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_file}")
        
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"generated_{img_file}")
        
        try:
            generate_image(
                pipe,
                input_path,
                prompt,
                output_path,
                condition_type,
                **kwargs
            )
        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
            continue
    
    print(f"\n✓ Batch processing complete! Results in: {output_dir}")

def interactive_mode(pipe, condition_type="scribble"):
    """Interactive mode for generating images"""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Generate images interactively. Type 'quit' to exit.")
    print("="*60 + "\n")
    
    while True:
        # Get input image
        input_path = input("Input image path (or 'quit'): ").strip()
        if input_path.lower() == 'quit':
            break
        
        if not os.path.exists(input_path):
            print(f"❌ File not found: {input_path}")
            continue
        
        # Get prompt
        prompt = input("Prompt: ").strip()
        if not prompt:
            prompt = "a realistic photo"
        
        # Get output path
        output_path = input("Output path (or press Enter for 'output.png'): ").strip()
        if not output_path:
            output_path = "output.png"
        
        # Generate
        try:
            generate_image(pipe, input_path, prompt, output_path, condition_type)
            print(f"\n✓ Success! Image saved to: {output_path}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="ControlNet Inference")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained ControlNet model")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "batch", "interactive"],
                        help="Inference mode")
    
    # Single/batch mode arguments
    parser.add_argument("--input", type=str,
                        help="Input image path (single mode) or directory (batch mode)")
    parser.add_argument("--output", type=str, default="output.png",
                        help="Output image path (single) or directory (batch)")
    parser.add_argument("--prompt", type=str,
                        help="Text prompt for generation")
    
    # Generation parameters
    parser.add_argument("--condition_type", type=str, default="scribble",
                        choices=["scribble", "canny"],
                        help="Type of conditioning")
    parser.add_argument("--negative_prompt", type=str,
                        default="low quality, deformed, extra limbs, blurry, bad lighting",
                        help="Negative prompt")
    parser.add_argument("--num_steps", type=int, default=25,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=9.0,
                        help="Guidance scale")
    parser.add_argument("--controlnet_scale", type=float, default=1.0,
                        help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    # Model parameters
    parser.add_argument("--base_model", type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="Base Stable Diffusion model")
    
    args = parser.parse_args()
    
    # Load model
    pipe = load_model(args.model_path, args.base_model)
    
    # Run inference based on mode
    if args.mode == "single":
        if not args.input or not args.prompt:
            print("❌ Error: --input and --prompt required for single mode")
            return
        
        generate_image(
            pipe=pipe,
            input_image_path=args.input,
            prompt=args.prompt,
            output_path=args.output,
            condition_type=args.condition_type,
            negative_prompt=args.negative_prompt,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            controlnet_scale=args.controlnet_scale,
            seed=args.seed
        )
    
    elif args.mode == "batch":
        if not args.input or not args.prompt:
            print("❌ Error: --input (directory) and --prompt required for batch mode")
            return
        
        batch_generate(
            pipe=pipe,
            input_dir=args.input,
            output_dir=args.output,
            prompt=args.prompt,
            condition_type=args.condition_type,
            negative_prompt=args.negative_prompt,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            controlnet_scale=args.controlnet_scale,
            seed=args.seed
        )
    
    elif args.mode == "interactive":
        interactive_mode(pipe, args.condition_type)
    
    print("\n✓ Done!")

if __name__ == "__main__":
    # Example usage in script mode (for Colab)
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided - show usage examples
        print("="*60)
        print("CONTROLNET INFERENCE")
        print("="*60)
        print("\nUsage examples:")
        print("\n1. Single image:")
        print("   python inference.py --model_path /path/to/model \\")
        print("                       --mode single \\")
        print("                       --input image.jpg \\")
        print("                       --prompt 'a realistic photo of a cat' \\")
        print("                       --output result.png")
        print("\n2. Batch processing:")
        print("   python inference.py --model_path /path/to/model \\")
        print("                       --mode batch \\")
        print("                       --input ./input_images/ \\")
        print("                       --output ./output_images/ \\")
        print("                       --prompt 'a realistic photo'")
        print("\n3. Interactive mode:")
        print("   python inference.py --model_path /path/to/model \\")
        print("                       --mode interactive")
        print("\n" + "="*60)
        print("\nFor Google Colab, you can also use it as a library:")
        print("\n```python")
        print("from inference import load_model, generate_image")
        print("")
        print("pipe = load_model('/content/drive/MyDrive/AML/controlnet_trained')")
        print("generate_image(")
        print("    pipe=pipe,")
        print("    input_image_path='sketch.jpg',")
        print("    prompt='a realistic photo of a cat',")
        print("    output_path='output.png'")
        print(")")
        print("```")
        print()
    else:
        main()

