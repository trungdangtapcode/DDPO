"""
Inference script for DDPO that visualizes each denoising step.
Allows specification of device and shows intermediate images during the diffusion process.
"""

import os
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime


def latent_to_image(vae, latent):
    """Convert latent tensor to PIL Image."""
    with torch.no_grad():
        # Decode the latent
        image = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        if image.shape[0] == 1:
            image = image[0]
        return Image.fromarray(image)


@torch.no_grad()
def inference_with_intermediate_steps(
    pipeline,
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    negative_prompt=None,
    height=512,
    width=512,
    save_dir="outputs",
    save_intermediates=True,
):
    """
    Run inference and save images at each denoising step.
    
    Args:
        pipeline: StableDiffusionPipeline instance
        prompt: Text prompt for generation
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        negative_prompt: Negative prompt (optional)
        height: Image height
        width: Image width
        save_dir: Directory to save intermediate images
        save_intermediates: Whether to save all intermediate steps
    
    Returns:
        final_image: Final generated PIL Image
        all_images: List of PIL Images for each step
    """
    device = pipeline.device
    
    # Create save directory
    if save_intermediates:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(save_dir) / f"inference_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving intermediate images to: {output_dir}")
    
    # Encode prompt
    do_classifier_free_guidance = guidance_scale > 1.0
    
    text_inputs = pipeline.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    
    prompt_embeds = pipeline.text_encoder(text_input_ids.to(device))[0]
    
    # Encode negative prompt if using classifier-free guidance
    if do_classifier_free_guidance:
        if negative_prompt is None:
            negative_prompt = ""
        
        uncond_tokens = pipeline.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_prompt_embeds = pipeline.text_encoder(uncond_tokens.input_ids.to(device))[0]
        
        # Concatenate for classifier-free guidance
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    
    # Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    
    # Prepare latent variables
    num_channels_latents = pipeline.unet.config.in_channels
    batch_size = 1
    shape = (batch_size, num_channels_latents, height // 8, width // 8)
    latents = torch.randn(shape, device=device, dtype=prompt_embeds.dtype)
    latents = latents * pipeline.scheduler.init_noise_sigma
    
    # Store all images
    all_images = []
    
    # Denoising loop
    print(f"\nGenerating: '{prompt}'")
    print(f"Steps: {num_inference_steps}, Guidance Scale: {guidance_scale}")
    print("-" * 80)
    
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # Expand latents if doing classifier-free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise residual
        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        
        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute previous noisy sample
        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Convert latent to image and save
        if save_intermediates:
            img = latent_to_image(pipeline.vae, latents)
            all_images.append(img)
            
            # Save intermediate image
            img.save(output_dir / f"step_{i:03d}_t_{t:04d}.png")
            
            # Print progress
            if (i + 1) % 10 == 0 or i == len(timesteps) - 1:
                print(f"Step {i+1}/{len(timesteps)}: t={t}, saved to step_{i:03d}_t_{t:04d}.png")
    
    # Final image
    final_image = latent_to_image(pipeline.vae, latents)
    
    if save_intermediates:
        final_image.save(output_dir / "final.png")
        print(f"\nFinal image saved to: {output_dir / 'final.png'}")
        
        # Save prompt info
        with open(output_dir / "prompt.txt", "w") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Negative Prompt: {negative_prompt}\n")
            f.write(f"Steps: {num_inference_steps}\n")
            f.write(f"Guidance Scale: {guidance_scale}\n")
            f.write(f"Height: {height}\n")
            f.write(f"Width: {width}\n")
    
    return final_image, all_images


def main():
    parser = argparse.ArgumentParser(description="DDPO Inference with Intermediate Steps Visualization")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--model", type=str, default="kvablack/ddpo-compressibility", 
                        help="Pretrained model path or HuggingFace model name")
    parser.add_argument("--hf_token", type=str, default=None, 
                        help="HuggingFace access token (or set HF_TOKEN env var)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory (default: ~/.cache/huggingface)")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to finetuned checkpoint (LoRA weights)")
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="Device to use (e.g., cuda:0, cuda:1, cuda:3, cpu)")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--no_intermediates", action="store_true", 
                        help="Don't save intermediate steps (only final image)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA weights from checkpoint")
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available() and "cuda" in args.device:
        device = torch.device(args.device)
        print(f"Using device: {args.device}")
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed: {args.seed}")
    
    # Load pipeline
    print(f"\nLoading model: {args.model}")
    pipeline = None
    
    try:
        print("Attempting to load model...")
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            use_auth_token=hf_token,
            cache_dir=args.cache_dir,
            local_files_only=False,
        )
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"\n❌ Failed to load model from HuggingFace Hub")
        print(f"Error: {type(e).__name__}")
        
        # Check if it's a network/download issue
        err_str = str(e)
        if "local_files_only" in err_str or "outgoing traffic" in err_str or "cached locally" in err_str or "Cannot find an appropriate cached" in err_str:
            print("\n" + "="*80)
            print("MODEL NOT FOUND - DOWNLOAD REQUIRED")
            print("="*80)
            print("\nThe model is not cached locally and cannot be downloaded automatically.")
            print("\n📥 TO FIX THIS, download the model first:")
            print("\n  Option 1 - Download with Python (recommended):")
            print("  ---------------------------------")
            print("  python scripts/download_model.py --model {}".format(args.model))
            print("\n  Option 2 - Manual download:")
            print("  ---------------------------")
            print("  python -c \"from diffusers import StableDiffusionPipeline; \\")
            print("  StableDiffusionPipeline.from_pretrained('{}')\"".format(args.model))
            print("\n  Option 3 - Use a local model path:")
            print("  -----------------------------")
            print("  python scripts/inference.py \\")
            print("    --prompt 'your prompt' \\")
            print("    --model /path/to/local/stable-diffusion \\")
            print("    --device cuda:3")
            print("\n  Option 4 - Login to HuggingFace (if needed):")
            print("  --------------------------------------------")
            print("  pip install huggingface-hub")
            print("  huggingface-cli login")
            print("  Or set environment variable HF_TOKEN with a valid token:")
            print("  export HF_TOKEN=your_token_here")
            print("\n💡 The model is ~4-5 GB and only needs to be downloaded once.")
            print("   After downloading, it will be cached for future use.")
            print("\n📚 For more help, see: TROUBLESHOOTING.md")
            print("="*80)
        else:
            print(f"\nUnexpected error: {e}")
            print("\n🔧 Troubleshooting:")
            print("1. Check internet connection: ping huggingface.co")
            print("2. Check disk space: df -h ~")
            print("3. Try a different model or local path")
            print("4. See TROUBLESHOOTING.md for detailed help")
        
        import sys
        sys.exit(1)
    
    # Set scheduler to DDIM
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    # Disable safety checker
    pipeline.safety_checker = None
    
    # Move to device
    pipeline = pipeline.to(device)
    
    # Load checkpoint if provided
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint_path = Path(args.checkpoint)
        
        if args.use_lora:
            # Load LoRA weights
            if (checkpoint_path / "pytorch_lora_weights.safetensors").exists():
                pipeline.load_lora_weights(checkpoint_path, weight_name="pytorch_lora_weights.safetensors")
                print("Loaded LoRA weights")
            elif (checkpoint_path / "pytorch_lora_weights.bin").exists():
                pipeline.load_lora_weights(checkpoint_path, weight_name="pytorch_lora_weights.bin")
                print("Loaded LoRA weights")
            else:
                print("Warning: LoRA weights not found in checkpoint")
        else:
            # Load full model weights
            if (checkpoint_path / "unet" / "diffusion_pytorch_model.safetensors").exists():
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path / "unet" / "diffusion_pytorch_model.safetensors")
                pipeline.unet.load_state_dict(state_dict)
                print("Loaded UNet weights")
            elif (checkpoint_path / "unet" / "diffusion_pytorch_model.bin").exists():
                state_dict = torch.load(checkpoint_path / "unet" / "diffusion_pytorch_model.bin")
                pipeline.unet.load_state_dict(state_dict)
                print("Loaded UNet weights")
            else:
                print("Warning: UNet weights not found in checkpoint")
    
    # Enable memory optimizations
    if device.type == "cuda":
        pipeline.enable_attention_slicing()
        print("Enabled attention slicing for memory optimization")
    
    # Run inference
    print("\n" + "=" * 80)
    final_image, all_images = inference_with_intermediate_steps(
        pipeline,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        height=args.height,
        width=args.width,
        save_dir=args.save_dir,
        save_intermediates=not args.no_intermediates,
    )
    print("=" * 80)
    
    print(f"\n✓ Generation complete!")
    print(f"  Total denoising steps: {len(all_images) if all_images else args.steps}")
    print(f"  Final image size: {final_image.size}")


if __name__ == "__main__":
    main()
