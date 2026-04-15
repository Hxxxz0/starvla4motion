#!/usr/bin/env python3
"""
Test script to generate motion from text prompt: "a man walk forward"
Uses the trained MotionAR model to generate action sequences.
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# Monkey-patch Qwen2_5_VLConfig to add hidden_size before importing starVLA
from transformers import AutoConfig
vlm_cfg = AutoConfig.from_pretrained("./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
if hasattr(vlm_cfg, "text_config") and hasattr(vlm_cfg.text_config, "hidden_size"):
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    Qwen2_5_VLConfig.hidden_size = property(lambda self: self.text_config.hidden_size)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf
from starVLA.model.framework.base_framework import build_framework

def load_model(checkpoint_path, device="cuda"):
    """Load the trained MotionAR model."""
    print("Loading model from:", checkpoint_path)
    checkpoint_path = Path(checkpoint_path)
    config_dir = checkpoint_path.parent.parent
    config = OmegaConf.load(config_dir / "config.full.yaml")
    
    model = build_framework(config)
    print("Loading weights...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded!")
    print("  Device:", device)
    print("  Params:", sum(p.numel() for p in model.parameters()))
    return model, config

def generate_motion_from_text(model, config, text_prompt, num_chunks=20, device="cuda"):
    """Generate motion sequence from text prompt."""
    print(f"\nGenerating motion for text: '{text_prompt}'")
    
    # Create example with text prompt and empty observation latents
    example = {
        "lang": text_prompt,
        "image": [],  # No images, text-only
        "obs_latent": np.zeros((0, 64), dtype=np.float32)  # Empty initial observation
    }
    
    model.eval()
    with torch.inference_mode():
        result = model.predict_action(
            examples=[example],
            num_chunks=num_chunks,
            max_obs_frames=config.framework.max_obs_frames
        )
    
    generated = result["normalized_actions"][0]
    print(f"Generated motion shape: {generated.shape}")
    print(f"  Timesteps: {generated.shape[0]}")
    print(f"  Action dim: {generated.shape[1]}")
    
    return generated

def assess_quality(generated):
    """Assess the quality of generated motion."""
    print("\n" + "="*60)
    print("Quality Assessment")
    print("="*60)
    
    print("\n1. NaN/Inf Check:")
    nan_count = int(np.isnan(generated).sum())
    inf_count = int(np.isinf(generated).sum())
    print(f"  Shape: {generated.shape}")
    print(f"  NaN: {nan_count}, Inf: {inf_count}")
    print(f"  {'PASS' if nan_count == 0 and inf_count == 0 else 'FAIL'}")
    
    print("\n2. Value Range:")
    print(f"  Min: {float(generated.min()):.4f}")
    print(f"  Max: {float(generated.max()):.4f}")
    print(f"  Mean: {float(generated.mean()):.4f}")
    print(f"  Std: {float(generated.std()):.4f}")
    
    print("\n3. Smoothness (temporal differences):")
    if generated.shape[0] > 1:
        diffs = np.diff(generated, axis=0)
        diff_magnitudes = np.linalg.norm(diffs, axis=-1)
        print(f"  Mean diff magnitude: {float(diff_magnitudes.mean()):.4f}")
        print(f"  Max diff magnitude: {float(diff_magnitudes.max()):.4f}")
    
    return nan_count == 0 and inf_count == 0

def save_motion(generated, text_prompt, output_dir):
    """Save generated motion to npz file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_text = text_prompt.replace(" ", "_").replace("/", "_")[:30]
    filename = f"motion_{safe_text}_{timestamp}.npz"
    filepath = out_path / filename
    
    # Save with metadata
    np.savez(
        filepath,
        generated=generated,
        text_prompt=text_prompt,
        shape=generated.shape,
        timestep=generated.shape[0],
        action_dim=generated.shape[1],
        generated_at=datetime.now().isoformat()
    )
    
    abs_path = filepath.absolute()
    print(f"\nSaved to: {abs_path}")
    return str(abs_path)

def main():
    print("="*60)
    print("starVLA Text-to-Motion Generation Test")
    print("="*60)
    
    # Configuration
    checkpoint = "results/Checkpoints/motion_ar_cuda0_bs32_full_20260411/checkpoints/steps_100000_pytorch_model.pt"
    text_prompt = "a man walks in a circle"
    num_chunks = 20  # Number of chunks to generate
    output_dir = "results/text_to_motion"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    cp = Path(checkpoint)
    if not cp.is_absolute():
        cp = PROJECT_ROOT / cp
    model, config = load_model(cp, device)
    
    # Generate motion
    generated = generate_motion_from_text(model, config, text_prompt, num_chunks, device)
    
    # Assess quality
    is_valid = assess_quality(generated)
    
    # Save results
    abs_path = save_motion(generated, text_prompt, output_dir)
    
    print("\n" + "="*60)
    print(f"SUMMARY: {'Quality looks GOOD' if is_valid else 'Issues found'}")
    print("="*60)
    print(f"\nOutput file absolute path: {abs_path}")
    
    return abs_path

if __name__ == "__main__":
    main()
