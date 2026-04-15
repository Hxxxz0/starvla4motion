#!/usr/bin/env python3
import os, sys, json, argparse
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

from starVLA.dataloader.motion_latent_datasets import MotionLatentDataset
from omegaconf import OmegaConf
from starVLA.model.framework.base_framework import build_framework

def load_model(checkpoint_path, device="cuda"):
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

def load_test_data(data_root, num_samples=5, chunk_size=15, max_obs_frames=150):
    print("Loading data from:", data_root)
    dataset = MotionLatentDataset(data_root_dir=data_root, split="test", chunk_size=chunk_size, max_obs_frames=max_obs_frames, max_motion_length=600, fps=20)
    print("Dataset:", len(dataset), "samples")
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    return [dataset[i] for i in indices], dataset

def run_inference(model, samples, config, device, num_chunks=20):
    print("Running inference...")
    generated, ground_truth = [], []
    model.eval()
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            example = {"lang": sample["lang"], "image": sample.get("image", []), "obs_latent": sample["obs_latent"]}
            try:
                result = model.predict_action(examples=[example], num_chunks=num_chunks, max_obs_frames=config.framework.max_obs_frames)
                generated.append(result["normalized_actions"][0])
                ground_truth.append(sample["action"])
            except Exception as e:
                print("Error sample", i, ":", e)
                continue
    print("Generated:", len(generated), "samples")
    return generated, ground_truth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/Checkpoints/motion_ar_cuda0_bs32_full_20260411/checkpoints/steps_100000_pytorch_model.pt")
    parser.add_argument("--data-root", default="/limx_embap/tos/user/Jensen/dataset/robot_humanml_data_v2")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--num-chunks", type=int, default=20)
    parser.add_argument("--output-dir", default="results/test_generation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("="*60)
    print("starVLA Motion Generation Test")
    print("="*60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    
    cp = Path(args.checkpoint)
    if not cp.is_absolute(): cp = PROJECT_ROOT / cp
    model, config = load_model(cp, device)
    
    samples, dataset = load_test_data(Path(args.data_root), args.num_samples, config.framework.action_model.action_horizon, config.framework.max_obs_frames)
    generated, gt = run_inference(model, samples, config, device, args.num_chunks)
    
    if not generated:
        print("No actions generated!")
        sys.exit(1)
    
    print("")
    print("="*60)
    print("Quality Assessment")
    print("="*60)
    
    all_gen = np.stack(generated, axis=0)
    
    print("")
    print("1. NaN/Inf:")
    nan_c, inf_c = int(np.isnan(all_gen).sum()), int(np.isinf(all_gen).sum())
    print("  Shape:", all_gen.shape)
    print("  NaN:", nan_c, "Inf:", inf_c)
    print("  PASS" if nan_c==0 and inf_c==0 else "  FAIL")
    
    print("")
    print("2. Range:")
    print("  Min:", float(all_gen.min()))
    print("  Max:", float(all_gen.max()))
    print("  Mean:", float(all_gen.mean()))
    print("  Std:", float(all_gen.std()))
    
    print("")
    print("3. Smoothness:")
    diffs = np.diff(all_gen, axis=1)
    dm = np.linalg.norm(diffs, axis=-1)
    print("  Mean diff:", float(dm.mean()))
    print("  Max diff:", float(dm.max()))
    
    print("")
    print("4. Diversity:")
    if len(generated) >= 2:
        flat = [a.flatten() for a in generated]
        dists = [np.linalg.norm(flat[i]-flat[j]) for i in range(len(flat)) for j in range(i+1, len(flat))]
        if dists:
            print("  Mean distance:", float(np.mean(dists)))
    
    print("")
    print("5. Ground Truth:")
    all_gt = np.stack(gt, axis=0)
    print("  Min:", float(all_gt.min()), "Max:", float(all_gt.max()))
    print("  Mean:", float(all_gt.mean()), "Std:", float(all_gt.std()))
    
    mses = [np.mean((g[:min(len(g),len(t))]-t[:min(len(g),len(t))])**2) for g,t in zip(generated,gt)]
    print("  Avg MSE:", float(np.mean(mses)))
    
    print("")
    print("="*60)
    print("SUMMARY: Quality looks GOOD" if nan_c==0 and inf_c==0 else "SUMMARY: Issues found")
    print("="*60)
    
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out/"report.json", "w") as f:
        json.dump({"nan":nan_c, "inf":inf_c, "shape":str(all_gen.shape), "mean":float(all_gen.mean()), "std":float(all_gen.std())}, f, indent=2)
    np.savez(out/"actions.npz", generated=np.array(generated, dtype=object), gt=np.array(gt, dtype=object))
    print("Saved to:", out)
    print("Done!")

if __name__ == "__main__":
    main()
