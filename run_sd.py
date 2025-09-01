#!/usr/bin/env python3
"""
Simple Stable Diffusion runner (auto CPU/GPU) with sane defaults for sd-turbo.

Examples:
  python run_sd.py --prompt "a watercolor cute robot" --out out.png
  python run_sd.py --prompt "cinematic cyberpunk alley" --steps 8 --cfg 1.8 --out neon.png
  python run_sd.py --model stabilityai/sd-turbo --seed 42 --neg "low quality, blurry"
  python run_sd.py --height 384 --width 384
"""

import argparse
from pathlib import Path
import time
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True, help="Text prompt")
    p.add_argument("--neg", default="", help="Negative prompt")
    p.add_argument("--model", default="stabilityai/sd-turbo", help="HF model id or local path")
    p.add_argument("--steps", type=int, default=6, help="Inference steps (sd-turbo works well at 4–8)")
    p.add_argument("--cfg", type=float, default=1.8, help="Guidance scale (sd-turbo: 1.0–2.0)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    p.add_argument("--out", default="sd_out.png", help="Output image path")
    p.add_argument("--unsafe", action="store_true", help="Disable safety checker (NOT recommended for public apps)")
    return p.parse_args()

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        dtype=dtype,  # (newer diffusers prefers `dtype` vs `torch_dtype`)
        safety_checker=None if args.unsafe else None  # keep default off for sd-turbo; set your policy as needed
    )
    # Use a fast scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if device == "cuda":
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing()

    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=(args.neg or None),
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        height=args.height,
        width=args.width,
        generator=generator,
    )
    img = result.images[0]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    dt = time.time() - t0
    print(f"✅ Saved: {out_path} | device={device} steps={args.steps} cfg={args.cfg} "
          f"size={args.width}x{args.height} time={dt:.1f}s")

if __name__ == "__main__":
    main()
