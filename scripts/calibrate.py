#!/usr/bin/env python3
"""Calibrate CleanSight on clean samples to obtain detection statistics.

Usage
-----
::

    python scripts/calibrate.py \
        --model_path liuhaotian/llava-v1.5-7b \
        --clean_data_path /path/to/clean_samples.json \
        --output_path configs/calibrated.yaml \
        --detection_layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
        --image_token_start 35 \
        --image_token_length 576 \
        --num_samples 200
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cleansight import CleanSightConfig, CleanSightDefense


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate CleanSight defense")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path or HF hub ID of the LVLM")
    parser.add_argument("--model_type", type=str, default="llava",
                        choices=["llava", "instructblip", "qwen2vl"],
                        help="Model architecture type")
    parser.add_argument("--clean_data_path", type=str, required=True,
                        help="JSON file with clean calibration samples")
    parser.add_argument("--image_dir", type=str, default="",
                        help="Base directory for image paths")
    parser.add_argument("--output_path", type=str, default="configs/calibrated.yaml",
                        help="Path to save calibrated config")
    parser.add_argument("--detection_layers", type=int, nargs="+",
                        default=list(range(10, 25)),
                        help="Layer indices for detection")
    parser.add_argument("--image_token_start", type=int, default=35)
    parser.add_argument("--image_token_length", type=int, default=576)
    parser.add_argument("--gamma_percentile", type=float, default=99.0)
    parser.add_argument("--prune_threshold", type=float, default=0.02)
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of clean samples for calibration")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_llava(model_path, device):
    from llava.model.builder import load_pretrained_model
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava", device_map=device
    )
    return model, tokenizer, image_processor


def load_clean_data(path: str, num_samples: int):
    """Load calibration data. Expected format: list of dicts with
    ``{"image": "path/to/img.jpg", "question": "..."}``
    """
    with open(path) as f:
        data = json.load(f)
    return data[:num_samples]


def main():
    args = parse_args()

    # Build config
    config = CleanSightConfig(
        detection_layers=args.detection_layers,
        image_token_start_index=args.image_token_start,
        image_token_length=args.image_token_length,
        gamma_percentile=args.gamma_percentile,
        prune_threshold=args.prune_threshold,
    )
    defense = CleanSightDefense(config)

    # Load model
    print(f"Loading model: {args.model_path}")
    if args.model_type == "llava":
        model, tokenizer, image_processor = load_llava(args.model_path, args.device)
    else:
        raise NotImplementedError(f"Model type {args.model_type} not yet supported in this script")

    # Patch model
    defense.patch(model)
    defense.set_mode("calibrate")

    # Load clean data
    samples = load_clean_data(args.clean_data_path, args.num_samples)
    print(f"Calibrating on {len(samples)} clean samples...")

    # Run calibration
    model.eval()
    with torch.no_grad():
        for sample in tqdm(samples, desc="Calibrating"):
            defense.reset()

            image_path = os.path.join(args.image_dir, sample["image"])
            image = Image.open(image_path).convert("RGB")
            question = sample["question"]

            # Prepare inputs (LLaVA-specific; adapt for other models)
            from llava.mm_utils import process_images, tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX

            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = image_tensor.to(model.device, dtype=model.dtype)

            prompt = f"<image>\n{question}"
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)

            model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=1,
                do_sample=False,
            )

    # Fit and save
    defense.fit()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    defense.save(args.output_path)
    print(f"Calibration complete. Stats saved to {args.output_path}")
    print(f"  Threshold (gamma): {defense.detector.threshold:.4f}")
    print(f"  Feature dim: {len(defense.detector.mu)}")


if __name__ == "__main__":
    main()
