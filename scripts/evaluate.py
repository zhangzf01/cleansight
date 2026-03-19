#!/usr/bin/env python3
"""Evaluate CleanSight defense: measure detection AUROC and defended accuracy.

Usage
-----
::

    python scripts/evaluate.py \
        --model_path liuhaotian/llava-v1.5-7b \
        --config_path configs/calibrated.yaml \
        --test_data_path /path/to/test_data.json \
        --image_dir /path/to/images
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cleansight import CleanSightConfig, CleanSightDefense


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CleanSight defense")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="llava",
                        choices=["llava", "instructblip", "qwen2vl"])
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to calibrated CleanSight config YAML")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="JSON with test samples; each has 'is_poisoned' field")
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def compute_auroc(labels, scores):
    """Compute AUROC without sklearn dependency."""
    labels = np.array(labels)
    scores = np.array(scores)

    # Sort by descending score
    order = np.argsort(-scores)
    labels = labels[order]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = 0
    fp = 0
    auc = 0.0
    for label in labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp

    return auc / (n_pos * n_neg)


def main():
    args = parse_args()

    # Load defense
    defense = CleanSightDefense.load(args.config_path)

    # Load model
    print(f"Loading model: {args.model_path}")
    if args.model_type == "llava":
        from llava.model.builder import load_pretrained_model
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, None, "llava", device_map=args.device
        )
    else:
        raise NotImplementedError(f"Model type {args.model_type} not yet supported")

    # Patch and set to defend mode
    defense.patch(model)
    defense.set_mode("defend")

    # Load test data
    with open(args.test_data_path) as f:
        test_data = json.load(f)

    print(f"Evaluating on {len(test_data)} samples...")

    labels = []
    scores = []
    detections = []
    results = []

    model.eval()
    with torch.no_grad():
        for sample in tqdm(test_data, desc="Evaluating"):
            defense.reset()

            is_bd = sample.get("is_poisoned", False)
            image_path = os.path.join(args.image_dir, sample["image"])
            image = Image.open(image_path).convert("RGB")
            question = sample["question"]

            # Prepare inputs (LLaVA-specific)
            from llava.mm_utils import process_images, tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX

            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = image_tensor.to(model.device, dtype=model.dtype)

            prompt = f"<image>\n{question}"
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)

            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Collect detection results
            labels.append(int(is_bd))
            detected = defense.was_poisoned
            detections.append(int(detected))

            # Compute anomaly score for AUROC
            feat_list = [
                defense._layer_ratios[l]
                for l in defense.config.detection_layers
                if l in defense._layer_ratios
            ]
            if feat_list:
                feature = torch.cat(feat_list, dim=0).numpy()
                score = defense.detector.score(feature)
            else:
                score = 0.0
            scores.append(score)

            results.append({
                "is_poisoned": is_bd,
                "detected": detected,
                "score": score,
                "response": response,
            })

    # Compute metrics
    labels_arr = np.array(labels)
    detections_arr = np.array(detections)

    n_clean = (labels_arr == 0).sum()
    n_poison = (labels_arr == 1).sum()

    if n_poison > 0:
        tp = ((labels_arr == 1) & (detections_arr == 1)).sum()
        detection_rate = tp / n_poison
    else:
        detection_rate = float("nan")

    if n_clean > 0:
        fp = ((labels_arr == 0) & (detections_arr == 1)).sum()
        false_positive_rate = fp / n_clean
    else:
        false_positive_rate = float("nan")

    auroc = compute_auroc(labels, scores)

    print("\n" + "=" * 50)
    print("CleanSight Evaluation Results")
    print("=" * 50)
    print(f"Total samples:       {len(labels)}")
    print(f"Clean / Poisoned:    {n_clean} / {n_poison}")
    print(f"Detection Rate:      {detection_rate:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")
    print(f"AUROC:               {auroc:.4f}")
    print("=" * 50)

    # Save results
    output_dir = os.path.dirname(args.config_path) or "."
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "metrics": {
                "detection_rate": float(detection_rate),
                "false_positive_rate": float(false_positive_rate),
                "auroc": float(auroc),
            },
            "per_sample": results,
        }, f, indent=2)
    print(f"Detailed results saved to {results_path}")


if __name__ == "__main__":
    main()
