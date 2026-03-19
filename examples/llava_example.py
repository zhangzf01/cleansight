#!/usr/bin/env python3
"""Minimal example: defend LLaVA-1.5 with CleanSight at test time.

This script demonstrates the three-step workflow:
1. Load model & patch with CleanSight
2. Calibrate on a few clean samples (or load pre-calibrated stats)
3. Run defended inference
"""

import os
import sys
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cleansight import CleanSightConfig, CleanSightDefense


def main():
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    model_path = "liuhaotian/llava-v1.5-7b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = CleanSightConfig(
        detection_layers=list(range(10, 25)),
        image_token_start_index=35,
        image_token_length=576,
        gamma_percentile=99.0,
        prune_threshold=0.02,
    )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava", device_map=device
    )

    # ------------------------------------------------------------------
    # Patch model with CleanSight
    # ------------------------------------------------------------------
    defense = CleanSightDefense(config)
    defense.patch(model)

    # ------------------------------------------------------------------
    # Option A: Calibrate (run once, save stats)
    # ------------------------------------------------------------------
    # defense.set_mode("calibrate")
    # for img_path, question in clean_samples:
    #     defense.reset()
    #     image = Image.open(img_path).convert("RGB")
    #     image_tensor = process_images([image], image_processor, model.config)
    #     image_tensor = image_tensor.to(model.device, dtype=model.dtype)
    #     input_ids = tokenizer_image_token(
    #         f"<image>\n{question}", tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    #     ).unsqueeze(0).to(model.device)
    #     model.generate(input_ids, images=image_tensor, max_new_tokens=1)
    # defense.fit()
    # defense.save("configs/calibrated.yaml")

    # ------------------------------------------------------------------
    # Option B: Load pre-calibrated stats
    # ------------------------------------------------------------------
    # defense = CleanSightDefense.load("configs/calibrated.yaml")
    # defense.patch(model)

    # ------------------------------------------------------------------
    # Defend at test time
    # ------------------------------------------------------------------
    defense.set_mode("defend")

    # Example inference
    image = Image.open("test_image.jpg").convert("RGB")
    question = "Describe this image."

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=model.dtype)

    prompt = f"<image>\n{question}"
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    defense.reset()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=512,
            do_sample=False,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Response: {response}")
    print(f"Poisoned detected: {defense.was_poisoned}")


if __name__ == "__main__":
    main()
