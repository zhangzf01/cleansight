# CleanSight

**Test-Time Attention Purification for Backdoored Large Vision Language Models**

> Zhifang Zhang, Bojun Yang, Shuo He, Weitong Chen, Wei Emma Zhang, Olaf Maennel, Lei Feng, Miao Xu

Thanks for the interest of our [[paper]](https://arxiv.org/abs/2603.12989) accepted at CVPR 2026 

CleanSight is a **training-free, plug-and-play** defense that operates purely at test time to protect Large Vision-Language Models (LVLMs) against backdoor attacks. It detects poisoned inputs by measuring abnormal vision-to-text attention redistribution and neutralizes backdoor activation by pruning suspicious visual tokens.

## Key Idea: Attention Stealing

Backdoor triggers in LVLMs do not influence prediction through low-level visual patterns, but through **abnormal cross-modal attention redistribution** — trigger-bearing visual tokens "steal" attention away from the textual context. CleanSight exploits this by:

1. **Detecting** poisoned inputs via whitened L2 distance of per-head vision-to-text attention ratios
2. **Pruning** suspicious high-attention visual tokens by adding large negative bias to attention logits

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from cleansight import CleanSightConfig, CleanSightDefense

# 1. Configure
config = CleanSightConfig(
    detection_layers=list(range(10, 25)),  # layers for detection
    image_token_start_index=35,            # model-specific
    image_token_length=576,                # e.g., 576 for LLaVA-1.5
)
defense = CleanSightDefense(config)

# 2. Patch the model (model-agnostic, no source code changes)
defense.patch(model)

# 3. Calibrate on ~200 clean samples
defense.set_mode("calibrate")
for batch in clean_loader:
    defense.reset()
    model.generate(**batch)
defense.fit()
defense.save("configs/calibrated.yaml")

# 4. Defend at test time
defense.set_mode("defend")
for batch in test_loader:
    defense.reset()
    output = model.generate(**batch)
    if defense.was_poisoned:
        print("Backdoor detected and neutralized!")
```

## Project Structure

```
cleansight/
├── cleansight/
│   ├── __init__.py        # Public API
│   ├── config.py          # CleanSightConfig dataclass
│   ├── detector.py        # Anomaly detector (whitened L2 distance)
│   ├── pruner.py          # Attention pruner (token masking)
│   ├── defense.py         # Main CleanSightDefense orchestrator
│   └── hook.py            # Model-agnostic attention patching
├── scripts/
│   ├── calibrate.py       # CLI: calibrate on clean samples
│   └── evaluate.py        # CLI: evaluate detection & defense
├── configs/
│   └── default.yaml       # Default hyperparameters
├── examples/
│   └── llava_example.py   # End-to-end LLaVA example
├── setup.py
└── README.md
```

## Calibration

Calibrate CleanSight on clean samples to obtain detection statistics:

```bash
python scripts/calibrate.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --clean_data_path /path/to/clean_samples.json \
    --output_path configs/calibrated.yaml \
    --num_samples 200
```

The clean data JSON should be a list of `{"image": "path.jpg", "question": "..."}`.

## Evaluation

```bash
python scripts/evaluate.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --config_path configs/calibrated.yaml \
    --test_data_path /path/to/test_data.json \
    --image_dir /path/to/images
```

Test data JSON entries should include an `is_poisoned` boolean field for AUROC computation.

## Supported Models

CleanSight is architecture-agnostic. Tested on:

- **LLaVA-1.5** (7B / 13B)
- **InstructBLIP**
- **Qwen2-VL** (2B / 7B)
- **Qwen3-VL** (2B / 8B / 32B)

## Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `detection_layers` | Layer indices for attention ratio extraction | 10–24 |
| `image_token_start_index` | Start position of visual tokens | 35 |
| `image_token_length` | Number of visual tokens | 576 |
| `gamma_percentile` | Detection threshold percentile | 99 |
| `prune_threshold` (τ) | Per-head attention threshold for pruning | 0.02 |

## Citation

```bibtex
@article{zhang2026cleansight,
  title={Test-Time Attention Purification for Backdoored Large Vision Language Models},
  author={Zhang, Zhifang and Yang, Bojun and He, Shuo and Chen, Weitong and Zhang, Wei Emma and Maennel, Olaf and Feng, Lei and Xu, Miao},
  journal={arXiv preprint arXiv:2603.12989},
  year={2026}
}
```

## License

Apache License 2.0
