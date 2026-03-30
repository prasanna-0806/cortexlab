---
license: cc-by-nc-4.0
library_name: cortexlab
tags:
  - neuroscience
  - fmri
  - brain-encoding
  - multimodal
  - tribe-v2
  - brain-alignment
  - cognitive-load
language:
  - en
pipeline_tag: other
---

# CortexLab

Enhanced multimodal fMRI brain encoding toolkit built on [Meta's TRIBE v2](https://github.com/facebookresearch/tribev2).

CortexLab extends TRIBE v2 with streaming inference, interpretability tools, cross-subject adaptation, brain-alignment benchmarking, and cognitive load scoring.

## What This Repo Contains

This is a **code-only** repository. It does not contain pretrained weights. The pretrained TRIBE v2 model is hosted by Meta at [`facebook/tribev2`](https://huggingface.co/facebook/tribev2).

## Features

| Feature | Description |
|---|---|
| **Streaming Inference** | Sliding-window real-time predictions from live feature streams |
| **ROI Attention Maps** | Visualize which brain regions attend to which temporal moments |
| **Modality Attribution** | Per-vertex importance scores for text, audio, and video |
| **Cross-Subject Adaptation** | Ridge regression or nearest-neighbour adaptation for new subjects |
| **Brain-Alignment Benchmark** | Score how "brain-like" any AI model's representations are (RSA, CKA, Procrustes) |
| **Cognitive Load Scorer** | Predict cognitive demand of media from predicted brain activation patterns |

## Prerequisites

The pretrained TRIBE v2 model uses **LLaMA 3.2-3B** as its text encoder. You must:

1. Accept Meta's LLaMA license at [llama.meta.com](https://llama.meta.com/)
2. Request access on [HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-3B)
3. Authenticate: `huggingface-cli login`

## Installation

```bash
git clone https://github.com/siddhant-rajhans/cortexlab.git
cd cortexlab
pip install -e ".[analysis]"
```

## Quick Start

### Inference

```python
from cortexlab.inference.predictor import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", device="auto")
events = model.get_events_dataframe(video_path="clip.mp4")
preds, segments = model.predict(events)
```

### Brain-Alignment Benchmark

```python
from cortexlab.analysis import BrainAlignmentBenchmark

bench = BrainAlignmentBenchmark(brain_predictions, roi_indices=roi_indices)
result = bench.score_model(clip_features, method="rsa")
print(f"Alignment: {result.aggregate_score:.3f}")
```

### Cognitive Load Scoring

```python
from cortexlab.analysis import CognitiveLoadScorer

scorer = CognitiveLoadScorer(roi_indices)
result = scorer.score_predictions(predictions)
print(f"Overall load: {result.overall_load:.2f}")
```

## Compute Requirements

| Component | VRAM | Notes |
|---|---|---|
| TRIBE v2 encoder | ~1 GB | Small (1.15M params) |
| LLaMA 3.2-3B (text) | ~8 GB | Features cached after first run |
| V-JEPA2 (video) | ~6 GB | Features cached after first run |
| Wav2Vec-BERT (audio) | ~3 GB | Features cached after first run |

Minimum: 16 GB VRAM GPU for full inference. CPU works but is slow. Analysis tools (benchmark, cognitive load) work with zero GPU on precomputed predictions.

## Architecture

```
src/cortexlab/
  core/          Model architecture, attention extraction, subject adaptation
  data/          Dataset loading, transforms, HCP ROI utilities
  training/      PyTorch Lightning training pipeline
  inference/     Predictor, streaming, modality attribution
  analysis/      Brain-alignment benchmark, cognitive load scorer
  viz/           Brain surface visualization (nilearn, pyvista)
```

## License

CC BY-NC 4.0 (non-commercial use only), inherited from TRIBE v2.

This project does not redistribute pretrained weights. Users must download weights directly from [`facebook/tribev2`](https://huggingface.co/facebook/tribev2).

## Citation

If you use CortexLab in your research, please cite the original TRIBE v2 paper:

```bibtex
@article{dascoli2026tribe,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, St{\'e}phane and others},
  year={2026}
}
```

## Links

- **GitHub**: [siddhant-rajhans/cortexlab](https://github.com/siddhant-rajhans/cortexlab)
- **TRIBE v2**: [facebookresearch/tribev2](https://github.com/facebookresearch/tribev2)
- **Pretrained weights**: [facebook/tribev2](https://huggingface.co/facebook/tribev2)
