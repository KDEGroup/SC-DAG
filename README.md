# SC-DAG — Semantic-Constrained Diffusion Attacks for Stealthy Exposure Manipulation in Visually-Aware Recommender Systems

**Formal / Academic README (English)** — suitable for the repository front page and paper supplement.

---

## Overview

**SC-DAG** (Semantic-Constrained Diffusion Adversarial Generation) is a latent diffusion–based framework designed to generate visually imperceptible adversarial images that increase target item exposure in visually-aware recommender systems (VARS). SC-DAG concentrates perturbations on semantically meaningful foreground regions via contour conditioning and steers latent sampling using reference images selected by a hybrid popularity–semantic similarity criterion.

This repository contains the official implementation used in the CIKM 2025 paper:

> Lin, Z., Qian, Y., Li, X., Lyu, Z., & Li, H. (2025). *SC-DAG: Semantic-Constrained Diffusion Attacks for Stealthy Exposure Manipulation in Visually-Aware Recommender Systems.* CIKM 2025.

---

## Contents

```
SC-DAG/
├─ data/                      # Reference image selection scripts
├─ FaceParsing/               # Semantic segmentation utilities
├─ models/                    # pretrained models / encoders (links or checkpoints)
├─ generate.py                # main SC-DAG generation script
├─ eval.py                    # evaluation harness for VARS attack assessment
├─ requirements.txt           # suggested Python deps
├─ res/                       # default output directory
└─ README.md                  # this file
```

---

## Requirements

* Python 3.8+
* PyTorch (compatible with your CUDA)
* Typical packages: `numpy`, `tqdm`, `Pillow`, `opencv-python`, `scikit-image`, `torchvision`, `transformers`, `diffusers` (or the repo's listed `requirements.txt`)

> See `requirements.txt` for exact versions. We recommend running in a virtual environment.

---

## Quickstart (Commands)

1. **Reference image retrieval**

   ```bash
   python SC-DAG/data/get_ref.py
   ```

   Produces the reference image(s) used by the hybrid reference selection module.

2. **Semantic mask generation**

   ```bash
   python SC-DAG/FaceParsing/semantic.py
   ```

   Produces binary/semantic masks (foreground vs. background) for target images.

3. **Adversarial image generation**

   ```bash
   python generate.py --model ResNet152 --t 250 --save res/ --num 1
   ```

   * `--model`: visual encoder used for feature extraction (e.g., `ResNet152`).
   * `--t`: number of diffusion sampling steps (example: 250).
   * `--save`: directory to store results (images will appear in `res/img/`).
   * `--num`: number of adversarial images to generate (per target).

   Generated images are saved under:

   ```
   res/img/
   ```

4. **Recommendation evaluation**

   ```bash
   python SC-DAG/eval.py --adv_item_path res/img
   ```

   Replace the image path with your generated image(s). `eval.py` runs the chosen VARS victim models and reports metrics such as HR@K and FID.

---

## Configuration & Hyperparameters

Key hyperparameters implemented in the code:

* `s` — attack strength scalar for adversarial gradient injection.
* `t` — diffusion sampling steps (larger t often yields finer control; tradeoff with runtime).
* `alpha` (`α`) — balancing factor for hybrid reference selection (popularity vs. semantic similarity).
* `lambda` (`λ`) — regularization weight preserving original semantics.
* `num` — number of outputs per target item.

Default values used in experiments (paper): `s=300`, `t=51` (paper experiments); in released code you may use larger `t` for higher fidelity.

---

## Implementation Notes

* SC-DAG employs a latent diffusion backbone (Stable Diffusion v2 inpainting checkpoint used in experiments) and DeepLabV3 (ResNet-101) for semantic segmentation.
* The hybrid reference selector ranks candidates by normalized popularity and CLIP-based visual similarity; the selected reference guides adversarial gradient direction.
* The adversarial gradient is computed in latent space and injected iteratively during denoising to steer the sample towards features of popular & semantically similar items.
* Evaluation follows a two-stage recommendation pipeline (BPR for candidate selection + VARS model for ranking) and reports HR@K for target and non-target items, and FID for visual stealth.

---

## Datasets & Victim Models

The experiments in the paper used publicly available datasets commonly adopted in VARS research (examples described in the paper):

* **Amazon Men’s Clothing** (subset)
* **Tradesy.com**

Victim models included: **VBPR**, **DVBPR**, and **AMR**.

Links and dataset processing scripts (if any) are provided in `data/` or documented in repository notes. Please follow license/usage rules of each dataset.

---

## Reproducing Paper Results

1. Prepare datasets and preprocess as described in the paper.
2. Place pretrained encoder/decoder/checkpoints in `models/` or point to Hugging Face checkpoints as configured.
3. Run the provided training/evaluation scripts in the order listed in *Quickstart*.
4. Use the same hyperparameters reported in the paper for comparable results.

---

## Ethical Statement & Disclaimer

This repository is released for **academic and defensive research** on recommender system robustness. The methods herein can reveal vulnerabilities in VARS and should be used responsibly.

**Do not** apply these techniques against systems without explicit authorization. The authors / repository maintainers assume no responsibility for misuse.

---

## Citation

If you use this code or dataset processing for research, please cite:

```bibtex
@inproceedings{lin2025scdag,
  title={SC-DAG: Semantic-Constrained Diffusion Attacks for Stealthy Exposure Manipulation in Visually-Aware Recommender Systems},
  author={Lin, Ze and Qian, Yuqiu and Li, Xiaodong and Lyu, Ziyu and Li, Hui},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM ’25)},
  year={2025}
}
```

---

## Contact

Main author: Ze Lin — `linze@stu.xmu.edu.cn`
Project page / code: `https://github.com/KDEGroup/SC-DAG`

