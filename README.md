# SC-DAG

This repository provides the implementation of **SC-DAG** proposed in:

> **SC-DAG: Semantic-Constrained Diffusion Attacks for Stealthy Exposure Manipulation in Visually-Aware Recommender Systems**,
> *CIKM 2025*

The code focuses on **generating adversarial images under semantic constraints** using a latent diffusion backbone.
Evaluation in the paper is conducted using the **open-source AIP  and its evaluation pipeline**.

---

## Repository Structure

```
SC-DAG/
├── generate.py            # Main script for adversarial image generation
├── configs/               # Stable Diffusion (inpainting) configuration files
├── SemanticMask/          # Semantic segmentation for foreground masking
├── utils.py                
├── ldm/
└── README.md
```

---

## Requirements

* Python 3.8+
* PyTorch
* torchvision, numpy, Pillow

> A Stable Diffusion v2 inpainting checkpoint is required but **not included**.
> Please download it from the official source and place it under `pretrained_model/`.

---

## Usage

### Generate adversarial images

```bash
python generate.py \
  --model ResNet152 \
  --t 250 \
  --s 300 \
  --num 1 \
  --save res/
```

**Arguments:**

* `--model` : visual encoder used for adversarial guidance
* `--t` : diffusion timestep for latent initialization
* `--s` : adversarial guidance strength
* `--num` : number of samples to generate
* `--save` : output directory

Generated images are saved to:

```
res/img/
```

---

## Notes

* Semantic masks are computed during sampling to restrict perturbations to foreground regions.
* Evaluation of generated adversarial images follows the **open-source AIP recommendation and evaluation framework**, as described in the paper.
* The released code is intended as a **research reference implementation**.

---

## Disclaimer

This code is released for **academic research purposes only**.
Please use responsibly.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{lin2025scdag,
  title={SC-DAG: Semantic-Constrained Diffusion Attacks for Stealthy Exposure Manipulation in Visually-Aware Recommender Systems},
  author={Lin, Ze and Qian, Yuqiu and Li, Xiaodong and Lyu, Ziyu and Li, Hui},
  booktitle={Proceedings of the ACM International Conference on Information and Knowledge Management (CIKM)},
  year={2025}
}
```

---

## Contact

Ze Lin
Email: [linze@stu.xmu.edu.cn](mailto:linze@stu.xmu.edu.cn)

