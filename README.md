# TSTR-KD: Two-Stage Knowledge Distillation for Transmission-Line Fitting Detection

Official research resources for TSTR-KD, including experimental results, pretrained weights, validation data, and evaluation instructions.

## Abstract

Internet of Things (IoT)-oriented transmission-line inspection calls for accurate and lightweight fitting detection from unmanned aerial vehicle (UAV) imagery on edge devices. Deploying accurate deep learning detectors on resource-constrained devices remains challenging due to high computational complexity. Knowledge distillation can improve the detection accuracy of lightweight student models, but performance is limited by weak texture representation, background noise, and cross-structural gaps between teacher and student networks. To address these issues, this paper proposes a two-stage knowledge distillation method fusing texture reconstruction and discriminative region enhancement (TSTR-KD). A pixel-level texture alignment distillation (PTAD) mechanism reconstructs student features into high-resolution pixel space with adaptive texture attention, enhancing fine-grained texture representation. A channel-adaptive soft masked distillation (CASMD) strategy generates refined soft masks from spatial saliency and channel thresholds, guiding the student to focus on discriminative regions. The lead teacher-junior teacher-student (L-J-S) framework uses a junior teacher as a knowledge buffer to bridge the teacher-student gap. Experiments on the public electric power fitting dataset (EPFD) and the self-developed MA-Dataset demonstrate that TSTR-KD improves lightweight student detection accuracy on both transmission-line fitting benchmarks.

## Experimental Results

### Baseline

| Run | mAP (%) | AP50 (%) | AP75 (%) |
|---|---:|---:|---:|
| Run 1 | 44.1698 | 73.7394 | 46.7116 |
| Run 2 | 44.3541 | 73.5874 | 49.0202 |
| Run 3 | 44.0745 | 73.4931 | 47.9725 |
| **Mean ± SD** | **44.20 ± 0.14** | **73.61 ± 0.12** | **47.90 ± 1.16** |

### TSTR-KD

| Run | mAP (%) | AP50 (%) | AP75 (%) |
|---|---:|---:|---:|
| Run 1 | 46.5175 | 77.5328 | 49.9367 |
| Run 2 | 46.5045 | 77.7631 | 49.9480 |
| Run 3 | 46.5532 | 78.2112 | 51.0022 |
| **Mean ± SD** | **46.53 ± 0.03** | **77.84 ± 0.34** | **50.30 ± 0.61** |

The reported results summarize three experimental runs for each method. Mean values and standard deviations are reported to account for experimental variability.

## Pretrained Weights

Pretrained checkpoints for the baseline model and TSTR-KD are available for model evaluation and research reproducibility.

All six checkpoints can be downloaded from the [TSTR-KD Pretrained Weights Release (v1.0.0)](https://github.com/ncepuzzy/TSTR-KD/releases/tag/v1.0.0).

| Model | Run | Download |
|---|---|---|
| Baseline | Run 1 | [baseline_run1.pt](https://github.com/ncepuzzy/TSTR-KD/releases/download/v1.0.0/baseline_run1.pt) |
| Baseline | Run 2 | [baseline_run2.pt](https://github.com/ncepuzzy/TSTR-KD/releases/download/v1.0.0/baseline_run2.pt) |
| Baseline | Run 3 | [baseline_run3.pt](https://github.com/ncepuzzy/TSTR-KD/releases/download/v1.0.0/baseline_run3.pt) |
| TSTR-KD | Run 1 | [tstr_kd_run1.pt](https://github.com/ncepuzzy/TSTR-KD/releases/download/v1.0.0/tstr_kd_run1.pt) |
| TSTR-KD | Run 2 | [tstr_kd_run2.pt](https://github.com/ncepuzzy/TSTR-KD/releases/download/v1.0.0/tstr_kd_run2.pt) |
| TSTR-KD | Run 3 | [tstr_kd_run3.pt](https://github.com/ncepuzzy/TSTR-KD/releases/download/v1.0.0/tstr_kd_run3.pt) |

The released checkpoints correspond to the experimental runs reported in the results section. Please use the corresponding model configuration and evaluation protocol when reproducing the reported results.

## Validation Dataset


We use the validation split of the publicly available Electric Power Fitting Dataset (EPFD) to evaluate the baseline model and TSTR-KD.

- **Dataset:** Electric Power Fitting Dataset (EPFD)
- **Data split:** Validation set
- **Contents:** Validation images and corresponding annotations
- **Original Dataset Download:** [EPFD Dataset](https://github.com/YannickKeane/EPFD)

The same validation split and evaluation protocol are used for both the baseline model and TSTR-KD to ensure a fair comparison. Please refer to the original dataset documentation and applicable data usage terms.

## Evaluation

Evaluation requires the YOLOv7 codebase. Copy the released `test_epfd.py` and the complete `evaluation/` directory into the YOLOv7 repository root, alongside `models/` and `utils/`:

```text
yolov7/
├── models/
├── utils/
├── test_epfd.py
└── evaluation/
    ├── validate_epfd.py
    ├── coco_metrics.py
    ├── checkpoints.json
    ├── dataset_reference.json
    └── source_reference.json
```

### Preparation

Install the YOLOv7 dependencies and the COCO evaluation package:

```bash
pip install -r requirements.txt
pip install pycocotools==2.0.7
```

Place the six released checkpoints in `weights/`. Prepare the original EPFD validation images and corresponding YOLO-format labels as follows:

```text
datasets/EPFD/images/val/
datasets/EPFD/labels/val/
```

### Evaluation Commands

Run the following commands from the YOLOv7 repository root using trusted released checkpoints.

Evaluate a baseline checkpoint:

```bash
python test_epfd.py --weights weights/baseline_run1.pt --data-root datasets/EPFD --device 0 --output results/baseline_run1 --allow-different-source --trust-checkpoints
```

Evaluate a distilled checkpoint:

```bash
python test_epfd.py --weights weights/tstr_kd_run1.pt --data-root datasets/EPFD --device 0 --output results/tstr_kd_run1 --allow-different-source --trust-checkpoints
```

Replace `run1` with `run2` or `run3` in both the checkpoint path and output directory to evaluate the remaining checkpoints.


## Code Availability

The source code of TSTR-KD will be publicly released upon acceptance of the paper. Currently, this repository provides the pretrained model weights and experimental results. The source code and implementation details will be made available to facilitate reproducibility after acceptance.

## Citation

The full citation will be added when the publication details are finalized.

## License and Data Availability

Licensing terms for the source code, pretrained weights, and validation data will be specified separately as appropriate. Please check the applicable permissions before using or redistributing these materials.
