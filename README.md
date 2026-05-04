<div align="center">

# 🌊 WaveMamba

### Wave-Inspired Cross-Modal Fusion for Robust Event-Image Semantic Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2026-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.0-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)

Official implementation of **WaveMamba** — a dual-branch event-image semantic segmentation framework built on **GroupMamba** encoders and the proposed **Mamba-Wave Cross-Modal Fusion (MWCMF)** module.

Evaluated on **DDD17** and **DSEC-Semantic**.

</div>

---

## 📖 Overview

Event cameras complement frame images under rapid motion, low illumination, and high dynamic range — making event-image semantic segmentation a promising approach for autonomous driving and robotics. WaveMamba combines:

- **GroupMamba** encoders for hierarchical image-event representation learning
- **MWCMF** for wave-guided cross-modal fusion
- **Channel-Wave** and **Spatial-Wave** interaction branches
- A lightweight segmentation decoder for dense prediction

---

## ✨ Highlights

| Feature | Details |
|---|---|
| 🔀 Dual-Branch Encoder | GroupMamba for both image and event streams |
| 🌊 Novel Fusion Module | MWCMF for implicit spectral cross-modal propagation |
| 🏆 Strong Benchmarks | State-of-the-art results on DDD17 and DSEC-Semantic |
| 🌧️ Robustness | Improved performance under rain, fog, and low-light scenes |

---

## 🛠️ Installation

WaveMamba builds on **GroupMamba**, following its installation style (based on **VMamba**). We recommend a dedicated Conda environment.

**Requirements:**
- Python 3.10
- PyTorch ≥ 2.0
- CUDA ≥ 11.8

> Lower versions of PyTorch/CUDA may work but are not officially tested.

```bash
# Create and activate a conda environment
conda create -n wavemamba python=3.10 -y
conda activate wavemamba

# Install dependencies
pip install -r requirements.txt

# Install selective scan kernel
cd kernels/selective_scan
pip install .
cd ../..
```

<details>
<summary><b>Alternative Installation</b></summary>

If you already have a working **MambaSeg** environment, you may use it as a base. After that, install the WaveMamba-specific dependencies using `pip install -r requirements.txt` from the repository root.

</details>

---

## 📦 Datasets

### DDD17

The DDD17 dataset with semantic segmentation labels can be obtained from the **Ev-SegNet** project:

- **Ev-SegNet repository:** https://github.com/Shathe/Ev-SegNet
- **Preprocessed DDD17 (ESS project):** https://download.ifi.uzh.ch/rpg/ESS/ddd17_seg.tar.gz

> ⚠️ If you use DDD17 with semantic labels, please cite both **DDD17** and **Ev-SegNet**.

### DSEC-Semantic

Download from the official source: https://dsec.ifi.uzh.ch/dsec-semantic/

**Expected directory structure:**

```
seq_name (e.g. zurich_city_00_a)
├── semantic
│   ├── left
│   │   ├── 11classes
│   │   │   └── data
│   │   │       ├── 000000.png
│   │   │       └── ...
│   │   └── 19classes
│   │       └── data
│   │           ├── 000000.png
│   │           └── ...
│   └── timestamps.txt
├── events
│   └── left
│       ├── events.h5
│       └── rectify_map.h5
└── images
    └── left
        ├── rectified
        │   ├── 000000.png
        │   └── ...
        ├── ev_inf
        │   ├── 000000.png
        │   └── ...
        └── timestamps.txt
```

> **Note:** The `ev_inf` folder contains paired image samples spatially aligned with event data.  
> For image-event alignment issues, see https://github.com/uzh-rpg/DSEC/issues/25  
> You may also use `envimage.py` for image calibration if needed.

---

## 🔑 Pretrained Backbone

Before training, download the **GroupMamba-Small** ImageNet pretrained weights:

📥 [GroupMamba-Small weights (Google Drive)](https://drive.google.com/file/d/1vTN9ynDcsDuOVrcT9GcQ5nBSk-hXySlh/view)

Place the checkpoint in the directory expected by your config file, then update the path under `configs/` accordingly.

---

## 🚀 Training

Before training, ensure you have:
1. Set dataset paths correctly in the config files
2. Set the pretrained checkpoint path correctly
3. Updated configuration files under `configs/`

**Train on DDD17:**
```bash
python train_ddd17.py
```

**Train on DSEC:**
```bash
python train_dsec.py
```

---

## 📊 Evaluation

Before evaluation, set `EVAL.weight_path` in the corresponding config file to your target checkpoint and verify the config path used inside the evaluation script.

**Evaluate on DDD17:**
```bash
python evaluate_ddd17.py
```

**Evaluate on DSEC:**
```bash
python evaluate_dsec.py
```

---

## 📂 Pretrained Checkpoints

Download released WaveMamba checkpoints for direct evaluation:

| Dataset | Download |
|---|---|
| DDD17 | [Google Drive](https://drive.google.com/file/d/1gtT0JeS5dR7y45gYb338E1pBVOb5mVGv/view?usp=drive_link) |
| DSEC-Semantic | [Google Drive](https://drive.google.com/file/d/1y4vT-roos32DflbLGues4At63Cj9qbag/view?usp=drive_link) |

After downloading, update `EVAL.weight_path` in the relevant config file before running evaluation.

---

## 📝 Notes

- Ensure all dataset paths, checkpoint paths, and config paths are updated before training or evaluation.
- The repository assumes paired image-event data are organized exactly as described in the [Datasets](#-datasets) section.
- For DSEC, verify image-event alignment carefully, especially with custom preprocessing or calibration.
- If using a MambaSeg-based environment, verify that the selective scan kernel and all WaveMamba dependencies are correctly installed.

---

## 📄 Citation

If you find this repository useful in your research, please cite WaveMamba:

```bibtex
@article{wavemamba2026,
  title={WaveMamba: Wave-Inspired Cross-Modal Fusion for Robust Event-Image Semantic Segmentation},
  author={Adeel Feroz Mirza},
  journal={arXiv preprint},
  year={2026}
}
```

Please also cite the relevant datasets and prior works:

- DDD17
- Ev-SegNet
- DSEC-Semantic
- GroupMamba
- VMamba
- MambaSeg

---

## 🙏 Acknowledgment

This implementation builds upon prior open-source efforts in event-based semantic segmentation and state-space vision modeling, especially [GroupMamba](https://github.com/), [VMamba](https://github.com/), and [MambaSeg](https://github.com/).

---

## 📬 Contact

For questions, issues, or suggestions, please open a GitHub issue or contact the author through the email provided in the paper.
