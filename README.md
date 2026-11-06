# 🐾 PAINR: Sparse Photoacoustic Imaging Reconstruction via INR ✨

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA"></a>
  <a href="https://github.com/NVlabs/tiny-cuda-nn"><img src="https://img.shields.io/badge/tiny--cuda--nn-v2.0-FF6F00?style=flat" alt="tiny-cuda-nn"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat" alt="License: MIT"></a>
  <a href="README_CN.md"><img src="https://img.shields.io/badge/Docs-%E4%B8%AD%E6%96%87%E6%96%87%E6%A1%A3-pink?style=flat" alt="Chinese Documentation"></a>
</p>

An efficient, self-supervised PyTorch framework for sparse and limited-view Photoacoustic Tomography (PAT) reconstruction via Implicit Neural Representations (INR) powered by `tiny-cuda-nn`.

---

## 🌸 1. What is PAINR?

**Photoacoustic Tomography (PAT / PACT)** is a super cool hybrid imaging modality! 🌟 It marries optical absorption contrast with ultrasonic tissue penetration depth, letting us visualize microvasculature deep inside living tissue without harmful ionizing radiation.

However, in real-world biomedical scanning, detectors often cannot surround the entire tissue due to geometric obstructions or high-speed scanning constraints. This leads to **limited-view** and **sparse-sampling** dilemmas:
- 💔 Traditional Universal Back-Projection (UBP) suffers from severe missing-view streak artifacts and blurred vascular margins.
- 😿 Supervised deep learning approaches demand massive amounts of paired, artifact-free ground-truth scans, which are practically unobtainable in clinical settings.

**PAINR** offers a cute, physics-grounded way out with **Implicit Neural Representations (INR)**! 🎈

- 🧁 **Zero-Shot & Self-Supervised**: Trains right on the raw sinogram of an individual scan — zero pre-training, zero external labels, zero ground-truth required!
- 🌊 **Physics-Informed Forward Operator**: Formulates acoustic wave propagation (circular Radon integrals + temporal finite difference) into a differentiable loss loop.
- ⚡ **Instant Multi-Resolution Hash Grid**: Powered by NVIDIA's `tiny-cuda-nn`, capturing fine vessel edges within seconds.
- 🎨 **Infinite Resolution Query**: Continuous spatial mapping decoupled from discrete voxel grids.

---

## 🎨 2. How It Works (Method & Pipeline)

The photoacoustic forward propagation relates initial pressure $p_0(\mathbf{r})$ to transducer pressure $p(\mathbf{r}_d, t)$ through spherical integrals and time differentiation:

$$p(\mathbf{r}_d, t) \propto \frac{\partial}{\partial t} \left[ \frac{1}{c_0 t} \int_{|\mathbf{r}-\mathbf{r}_d| = c_0 t} p_0(\mathbf{r}) \, \mathrm{d}\mathbf{r} \right]$$

PAINR matches predictions with acquired raw sensor data using dual-arc concentric sampling ($r \pm \Delta r$):

```text
       ┌─────────────────────────────────────────────────────────────┐
       │             Acquired Limited-View Sinogram                  │
       │                   p_meas(θ, t) 📡                           │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼  🐾 (MSE Loss)
┌─────────────────────────────────────┴─────────────────────────────────────┐
│                       Physics-Driven Loss Function                        │
│             L = || p_meas(θ, t) - p_pred(θ, t) ||^2                       │
└─────────────────────────────────────▲─────────────────────────────────────┘
                                      │ Backward Gradients 💫
┌─────────────────────────────────────┴─────────────────────────────────────┐
│                    Differentiable Forward PA Operator 🌊                  │
│  - Dual-radius circular arc coordinates (r + Δr, r - Δr)                  │
│  - Ray angular integration: \int p0(r) dl                                 │
│  - Numerical temporal derivative: [Proj(r+Δr) - Proj(r-Δr)] / (2 * Δt)    │
└─────────────────────────────────────▲─────────────────────────────────────┘
                                      │ Query p0(x, y) 🔍
┌─────────────────────────────────────┴─────────────────────────────────────┐
│                Implicit Neural Representation (INR) 🧠                    │
│   - Multi-Resolution Hash Encoding (tiny-cuda-nn, 14 levels) ⚡           │
│   - Fully-Fused MLP (64 neurons, 2 hidden layers, ReLU)                   │
└─────────────────────────────────────▲─────────────────────────────────────┘
                                      │ Spatial coordinates (x, y) 📍
┌─────────────────────────────────────┴─────────────────────────────────────┐
│                   Target 2D Continuous Coordinate Space                   │
└───────────────────────────────────────────────────────────────────────────┘
                                      │ Evaluation 🎉
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│           High-Quality Reconstructed Initial Pressure p0(x, y) 🖼️          │
│             (Streak artifacts suppressed, vascular details crisp)         │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ 3. Superpowers & Key Features

- 🥑 **No Prior Data Needed**: Learns on-the-fly per acquisition scan without dataset dependency.
- 🎯 **Rigorous Physical Acoustics**: Faithfully simulates wave arrival times ($r = c_0 \cdot t$) and sensor angular aperture.
- 🎀 **Clean & Artifact-Free**: Implicit smoothness priors of neural networks naturally eliminate limited-view streak noise.
- 🔍 **Flexible Zooming**: Render at $256 \times 256$, $512 \times 512$, $1024 \times 1024$, or beyond with no re-training.
- 📦 **NIfTI Out of the Box**: Easy-peasy medical image loading with SimpleITK.

---

## 🗂️ 4. Codebase Map

```text
🐾 PAINR/
├── ⚙️ configs/
│   └── config.json          # tiny-cuda-nn multi-res hash grid & MLP config
├── 🧺 data/
│   └── raw/                 # Example NIfTI sinograms
│       └── img1_128_Norm.nii    # 128-sensor normalized demo sinogram
├── 📜 dataset.py               # Dataset class, ray coordinates & sinogram loader
├── 📐 utils.py                 # Coordinate grids & arc ray rotation geometry
├── 🚀 train.py                 # Main training & reconstruction script
├── 📋 requirements.txt         # Lightweight Python dependency list
├── 📄 LICENSE                  # MIT License
├── 📘 README.md                # English documentation
└── 🌸 README_CN.md             # Chinese documentation
```

---

## 🍰 5. Cozy Setup Guide

### 🐾 Step 1: Create your Conda environment

```bash
conda create -n painr python=3.10 -y
conda activate painr
```

### 🧸 Step 2: Install PyTorch with CUDA

Match your system's CUDA version (e.g., CUDA 12.1):

```bash
# CUDA 12.1 (Recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 🎀 Step 3: Install `tiny-cuda-nn`

Build NVIDIA's fast CUDA hash-grid extension:

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

> 💡 **Warm Tips for Windows Friends**:
> - Open the **"x64 Native Tools Command Prompt for VS"** from your Windows Start Menu before running `pip install` to make sure `cl.exe` is found.
> - To speed up compilation, lock in your GPU architecture with `set TCNN_CUDA_ARCHITECTURES=86` (for RTX 30xx) or `set TCNN_CUDA_ARCHITECTURES=89` (for RTX 40xx).

### 🍭 Step 4: Install Python packages

```bash
pip install -r requirements.txt
```

### 🎈 Step 5: Test that everything works!

```bash
python -c "import torch, tinycudann, SimpleITK; print('✨ CUDA available:', torch.cuda.is_available()); print('🎉 PAINR environment is all set!')"
```

---

## 🚀 6. Let's Run It! (Quick Start)

### 🐾 Start Optimization

To start reconstructing the default 128-transducer test case:

```bash
python train.py
```

Outputs will be saved in `outputs/YYYYMMDD_HHMMSS/`:
- 🖼️ `latest_recon.png`: Real-time snapshot updated after each epoch.
- 🏆 `final_recon.png`: Crisp $1024 \times 1024$ final reconstruction.
- 📈 `tensorboard_logs/`: Real-time loss curves.

### 🍓 Peek at Training Curves (TensorBoard)

```bash
tensorboard --logdir outputs
```
Hop over to `http://localhost:6006` in your browser to watch the loss drop like magic! ✨

---

## 🎛️ 7. Parameter Knobs & Tuning

### 🧪 Physical & Simulation Knobs (`train.py`)

All tuning knobs are conveniently listed right at the beginning of `train.py`:

| Parameter | Default | Meaning / Notes |
| :--- | :--- | :--- |
| `sin_path` | `data/raw/img1_128_Norm.nii` | Path to input sinogram file (`.nii`) |
| `num_sensors` ($\theta$) | `128` | Number of detector elements / angles |
| `sample_N` | `1200` | Time samples per sensor channel |
| `radius` | `0.04` ($40\text{ mm}$) | Transducer ring radius |
| `field_of_view` | `0.04` ($40\text{ mm}$) | Physical width of the FOV |
| `speed_of_sound` | `1500` ($\text{m/s}$) | Acoustic speed in medium (tissue/water) |
| `time_step` | `5e-8` ($50\text{ ns}$) | Sampling period ($\Delta t$) |
| `delta_t` | `1e-9` ($1\text{ ns}$) | Offset for numerical temporal derivative |
| `num_rays` | `400` | Angular ray count along circular arc |
| `grid_size` | `1024` | Reconstruction matrix size ($1024 \times 1024$) |
| `num_epochs` | `80` | Optimization rounds |
| `batch_size` | `8` | Sensors per optimization mini-batch |
| `lr` | `1e-3` | Initial learning rate (with StepLR schedule) |

### 🧠 Neural Network Knobs (`configs/config.json`)

```json
{
  "encoding": {
    "otype": "Grid",
    "type": "Hash",
    "n_levels": 14,
    "n_features_per_level": 8,
    "log2_hashmap_size": 24,
    "base_resolution": 1,
    "per_level_scale": 2,
    "interpolation": "Linear"
  },
  "network": {
    "otype": "FullyFusedMLP",
    "activation": "ReLu",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 2
  }
}
```

---

## 🧺 8. Bring Your Own Data

Want to reconstruct your own photoacoustic scan? Super easy!

1. 📦 **Format**: Export your sinogram to a 2D NIfTI (`.nii`) array with shape `(num_sensors, time_samples)`.
2. 🧼 **Normalization**: Normalize amplitudes (e.g. $[0, 1]$ or standardized) for gentle, stable gradients.
3. 📐 **Hardware Matching**: Update `radius`, `speed_of_sound`, and `time_step` in `train.py` to match your physical acquisition setup.

---

## 💌 9. Citation

If you find this code, data, or research paper helpful in your work, please cite our paper:

```bibtex
@article{XIAO2025100685,
  title = {Limited-view photoacoustic imaging reconstruction via high-quality self-supervised neural representation},
  author = {Youshen Xiao and Yuting Shen and Sheng Liao and Bowei Yao and Xiran Cai and Yuyao Zhang and Fei Gao},
  journal = {Photoacoustics},
  volume = {42},
  pages = {100685},
  year = {2025},
  issn = {2213-5979},
  doi = {https://doi.org/10.1016/j.pacs.2025.100685},
  url = {https://www.sciencedirect.com/science/article/pii/S2213597925000047}
}
```

---

## 💖 10. License & Warm Thanks

- 📄 Distributed under the [MIT License](LICENSE).
- 🧸 Special thanks to [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) by NVIDIA for fast neural primitives.
- 🍰 Thanks to the [PyTorch](https://pytorch.org/) and [SimpleITK](https://simpleitk.org/) communities!
