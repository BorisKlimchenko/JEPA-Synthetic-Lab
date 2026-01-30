# ⚡ Adaptive-Motion-Lab

**High-Performance AnimateDiff Pipeline with Hardware-Aware Optimization.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow.svg)](https://huggingface.co/docs/diffusers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

**Adaptive-Motion-Lab** is a production-ready wrapper around the [AnimateDiff](https://github.com/guoyww/AnimateDiff) architecture. It is designed to solve the "configuration hell" of running Latent Diffusion Models across heterogeneous hardware environments (from Google Colab T4 to NVIDIA H100).

Instead of hardcoding settings, this engine uses a **Strategy Pattern** to detect available VRAM and Compute Capability, dynamically injecting the optimal attention mechanisms (SDPA vs xFormers) and VRAM management policies.

## 🚀 Key Features

### 1. Hardware-Aware Dispatch (HAL)
The engine automatically profiles the GPU at runtime:
* **Ampere+ (A100, A6000, 3090/4090):** Unlocks `HighPerformanceStrategy`. Uses native PyTorch 2.0 SDPA (`F.scaled_dot_product_attention`) for maximum throughput. Disables aggressive offloading to keep latencies low.
* **Legacy/Consumer (T4, V100, <16GB VRAM):** Activates `SurvivalStrategy`. Enforces `xformers` memory-efficient attention, enables model CPU offload, and applies VAE Slicing/Tiling to prevent OOM (Out-of-Memory) errors.

### 2. Deterministic & Reproducible
* Cross-platform seeding via CPU-based `torch.Generator`.
* Strict prompt management via JSON configuration.

### 3. Modular CLI Architecture
* **Strategy Pattern:** Clean separation of concerns (`HardwareProfile` -> `OptimizationStrategy`).
* **CLI Support:** Run different experiments using the `--prompts` argument without changing the source code.

## 🛠 Engineering Stack

| **Domain** | **Stack & Instrumentation** |
| :--- | :--- |
| **Deep Learning** | ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white) ![Diffusers](https://img.shields.io/badge/🤗_Diffusers-v0.25+-FFD21E?logo=huggingface&logoColor=black) ![xFormers](https://img.shields.io/badge/Meta-xFormers-blue) |
| **Generative R&D** | ![AnimateDiff](https://img.shields.io/badge/Model-AnimateDiff-orange) ![ControlNet](https://img.shields.io/badge/CV-ControlNet-4682B4) ![Stable Diffusion](https://img.shields.io/badge/SD-v1.5%2FXL-4B0082) ![Optimization](https://img.shields.io/badge/Task-Inference_Opt-green) |
| **Infrastructure** | ![Linux](https://img.shields.io/badge/Linux-Bash-FCC624?logo=linux&logoColor=black) ![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white) ![CUDA](https://img.shields.io/badge/NVIDIA-CUDA_Profiling-76B900?logo=nvidia&logoColor=white) ![Colab](https://img.shields.io/badge/Google-Colab_Pro-F9AB00?logo=googlecolab&logoColor=white) |
| **Architecture** | ![OOP](https://img.shields.io/badge/Pattern-OOP-lightgrey) ![SOLID](https://img.shields.io/badge/Principle-SOLID-lightgrey) ![Strategy](https://img.shields.io/badge/Design-Strategy_Pattern-9cf) ![Clean Arch](https://img.shields.io/badge/Arch-Clean_Code-success) |

## 📦 Installation

```bash
# Clone the repository
git clone [https://github.com/BorisKlimchenko/Adaptive-Motion-Lab.git](https://github.com/BorisKlimchenko/Adaptive-Motion-Lab.git)
cd Adaptive-Motion-Lab

# Install dependencies
pip install -r requirements.txt