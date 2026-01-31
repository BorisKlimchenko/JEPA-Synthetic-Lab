# ⚡ Adaptive-Motion-Lab

**High-Performance AnimateDiff Pipeline with Hardware-Aware Optimization.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow.svg)](https://huggingface.co/docs/diffusers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

**Adaptive-Motion-Lab** is a production-engineered wrapper for the [AnimateDiff](https://github.com/guoyww/AnimateDiff) architecture, specifically designed to mitigate "configuration fatigue" when deploying Latent Diffusion Models across heterogeneous hardware environments—ranging from entry-level T4 instances to high-end NVIDIA H100 clusters.

At its core, the engine utilizes a **Strategy Design Pattern** to perform runtime hardware detection. It dynamically orchestrates VRAM allocation policies and injects optimized attention mechanisms (such as **PyTorch SDPA** or **xFormers**) based on the detected Compute Capability and available memory overhead.

## 🏗️ Architecture

The engine implements a **Hardware Abstraction Layer (HAL)** that dynamically selects the execution strategy at runtime.

```mermaid
classDiagram
    class AdaptiveInferenceEngine {
        +run(scene_name)
        -_load_prompts()
        -_build_pipeline()
    }

    class HardwareProfile {
        +device : str
        +vram_gb : float
        +compute_capability : float
        +is_high_performance_node() bool
    }

    class InferenceStrategy {
        <<Interface>>
        +configure_pipeline()
        +get_resolution_limit()
    }

    class HighPerformanceStrategy {
        +Native SDPA (FlashAttention)
        +Max Resolution: 1024x1024
    }

    class ConsumerStrategy {
        +xFormers Memory Efficient
        +Model CPU Offload
        +VAE Slicing
        +Max Resolution: 512x512
    }

    AdaptiveInferenceEngine *-- HardwareProfile : Composition
    AdaptiveInferenceEngine o-- InferenceStrategy : Aggregation
    HardwareProfile --> InferenceStrategy : Determines Factory Output
    InferenceStrategy <|-- HighPerformanceStrategy : Implements
    InferenceStrategy <|-- ConsumerStrategy : Implements
```    

## 🎬 Gallery

Demonstration of **Adaptive Inference** pipeline stages running on **NVIDIA T4** (Survival Mode):

* **Act 1: Latent Initialization** — Seeding the latent space and hardware profiling.
* **Act 2: Adaptive Motion Synthesis** — Generating temporal dynamics using memory-efficient attention (xFormers).
* **Act 3: Temporal Consistency** — Final stabilization and artifact refinement.

| Act 1: Noise Initialization | Act 2: Motion Synthesis |
| :---: | :---: |
| ![Chaos Mode](assets/Act_1_Chaos_42.gif) | ![Flow Mode](assets/Act_2_JEPA_Flow_108.gif) |
| *Running on T4 (Survival Strategy)* | *Running on T4 (Survival Strategy)* |

<details>
<summary>👁️ <b>View Act 3: Final Stabilization</b></summary>
<br>

![Structure Mode](assets/Act_3_Structure_777.gif)
*Result of full optimization pipeline on T4*
</details>

### Hardware Strategy Comparison

The engine automatically selects the best strategy based on your GPU:

| Strategy | Target Hardware | Optimization Features |
| :--- | :--- | :--- |
| **Survival Strategy** (Active in Demo) | NVIDIA T4 / Consumer GPUs | xFormers, CPU Offload, VAE Slicing |
| **High-Perf Strategy** | NVIDIA A100 / H100 / 4090 | Native SDPA (FlashAttention), Max Throughput |

## 🚀 Key Features

### 1. Hardware-Aware Dispatch (HAL)
The engine automatically profiles the GPU at runtime:
* **Ampere+ (A100, 3090+):** Unlocks `HighPerformanceStrategy` (Native PyTorch 2.0 SDPA).
* **Legacy/Consumer (T4, V100):** Activates `SurvivalStrategy` (xFormers + CPU Offload).

### 2. Deterministic & Reproducible
* Cross-platform seeding via CPU-based `torch.Generator`.
* Strict prompt management via JSON configuration.

## 📦 Installation & Usage

### 1. Cloud Execution (Google Colab)
For users without high-end local GPUs, use the provided launcher:
1. Open `notebooks/Colab_Launcher.ipynb` in GitHub.
2. Click the **"Open in Colab"** button or download the notebook to Drive.

### 2. Local Development
```bash
# Clone the repository
git clone [https://github.com/BorisKlimchenko/Adaptive-Motion-Lab.git](https://github.com/BorisKlimchenko/Adaptive-Motion-Lab.git)
cd Adaptive-Motion-Lab

# Install dependencies
pip install -r requirements.txt

# Run Inference (using default config)
python main.py

# Run Inference (using custom config)
python main.py --prompts configs/default_scene.json