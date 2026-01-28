# JEPA-Synthetic-Lab: Conceptual Visualization of Joint Embedding Architectures

**An experimental pipeline for visualizing the transition from stochastic hallucination to structural prediction using Latent Diffusion Models.**

---

## ⚠️ Scientific Disclaimer

**This project is a visual research tool, not an implementation of the I-JEPA or V-JEPA architectures proposed by LeCun et al.**

* **Actual JEPA:** Learns abstract representations by predicting missing information in embedding space without reconstruction.
* **This Lab:** Uses Pixel/Latent reconstruction (Autoencoding) and Temporal Attention to *visualize* the concept of stability.

The code is intended for educational demonstrations of AI safety concepts and architectural topology visualizations.



---

## 🔬 Abstract

This repository contains the implementation of **SMA-01 Core Engine**, a synthetic laboratory designed to simulate the *phenomenology* of Joint Embedding Predictive Architectures (JEPA). 

Unlike traditional JEPA implementations (which learn semantic representations via energy minimization), this project utilizes **Generative Diffusion Models (AnimateDiff + Stable Diffusion)** to visualize the theoretical shift from High-Entropy States (LLM Hallucinations) to Low-Entropy Predictive States (World Model Stability).

The pipeline maps the concept of "Prediction in Latent Space" to "Temporal Consistency in Diffusion Motion Modules".

## 📐 Mathematical & Architectural Basis

The simulation rests on the correlation between **Temporal Attention** in Video Diffusion and **Predictive Stability** in World Models.

### 1. The Visualization Proxy
We model the transition using the standard Latent Diffusion process, where the reverse process $p_\theta(x_{0:T})$ is guided to simulate structural emergence:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

* **State $\mathcal{H}$ (Hallucination):** Modeled via high classifier-free guidance scale and low temporal smoothing, resulting in stochastic variance between frames.
* **State $\mathcal{S}$ (Structure/JEPA):** Modeled via `AnimateDiff` Motion Modules (Temporal Transformers), where the attention mechanism enforces temporal coherence $C_t$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Here, increasing the `motion_scale` parameter effectively restricts the latent trajectory variance, serving as a visual metaphor for the energy-based prediction constraints in JEPA.

## 🛠 Core Features

* **SMA-01 Engine:** A modified diffusion pipeline optimizing `EulerDiscreteScheduler` for rapid latent space traversal.
* **Adaptive Compute:** Heuristic resource allocation (A100/H100 vs Consumer GPU) for VRAM management (compatible with Google Colab Pro).
* **Deterministic Sampling:** Strict seeding mechanism for `torch.Generator` to ensure reproducibility of latent noise patterns.
* **Dynamic Motion Control:** Programmatic adjustment of Motion Module weights to simulate the "focusing" of a predictive model.

## 🚀 Installation & Usage

### Prerequisites
* Python 3.10+
* CUDA 11.8+
* PyTorch 2.0+ (with `xformers` recommended for efficiency)
* Google Colab Pro (Recommended for High-VRAM generation)

### Setup
```bash
git clone [https://github.com/BorisKlimchenko/JEPA-Synthetic-Lab.git](https://github.com/BorisKlimchenko/JEPA-Synthetic-Lab.git)
cd JEPA-Synthetic-Lab
pip install -r requirements.txt