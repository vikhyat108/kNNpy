# kNNpy

**kNNpy** is a Python package for computing the **k-Nearest Neighbor Cumulative Distribution Function (kNN-CDF)** — a powerful statistic designed to capture the full non-Gaussian information content of cosmological clustering. It provides a modular and efficient framework to analyze both projected (2D) and 3D large-scale structure data, going beyond traditional two-point statistics.

---

## Features

kNNpy provides the following functionalities:

- 📏 **Compute kNN-CDF statistics**:
  - `knn_3d` — for full 3D clustering analysis
  - `knn_2d_angular` — for angular (projected) clustering analysis
- 🧠 **Fisher Matrix Construction**:
  - Forecast cosmological parameter constraints
- 📈 **Peak Statistics**:
  - Analyze the high-density tail of the distribution
- 🔁 **Two-point Correlation Function (2PCF)**:
  - Standard pair-counting statistics for benchmarking
- 🧰 **Helper Submodules**:
  - Shared utilities for distance calculations, binning, and file handling
  - Designed to support both `knn_2d_angular` and `knn_3d` workflows

---

## Scientific Background

The `kNN-CDF` is defined as the empirical cumulative distribution of distances from volume-filling random points to their *k*-th nearest data point. It captures **all connected N-point functions** present in the data and is particularly sensitive to **non-Gaussian features** on small scales, making it a powerful alternative to traditional summary statistics like the correlation function or power spectrum.

The method was introduced in:

> **Banerjee & Abel (2020)**  
> *Nearest Neighbor distributions: new statistical measures for cosmological clustering*  
> [arXiv:2007.13342](https://arxiv.org/abs/2007.13342)

---

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/vikhyat108/kNNpy.git
cd kNNpy

