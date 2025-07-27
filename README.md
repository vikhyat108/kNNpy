# <img src="logo.jpeg" alt="kNNpy Logo" width="30"/> **kNNpy**

**kNNpy** is a Python package for computing the **k-Nearest Neighbor Cumulative Distribution Function (kNN-CDF)** — a powerful statistic designed to capture the full non-Gaussian information content of cosmological clustering. It provides a modular and efficient framework to analyze both 2D and 3D large-scale structure data, going beyond traditional two-point statistics.

---

## 📦 Features

kNNpy provides the following functionalities:

- **Compute kNN distributions**:
  - `kNN_3D` — for 3D clustering analysis, when we have the 3-D positions and/or the 3-D field    
    - Tracer auto  
    - Tracer cross tracer  
    - Tracer cross field
  - `kNN_2D` — for angular clustering analysis, when we have the (RA, Dec) and/or the 2-D field on the sky   
    - Tracer auto  
    - Tracer cross tracer  
    - Tracer cross field
- **Peak Statistics**:
  - Analyze the high-density tail of the distribution  
- **Helper Submodules**:
  - Shared utilities for distance calculations, binning, and file handling  
  - Designed to support both `kNN_3D` and `kNN_2D` workflows
- **Two-point Correlation Function (2PCF)**:
  - Standard pair-counting statistics for benchmarking  
- **Fisher Matrix Construction**:
  - Forecast cosmological parameter constraints  

---

## 📚 Dependencies

`kNNpy` makes use of the following packages:

- `numpy`
- `scipy`
- `pyfftw`
- `pylians`
- `healpy` *(optional)*
- `scikit-learn` *(optional)*
- `matplotlib` *(optional)*

These need to be installed before using the package, preferably in a fresh virtual environment (see instructions below).  
`healpy` and `scikit-learn` are optional, and can be skipped if you do not intend to use the `kNNpy.kNN_2D_Ang` module.  
Similarly, `matplotlib` is optional if you do not intend to use the `kNNpy.Auxiliary.PeakStatistics` module.

---

## 🛠 Installation

### 🐧 Linux/macOS

Change to your preferred installation directory, create a Python virtual environment and install the dependencies.

**⚠️ Warning**: Do **not** use `anaconda` or `miniconda` for the virtual environment as it may cause conflicts with `Pylians`. Also, do **not** name the virtual environment `kNNpy` to avoid namespace issues.

```bash
cd /path/to/installation/directory
python3 -m venv kNNpy_env
source kNNpy_env/bin/activate
pip install numpy scipy pyfftw Pylians healpy scikit-learn matplotlib
```

> 💡 **Note on Pylians installation**  
> Sometimes, the Pylians installation fails when using `pip` (especially on macOS). If that happens:
>
> - First install all other dependencies (without Pylians)
> - Then install Pylians in development mode following their official instructions:  
> 👉 [https://github.com/franciscovillaescusa/Pylians](https://github.com/franciscovillaescusa/Pylians)

If you **do not** want the optional dependencies, run instead:

```bash
pip install numpy scipy pyfftw Pylians
```

### 📥 Clone this repository

```bash
git clone https://github.com/vikhyat108/kNNpy.git
```

### 🧭 Set your `PYTHONPATH`

Export the path to the installed repository to your Python path:

- **Temporarily** (for current shell session):

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/installation/directory/kNNpy/"
```

- **Permanently** (recommended): add the line above to your `~/.bashrc` or `~/.zshrc`.

---

### 🪟 Windows

Unfortunately, `kNNpy` is currently not supported on Windows. Future support may be added in later releases.

---

## ▶️ Usage

Change to your working directory:

```bash
cd /my/working/directory
```

Activate your virtual environment:

```bash
source /path/to/virtual/environment/bin/activate
```

Set the `PYTHONPATH` (if not already done) and then in your Python scripts or notebooks, you can now:

```python
import kNNpy
```

---

## 🔬 Scientific Background

The `kNN CDF` is defined as the empirical cumulative distribution of distances from volume-filling random points to their *k*-th nearest data point. It captures **all connected N-point functions** present in the data and is particularly sensitive to **non-Gaussian features** on small scales, making it a powerful alternative to traditional summary statistics like the correlation function or power spectrum.

This methodology was introduced in:

> **Banerjee & Abel (2020)**  
> *Nearest Neighbor distributions: new statistical measures for cosmological clustering*  
> 📄 [arXiv:2007.13342](https://arxiv.org/abs/2007.13342)

---

## 📘 Documentation

The most updated documentation with examples can be found [**here**](https://kitnenikatnivasi.github.io/kNNpy_documentation_html/kNNpy.html).

---

## 📬 Contact

For comments, questions or bug reports, feel free to reach out:  
📧 **kitnenikatnivasi@gmail.com**

---

## 🌐 Website

🌍 You can find more about the codes and the team [**here**](https://kitnenikatnivasi.github.io)

