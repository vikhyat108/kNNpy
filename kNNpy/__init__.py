'''
<p align="center">
  <img src="https://github.com/vikhyat108/kNNpy/blob/main/logo2.png" alt="kNNpy Logo" width="240"/>
</p>

> A modular Python toolkit for computing **$k$-nearest neighbour (kNN) distributions** — advanced clustering statistics that go beyond the traditional two-point correlation function.

---

## Overview

`kNNpy` is a high-performance package for computing **$k$-nearest neighbour cumulative distribution functions (kNN CDFs)** — powerful summary statistics that are sensitive to **all connected $N$-point functions** of the matter density field.

Unlike the standard two-point correlation function, kNN statistics capture **non-Gaussian information**, making them a valuable tool for cosmological inference using modern survey data.

For theory, examples, and science behind the code, visit:  
**[kitnenikatnivasi.github.io](https://kitnenikatnivasi.github.io)**

---

## Submodules

For learning more about the submodules, click on the links below to be redirected a more comprehensive overview:
<ul>
    <li><a href="https://kitnenikatnivasi.github.io/kNNpy_documentation_html/kNNpy/Auxiliary.html">Auxiliary Functions</a>
    <ul>
        <li><a href="https://kitnenikatnivasi.github.io/kNNpy_documentation_html/kNNpy/Auxiliary/Fisher.html">Fisher Information</a></li>
        <li><a href="https://kitnenikatnivasi.github.io/kNNpy_documentation_html/kNNpy/Auxiliary/PeakStatistics.html">Peak Statistics</a></li>
        <li><a href="https://kitnenikatnivasi.github.io/kNNpy_documentation_html/kNNpy/Auxiliary/TPCF.html">Two Point Correlation Function)</a></li>
    </ul>
    </li>
<li><a href="https://kitnenikatnivasi.github.io/kNNpy_documentation_html/kNNpy/HelperFunctions.html">Helper Functions</a></li>
<li><a href="https://kitnenikatnivasi.github.io/kNNpy_documentation_html/kNNpy/HelperFunctions_2DA.html">Helper Functions for 2D Angular statistics</a></li>
<li><a href="https://kitnenikatnivasi.github.io/kNNpy_documentation_html/kNNpy/kNN_2D_Ang.html">2D Angular kNN CDFs</a></li>
<li><a href="https://kitnenikatnivasi.github.io/kNNpy_documentation_html/kNNpy/kNN_3D.html">3D kNN CDFs</li>
</ul>

---

## Installation

### Dependencies

`kNNpy` makes use of the following packages:

- `numpy`
- `scipy`
- `pyfftw`
- `Pylians`
- `healpy` *(optional)*
- `scikit-learn` *(optional)*
- `matplotlib` *(optional)*

These need to be installed before using the package, preferably in a fresh virtual environment (see instructions below).  
`scikit-learn` is optional, and can be skipped if you do not intend to use the `kNNpy.kNN_2D_Ang` module.  
Similarly, `healpy` and `matplotlib` are optional if you do not intend to use the `kNNpy.kNN_2D_Ang`, `kNNpy.Auxiliary.TPCF.TracerField2D` and `kNNpy.Auxiliary.PeakStatistics` modules.

---
### Linux/macOS

Change to your preferred installation directory, create a Python virtual environment and install the dependencies.

**Warning**: Do **not** use `anaconda` or `miniconda` for the virtual environment as it may cause conflicts with `Pylians`. Also, do **not** name the virtual environment `kNNpy` to avoid namespace issues.

```bash
cd /path/to/installation/directory
python3 -m venv kNNpy_env
source kNNpy_env/bin/activate
pip install numpy scipy pyfftw Pylians healpy scikit-learn matplotlib
```

> **Note on Pylians installation**  
> Sometimes, the Pylians installation fails when using `pip` (especially on macOS). If that happens:
>
> - First install all other dependencies (without Pylians)
> - Then install Pylians in development mode following their official instructions:  
> [Pylians_documentation](https://pylians3.readthedocs.io/en/master/installation.html#)

If you **do not** want the optional dependencies, replace the last line above by:

```bash
pip install numpy scipy pyfftw Pylians
```

### Clone this repository

```bash
git clone https://github.com/vikhyat108/kNNpy.git
```

### Set your `PYTHONPATH`

Export the path to the installed repository to your Python path:

- **Temporarily** (for current shell session):

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/installation/directory/kNNpy/"
```

- **Permanently** (recommended): add the line above to your `~/.bashrc` or `~/.zshrc`.

### Test the installation

To check if `kNNpy` is successfully installed and ready to use, run the following command:

```bash
cd /path/to/installation/directory/
cd kNNpy/Tests/
python3 import_all_modules.py
```

If you did not install the optional dependencies, replace the last line above by:

```bash
python3 import_required_modules.py
```
If no error message is returned, the installation is succesful.

---

### Windows

Unfortunately, `kNNpy` is currently not supported on Windows. Future support may be added in later releases.

---

## Tutorials

---

## Reference

---

## Licence and Credits



'''