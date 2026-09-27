'''
> A modular Python toolkit for computing ***k*-nearest neighbour (kNN) distributions** — advanced clustering statistics that go beyond the traditional two-point correlation function.

---

## Overview

`kNNpy` is a high-performance package for computing ***k*-nearest neighbour cumulative distribution functions (kNN CDFs)** — powerful summary statistics that are sensitive to **all connected *N*-point functions** of the matter density field.

Unlike the standard two-point correlation function, kNN statistics capture **non-Gaussian information**, making them a valuable tool for cosmological inference using modern survey data.

For theory, examples, and the science behind the code, visit the official website:  
**[kitnenikatnivasi.github.io](https://kitnenikatnivasi.github.io)**

---

## Submodules

For detailed API documentation, explore the submodules below:

- **[Auxiliary](https://kitnenikatnivasi.github.io/kNNpy_documentation_html/kNNpy/Auxiliary.html)**: Utility functions for Fisher Information, Peak Statistics, and Two-Point Correlation Functions.
- **[HelperFunctions](https://kitnenikatnivasi.github.io/kNNpy_documentation_html/kNNpy/HelperFunctions.html)**: Helper routines for calculating excess correlations and query grid generation.
- **[HelperFunctions_2DA](https://kitnenikatnivasi.github.io/kNNpy_documentation_html/kNNpy/HelperFunctions_2DA.html)**: Helper functions for 2D angular statistics.
- **[kNN_2D_Ang](https://kitnenikatnivasi.github.io/kNNpy_documentation_html/kNNpy/kNN_2D_Ang.html)**: 2D Angular kNN CDF functions.
- **[kNN_3D](https://kitnenikatnivasi.github.io/kNNpy_documentation_html/kNNpy/kNN_3D.html)**: 3D spatial kNN CDF functions.

---

## Tutorials

Hands-on Jupyter notebook tutorials demonstrating practical applications of `kNNpy` are available in the repository (`Tutorials/` folder) and on the website:

### 3D Spatial kNN Statistics
- **[3D Auto-Clustering](https://github.com/vikhyat108/kNNpy/blob/main/Tutorials/kNN_3D_tutorial_1_auto_clustering.ipynb)**: Computing auto-kNN cumulative distribution functions (CDFs) in 3D box simulations.
- **[3D Tracer-Field Cross-Clustering](https://github.com/vikhyat108/kNNpy/blob/main/Tutorials/kNN_3D_tutorial_2_tracer_field_cross_clustering.ipynb)**: Measuring cross-correlations between discrete point tracers and a continuous matter density field.
- **[3D Tracer-Tracer Cross-Clustering](https://github.com/vikhyat108/kNNpy/blob/main/Tutorials/kNN_3D_tutorial_3_tracer_tracer_cross_clustering.ipynb)**: Spatial cross-clustering between two discrete tracer populations (e.g. Galaxies and Clusters).

### 2D Angular kNN Statistics
- **[2D Angular Auto-Clustering](https://github.com/vikhyat108/kNNpy/blob/main/Tutorials/kNN_2D_Ang_tutorial_1_auto_clustering.ipynb)**: Angular kNN-CDFs on 2D projected sky maps.
- **[2D Angular Tracer-Field Cross-Clustering](https://github.com/vikhyat108/kNNpy/blob/main/Tutorials/kNN_2D_Ang_tutorial_2_tracer_field_cross_clustering.ipynb)**: Cross-correlating 2D point catalogs with projected density fields.
- **[2D Angular Tracer-Tracer Cross-Clustering](https://github.com/vikhyat108/kNNpy/blob/main/Tutorials/kNN_2D_Ang_tutorial_3_tracer_tracer_cross_clustering.ipynb)**: Cross-clustering between two 2D point populations.

### Cosmology & Auxiliary Tools
- **[Fisher Information Matrix](https://github.com/vikhyat108/kNNpy/blob/main/Tutorials/Fisher_tutorial.ipynb)**: Cosmological parameter constraint forecasts.
- **[Peak Statistics](https://github.com/vikhyat108/kNNpy/blob/main/Tutorials/PeakStatistics_tutorial.ipynb)**: Density field peak counts and map statistics.
- **[Two-Point Correlation Function (2PCF)](https://github.com/vikhyat108/kNNpy/blob/main/Tutorials/Tracer_field_2PCF_tutorial.ipynb)**: Standard 2PCF baseline comparisons.

---

## Getting Started & Website

For installation instructions, comprehensive theory, and interactive guides, please visit the main website:  
**[kNNpy Documentation & Website](https://kitnenikatnivasi.github.io/index.html)**
'''