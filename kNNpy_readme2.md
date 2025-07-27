---

## 📦 Features

**kNNpy** provides the following functionalities:

- **Compute kNN-CDF statistics**:
  - **$k$NN_3D** — for 3D clustering analysis when 3D positions and/or the 3D field are available:
    - Tracer auto
    - Tracer cross tracer
    - Tracer cross field
  - **$k$NN_2D** — for angular clustering analysis when (RA, Dec) and/or a 2D field on the sky are available:
    - Tracer auto
    - Tracer cross tracer
    - Tracer cross field

- **Peak Statistics**:
  - Analyze the high-density tail of the distribution

- **Helper Submodules**:
  - Shared utilities for distance calculations, binning, and file handling
  - Designed to support both **$k$NN_3D** and **$k$NN_2D** workflows

- **Two-point Correlation Function (2PCF)**:
  - Standard pair-counting statistics for benchmarking

- **Fisher Matrix Construction**:
  - Forecast cosmological parameter constraints

