# EELSNMF: NMF Analysis tailored to EELS Data

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**EELSNMF** is a Python library for the advanced decomposition of Electron Energy Loss Spectroscopy (EELS) datasets using Non-negative Matrix Factorization (NMF). 

The library implements and extends the mathematical logic described in:
> **Adrien Teurtrie** *et al*, "Non-negative matrix factorization for spectroscopic data analysis", *Mach. Learn.: Sci. Technol.* **5** 045050 (2024). [DOI: 10.1088/2632-2153/ad9192](https://iopscience.iop.org/article/10.1088/2632-2153/ad9192).

While the original paper focuses on Energy-dispersive X-ray spectroscopy (EDX), this library adapts that logic specifically for **EELS**. It provides custom models for the $G$ matrix (including cross-section models) and various specialized decomposition methods

## Installation

### From GitHub
While the package is in development, you can install it directly from the source:

```bash
pip install git+[https://github.com/PauTorru/EELSNMF.git](https://github.com/PauTorru/EELSNMF.git)
```
### Installation with Optional Capabilities
Specific experimental functionalities are available as "extras". You can install them by appending the flags in brackets:

* **GPU Support:** 
```bash
pip install "EELSNMF[gpu] @ git+https://github.com/PauTorru/EELSNMF.git"
```
* **EML data processing:** 
```bash
pip install "EELSNMF[EML] @ git+https://github.com/PauTorru/EELSNMF.git"
```
* **Torch Solvers:** 
```bash
install "EELSNMF[torch] @ git+https://github.com/PauTorru/EELSNMF.git"
```
* **Full Suite:** 
```bash
pip install "EELSNMF[all] @ git+https://github.com/PauTorru/EELSNMF.git"
```

*Note: If you are using ZSH (default on macOS), ensure you use quotes around the package name to avoid shell errors.*

## Getting Started
To begin, please refer to the Jupyter Notebook `demo.ipynb`. It provides a step-by-step guide the usage of the library.

## Research & Attribution
This software is provided under a **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. 

If you use this code in your research, please acknowledge the original work.

### Publications using EELSNMF
* *-*

