## Overview
This repository contains code associated with the following paper:

*F. H. Athina, R. Iqbal, Y. Zhang and T. Jerin, "RiWiX: Toward Automatic River Water Width Extraction From High-Resolution Satellite Imagery Using Swin Transformers," 
in IEEE Access, vol. 14, pp. 24139-24157, 2026, https://doi.org/10.1109/ACCESS.2026.3663480*. 

The paper introduces the **RiWiX (River Water Width Extraction)** framework that:
1) segments river-water surfaces from very high-resolution imagery using a Swin-Transformer-based segmentation model, and  
2) derives continuous river widths using a graph-based centerline extraction and perpendicular width measurement procedure.

* * *

## Project Structure
- datsets.py      #Implements the GLHDataset class and build_dataset(...).
Expects a dataset directory under root_path structured as:

root_path/
  train/
    img/
    label/
  val/
    img/
    label/


* * *

## Installation

Create a new conda envirionment and install the following libaries listed in requiremets.txt file.

- conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
- conda install conda-forge::numpu==1.26.4
- conda install conda-forge::einops
- conda install conda-forge::timm
- conda install conda-forge::matplotlib
- conda install conda-forge::wandb

* * *

## Citation
If you use this code or build upon this work, please cite:

```bibtex
@article{Athina2026RiWiX,
  author  = {Athina, Fahima Hasan and Iqbal, Razib and Zhang, Yifan and Jerin, Tasnuba},
  title   = {RiWiX: Toward Automatic River Water Width Extraction From High-Resolution Satellite Imagery Using Swin Transformers},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {24139-24157},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3663480}
}
