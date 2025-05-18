## Overview

This repository provides code for **Uncertainty Quantification in SPICE Reconstruction of Magnetic Resonance Spectroscopic Imaging (MRSI)**. It uses the Laplace approximation and Monte Carlo simulation to quantify the voxel-wise uncertainty and bias in metabolite concentration estimates derived from spectral fitting of SPICE-reconstructed MRSI data. 

## Project Structure

├── Basis_Fit_ESMRMB2025/ # Metabolite Basis functions for spectral fitting

├── Brain_img/ # Anatomical images and spatial priors

├── SPICE_Uncert_Ancillary.py # Helper functions

├── SPICE_Uncert_nbTest.ipynb # Notebook for running full pipeline

├── .gitignore # Ignore rules for datasets and outputs

├── requirements.txt # Environment requirement file

└── README.md # Project documentation (this file)

## Requirements

pip install -r requirements.txt

## Usage

1. Launch the notebook: SPICE_Uncert_nbTest.ipynb

2.Follow the notebook to: 

--- 1. Load basis and anatomical inputs

## Citation

If you use this code or find it helpful, please cite:

Tian Lyu, Simon M. Finney, Saad Jbabdi, William T. Clarke
**"Uncertainty Quantification in SPICE Reconstruction of MRSI"**,  in Proc. 41st Annu. Sci. Meeting ESMRMB, Marseille, France, Oct. 2025.

## License

© 2025 University of Oxford. All rights reserved.

This software is provided for academic and non-commercial use only.  
For licensing inquiries, please contact William Clarke @ william.clarke@ndcn.ox.ac.uk.
