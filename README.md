## Overview

This repository provides code for **Uncertainty Quantification in SPICE Reconstruction of Magnetic Resonance Spectroscopic Imaging (MRSI)**. It uses the Laplace approximation and Monte Carlo simulation to quantify the voxel-wise uncertainty and bias in metabolite concentration estimates derived from spectral fitting of SPICE-reconstructed MRSI data. 

## Project Structure

├── Basis_Fit_ESMRMB2025/ # Metabolite Basis functions for spectral fitting
├── Brain_img/ # Anatomical images and spatial priors
├── SPICE_Uncert_Ancillary.py # Helper functions
├── SPICE_Uncert_nbTest.ipynb # Notebook for running full pipeline
├── .gitignore # Ignore rules for datasets and outputs
└── README.md # Project documentation (this file)
