# Curlew Movement-State Classification

*Python workflow for animal-tracking, spatial environmental data integration, and supervised machine-learning comparison*

---

## Overview

This repository contains a Python workflow for classifying active and low-movement steps in GPS tracks from Eurasian curlews (*Numenius arquata*). The workflow combines GPS preprocessing, hourly track construction, spatiotemporal feature extraction, environmental raster sampling from WorldClim, grouped cross-validation, and comparison of logistic regression, Random Forest, and a PyTorch neural network.

The analysis uses open tracking data from five curlews tagged in Flanders, Belgium, between 2020 and 2024. One bird is excluded from model evaluation because it has too few valid hourly observations after preprocessing. The movement label is based on step speed and is used here as a simple analytical classification rather than a validated behavioural state.

## Workflow

The scripts are organised to be run in sequence:

1. **Download data** (`01_download_data.py`)  
   Downloads the CURLEW_VLAANDEREN GPS files from Zenodo and WorldClim bioclimatic and elevation rasters.

2. **Prepare movement and environmental features** (`02_prepare_features.py`)  
   Cleans and orders GPS fixes, keeps one fix per hour, calculates step movement, creates temporal variables, extracts WorldClim values, and prepares the modelling table.

3. **Train and compare models** (`03_train_compare_models.py`)  
   Fits logistic regression, Random Forest, and a PyTorch multilayer perceptron. Model evaluation uses leave-one-bird-out grouped cross-validation so observations from the held-out bird are not used for training.

4. **Create diagnostic outputs** (`04_map_and_diagnostics.py`)  
   Produces model-comparison plots, calibration curves, confusion matrices, per-bird performance summaries, feature importance, and a map of the movement labels.

## Data availability

- **Eurasian curlew GPS data:** CURLEW_VLAANDEREN, available from Zenodo: https://doi.org/10.5281/zenodo.15696532
- **Bioclimatic variables and elevation:** WorldClim 2.1: https://www.worldclim.org/data/worldclim21.html

The required files are downloaded automatically by `01_download_data.py`; no Movebank account or API key is required.

## Requirements

- **Python libraries:** `numpy`, `pandas`, `scikit-learn`, `torch`, `rasterio`, `matplotlib`

Install the required packages with:

```bash
pip install -r requirements.txt
```

Run the workflow with:

```bash
python 01_download_data.py
python 02_prepare_features.py
python 03_train_compare_models.py
python 04_map_and_diagnostics.py
```

## Data citation

Spanoghe, G., Janssens, K., Nijs, G., Govaert, S., Milotic, T., & Desmet, P. (2025). *CURLEW_VLAANDEREN - Eurasian curlews (Numenius arquata, Scolopacidae) breeding in Flanders (Belgium)*. Zenodo. https://doi.org/10.5281/zenodo.15696532

## Contact

Ali Moayedi  
University of St Andrews, UK  
am636@st-andrews.ac.uk
