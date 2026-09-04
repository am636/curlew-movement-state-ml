# Curlew Movement-State Classification

*Code and workflow for movement-state classification in GPS-tracked Eurasian curlews*

---

## Overview

This repository classifies active and low-movement steps in public GPS tracks from Eurasian curlews (*Numenius arquata*) using Python. GPS fixes are thinned to one per UTC hour and used to derive spatiotemporal predictors, while WorldClim climate and elevation rasters are sampled at the track locations. Logistic regression, Random Forest, and a PyTorch multilayer perceptron are compared using leave-one-bird-out cross-validation.

The source data contain five curlews tagged in Flanders, Belgium, between 2020 and 2024. Tracks with fewer than 500 valid hourly steps are excluded, which removes one 12-day record. The movement label is defined by a 0.5 km/h step-speed threshold and is an operational classification rather than a validated behavioural state. With four birds in the model evaluation, the performance estimates and feature rankings are descriptive.

## Workflow

The scripts are organised to be run in sequence:

1. **Download data** (`01_download_data.py`)  
   Downloads the CURLEW_VLAANDEREN GPS files from Zenodo and WorldClim bioclimatic and elevation rasters.

2. **Prepare movement and environmental features** (`02_prepare_features.py`)  
   Cleans and orders GPS fixes, keeps the first fix in each UTC hour, calculates step movement, creates temporal variables, extracts WorldClim values, and prepares the modelling table.

3. **Train and compare models** (`03_train_compare_models.py`)  
   Fits logistic regression, Random Forest, and a PyTorch multilayer perceptron. Model evaluation uses leave-one-bird-out cross-validation so observations from the held-out bird are not used for training. Random Forest importance is calculated by permuting features for each held-out bird.

4. **Create diagnostic outputs** (`04_map_and_diagnostics.py`)  
   Produces model-comparison plots, calibration curves, confusion matrices, per-bird performance summaries, feature importance, and a map of the movement labels.

## Data availability

- **Eurasian curlew GPS data:** CURLEW_VLAANDEREN, available from Zenodo: https://doi.org/10.5281/zenodo.15696532
- **Bioclimatic variables and elevation:** WorldClim 2.1: https://www.worldclim.org/data/worldclim21.html

The required files are downloaded automatically by `01_download_data.py`; no Movebank account or API key is required.

The Zenodo data are released under CC0 1.0. WorldClim permits academic and other non-commercial use but does not permit redistribution or commercial use without prior permission. The downloaded data are not stored in this repository; the MIT licence applies to the code.

## Requirements

- **Python libraries:** `numpy`, `pandas`, `scikit-learn`, `torch`, `rasterio`, `matplotlib`

The workflow has been tested with Python 3.12.

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

## Data citations

Spanoghe, G., Janssens, K., Nijs, G., Govaert, S., Milotic, T., & Desmet, P. (2025). *CURLEW_VLAANDEREN - Eurasian curlews (Numenius arquata, Scolopacidae) breeding in Flanders (Belgium)*. Zenodo. https://doi.org/10.5281/zenodo.15696532

Fick, S. E., & Hijmans, R. J. (2017). WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas. *International Journal of Climatology*, 37(12), 4302-4315.

## Contact

Ali Moayedi  
University of St Andrews, UK  
am636@st-andrews.ac.uk
