# Curlew movement-state classification with Python

This is a small machine-learning case study using open GPS tracking data from Eurasian curlews (*Numenius arquata*) tagged in Flanders, Belgium.

The aim is simple: use spatial, temporal and broad environmental context to classify whether a GPS step represents **active movement** or **low movement**, then compare three models:

- logistic regression;
- Random Forest;
- a small PyTorch neural network.

The movement label is derived from step speed and is used only as an operational label for this exercise. It is not intended as a validated behavioural-state classification.

## Data

Tracking data come from the open **CURLEW_VLAANDEREN** dataset published by the Research Institute for Nature and Forest (INBO). The fixed Zenodo release used here contains GPS files for 2020-2024 and reference data for five tagged Eurasian curlews.

**Tracking dataset**  
Spanoghe, G., Janssens, K., Nijs, G., Govaert, S., Milotic, T., & Desmet, P. (2025). *CURLEW_VLAANDEREN - Eurasian curlews (Numenius arquata, Scolopacidae) breeding in Flanders (Belgium)*. Zenodo. https://doi.org/10.5281/zenodo.15696532

The dataset is released under CC0 1.0.

The environmental variables come from **WorldClim 2.1**. The project uses annual mean temperature (BIO1), annual precipitation (BIO12), and elevation at 10-minute resolution.

The data are downloaded automatically. No Movebank account or API key is needed.

## Run the project

Create a Python environment and install the packages:

```bash
pip install -r requirements.txt
```

Run the complete project with:

```bash
python 01_download_data.py
python 02_prepare_features.py
python 03_train_compare_models.py
python 04_map_and_diagnostics.py
```

The download is modest (roughly 70 MB in total). Downloaded files are cached locally, so existing files are not downloaded again.

## What the scripts do

### `01_download_data.py`

Downloads the curlew GPS files from Zenodo and the WorldClim climate/elevation archives, then extracts the WorldClim rasters.

### `02_prepare_features.py`

Reads and orders GPS fixes by bird, keeps the first fix in each hour, calculates step distance and elapsed time, removes very short/long intervals and implausible step speeds, then creates the movement label.

A step is labelled active when its calculated speed is at least **5 km/h**. The script also writes a small sensitivity table for thresholds of 2, 5 and 10 km/h.

The predictors are deliberately limited to variables that do not directly contain the step speed used to define the target:

- longitude and latitude;
- time of day represented as sine/cosine terms;
- day of year represented as sine/cosine terms;
- WorldClim BIO1;
- WorldClim BIO12;
- elevation.

### `03_train_compare_models.py`

Compares logistic regression, Random Forest and a small multilayer perceptron implemented in PyTorch.

Validation is grouped by bird using leave-one-bird-out cross-validation. GPS fixes from the held-out bird are therefore never used to fit that fold's model.

The script saves out-of-fold predictions and reports balanced accuracy, ROC-AUC, PR-AUC, precision, recall, F1 and Brier score. It also saves mean Random Forest feature importance across the grouped folds. After cross-validation it fits one final PyTorch model to the complete dataset and saves its `state_dict` together with the feature names and scaling values. The saved all-data model is not used to report performance.

### `04_map_and_diagnostics.py`

Creates a small set of figures showing:

- the GPS locations and movement labels;
- overall model performance;
- probability calibration;
- confusion matrices;
- F1 score for each held-out bird;
- mean Random Forest feature importance across the grouped folds.

## Notes on interpretation

This is a deliberately small example rather than a study of curlew behaviour.

The main limitations are:

- only five tagged individuals are represented;
- the target is a simple speed-threshold label rather than an independently observed behaviour;
- WorldClim variables describe long-term climate rather than weather at each GPS timestamp;
- location itself can be informative, so good predictive performance should not be interpreted as evidence of a causal environmental relationship.

The grouped validation and calibration checks are included because repeated observations from the same animal can otherwise give an overly optimistic view of model performance.

## Repository structure

```text
curlew-movement-state-ml/
├── README.md
├── requirements.txt
├── 01_download_data.py
├── 02_prepare_features.py
├── 03_train_compare_models.py
├── 04_map_and_diagnostics.py
├── notebooks/
│   └── 01_explore_tracks.ipynb
├── data/
│   └── README.md
└── outputs/
    ├── figures/
    ├── tables/
    └── models/
```

## Author

Ali Moayedi  
University of St Andrews
