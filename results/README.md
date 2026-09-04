# Results

These small text outputs are included so the repository shows the result of a real end-to-end run without storing downloaded data or large generated files.

The final pipeline was tested on a clean GitHub-hosted Python 3.11 environment using the public Zenodo tracking files and WorldClim downloads.

After hourly thinning, QA and the minimum-track-length rule, the modelling table contained 61,903 rows from four birds. The full pipeline completed successfully with no Python warnings or errors.

`model_metrics.csv` contains leave-one-bird-out model performance.

`movement_threshold_sensitivity.csv` shows how the proportion labelled active changes under alternative fixed speed thresholds. The final analysis uses 0.5 km/h.
