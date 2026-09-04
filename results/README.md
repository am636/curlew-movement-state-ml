# Results

These CSV files were generated from the public Zenodo and WorldClim data with Python 3.12. The downloaded data and larger generated files are not stored in the repository.

After hourly thinning, QA, and the minimum-track-length rule, the modelling table contained 61,903 rows from four birds; 18.1% of rows were labelled as active movement. The complete pipeline ran without error.

`model_metrics.csv` reports the mean of the four held-out-bird results, giving each bird equal weight. `per_bird_metrics.csv` contains the corresponding results for each bird.

`movement_threshold_sensitivity.csv` shows how the active fraction changes under alternative fixed speed thresholds. The analysis uses 0.5 km/h.

`random_forest_permutation_importance.csv` contains feature importance calculated on the held-out bird in each fold. Given the sample of four birds and the correlated spatial predictors, these results are descriptive rather than population-level estimates.
