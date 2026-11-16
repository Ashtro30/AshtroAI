# Machine Learning Evaluation Best Practices

Internal guidance for the Meridian Labs data team.

## Data Splitting

Use a **70/15/15** split for train, validation and test sets unless the dataset is very small, in which case prefer k-fold cross-validation (k = 5). For time-series data always split chronologically and never shuffle across the time boundary.

## Preventing Data Leakage

- Fit scalers, encoders and imputers on the training split only.
- Deduplicate before splitting so near-identical samples never straddle the split.
- Keep the test set untouched until the final evaluation; report it once.

## Metrics

- Classification: report precision, recall, F1 and a confusion matrix, not accuracy alone, whenever classes are imbalanced.
- Object detection: report mAP@0.5 and mAP@0.5:0.95.
- Regression: report MAE and RMSE, plus a residual plot.
- Always include a trivial baseline (majority class or mean predictor) for comparison.

## Reproducibility

Pin random seeds, record library versions, and store every experiment's configuration next to its results. Models promoted to production require a model card describing intended use, training data, metrics and known limitations.
