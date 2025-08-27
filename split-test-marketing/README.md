
# split-test-marketing

This project predicts and evaluates marketing split tests (copy × placement) and produces calibrated probability predictions for the target action.

## CLI

```bash
# Train
cd "split-test-marketing"
python -m src.train --train_csv data/raw/train.csv --out_model models/model.joblib --target target --id_col userId

# Predict
python -m src.predict --model models/model.joblib --features_csv data/raw/test.csv --out_csv outputs/predictions/predictions.csv --id_col userId

# Evaluate
python -m src.evaluate --preds_csv outputs/predictions/predictions.csv --truth_csv data/raw/train.csv --id_col userId --target target
```
