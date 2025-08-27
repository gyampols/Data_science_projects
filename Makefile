
.PHONY: setup train predict evaluate

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	cd "split-test-marketing" && python -m src.train --train_csv data/raw/train.csv --out_model models/model.joblib --target target --id_col userId

predict:
	cd "split-test-marketing" && python -m src.predict --model models/model.joblib --features_csv data/raw/test.csv --out_csv outputs/predictions/predictions.csv --id_col userId

evaluate:
	cd "split-test-marketing" && python -m src.evaluate --preds_csv outputs/predictions/predictions.csv --truth_csv data/raw/train.csv --id_col userId --target target
