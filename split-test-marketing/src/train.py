
import argparse
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def build_parser():
    p = argparse.ArgumentParser(description="Train a simple baseline model with preprocessing.")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--out_model", required=True)
    p.add_argument("--target", required=True, help="Name of the binary target column (0/1).")
    p.add_argument("--id_col", default=None, help="ID column to exclude from features (e.g., userId).")
    return p

def main():
    args = build_parser().parse_args()
    df = pd.read_csv(args.train_csv)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {args.train_csv}")
    y = df[args.target]
    drop_cols = [args.target]
    if args.id_col and args.id_col in df.columns:
        drop_cols.append(args.id_col)
    X = df.drop(columns=drop_cols, errors="ignore")

    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False))]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    clf = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))])

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_va)[:, 1]
    print("Validation log loss:", log_loss(y_va, p))

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.out_model)
    print("Saved model to", args.out_model)

if __name__ == "__main__":
    main()
