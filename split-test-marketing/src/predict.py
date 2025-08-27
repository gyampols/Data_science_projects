
import argparse
from pathlib import Path
import pandas as pd
import joblib

def build_parser():
    p = argparse.ArgumentParser(description="Generate probability predictions using a trained model.")
    p.add_argument("--model", required=True, help="Path to trained joblib model.")
    p.add_argument("--features_csv", required=True, help="CSV of features (must include id_col).")
    p.add_argument("--out_csv", required=True, help="Output CSV with [id_col, pr_CTA].")
    p.add_argument("--id_col", required=True, help="Name of the ID column to preserve (e.g., userId).")
    return p

def main():
    args = build_parser().parse_args()
    model = joblib.load(args.model)
    df = pd.read_csv(args.features_csv)
    if args.id_col not in df.columns:
        raise ValueError(f"id_col '{args.id_col}' not in {args.features_csv}")
    ids = df[args.id_col]
    X = df.drop(columns=[args.id_col], errors="ignore")
    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({args.id_col: ids, "pr_CTA": proba})
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("Wrote predictions to", args.out_csv)

if __name__ == "__main__":
    main()
