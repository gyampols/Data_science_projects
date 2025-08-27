
import argparse
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def build_parser():
    p = argparse.ArgumentParser(description="Evaluate predictions against truth.")
    p.add_argument("--preds_csv", required=True, help="CSV with [id_col, pr_CTA].")
    p.add_argument("--truth_csv", required=True, help="CSV with [id_col, target].")
    p.add_argument("--id_col", required=True, help="ID column to merge on.")
    p.add_argument("--target", required=True, help="Target column name.")
    return p

def main():
    args = build_parser().parse_args()
    preds = pd.read_csv(args.preds_csv)
    truth = pd.read_csv(args.truth_csv)
    df = truth.merge(preds, on=args.id_col, how="inner")
    if "pr_CTA" not in df.columns:
        raise ValueError("Expected column 'pr_CTA' in preds.")
    if args.target not in df.columns:
        raise ValueError(f"Expected target column '{args.target}' in truth.")
    y = df[args.target]
    p = df["pr_CTA"]
    print("Log loss:", log_loss(y, p))
    print("ROC-AUC:", roc_auc_score(y, p))
    print("PR-AUC:", average_precision_score(y, p))

if __name__ == "__main__":
    main()
