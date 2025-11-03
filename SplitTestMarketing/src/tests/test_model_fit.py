import unittest 
import pandas as pd
import numpy as np

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src import train

data_folder = os.path.join(project_root, "data")  # absolute path to data directory


class TestModelMetrics(unittest.TestCase):
    def test_train_file_load(self):
        df_train = pd.read_csv(data_folder+'/train.csv')
        self.assertTrue(isinstance(df_train, pd.DataFrame), "tran.csv is not a dataframe")

    def test_columns(self):
        df_train = pd.read_csv(data_folder+'/train.csv')
        df_train=df_train.iloc[:100] # for faster testing
        model=train.train_model(df_train)
        id_col = "userId"
        target_col = "clickedCTA"  # as specified in the case
        all_cols = df_train.columns.tolist()
        post_click_cols = ["submittedform","scheduledappointment","mortgagevariation","revenue"]
        feature_cols = [c for c in all_cols if c not in set([target_col]) | set(post_click_cols) | set([id_col])]

        X_test = df_train[feature_cols]
        y_predicted = model.predict_proba(X_test)
        # ensure the output is array-like and non-empty
        arr = np.asarray(y_predicted)
        self.assertIsInstance(arr, np.ndarray, "model.predict_proba did not return a numpy array-like object")
        self.assertGreater(arr.size, 0, "Predicted array is empty")
        # number of predictions should match number of samples
        self.assertGreaterEqual(arr.ndim, 1, "Prediction array has unexpected number of dimensions")
        self.assertEqual(arr.shape[0], X_test.shape[0], "Number of predictions does not match number of samples")
        # probabilities should be within [0, 1]
        self.assertTrue(np.all(arr >= 0.0) and np.all(arr <= 1.0), "Predicted probabilities are outside [0, 1]")
        # if predictions are probabilistic per class, rows should sum to ~1
        if arr.ndim == 2 and arr.shape[1] > 1:
            row_sums = arr.sum(axis=1)
            # allow small numerical tolerance
            self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-6), f"Probability rows do not sum to 1: min={row_sums.min()}, max={row_sums.max()}")
