import unittest 
import pandas as pd
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
data_folder = "SplitTestMarketing/data"  # relative path from project root


class TestModelMetrics(unittest.TestCase):
    def test_train_file_load(self):
        df_train = pd.read_csv(data_folder+'/train.csv')
        self.assertTrue(isinstance(df_train, pd.DataFrame), "tran.csv is not a dataframe")

    def test_test_file_load(self):
        df_test = pd.read_csv(data_folder+'/test.csv')
        self.assertTrue(isinstance(df_test, pd.DataFrame),"test.csv is not a dataframe")
    
    def test_columns(self):
        df = pd.read_csv(data_folder+'/train.csv')
        expected_cols = ["userId", "clickedCTA", "estimatedAnnualIncome", "visitCount", "scrollDepth",
                   "sessionReferrer", "browser", "deviceType", "estimatedPropertyType", "ctaCopy", 
                   "ctaPlacement", 'date', 'pageURL','editorialSnippet']
        missing_cols = set(expected_cols) - set(df.columns)
        self.assertEqual(missing_cols, set(), f"Missing columns in DataFrame: {missing_cols}")
