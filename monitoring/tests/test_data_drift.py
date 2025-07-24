# Tests for data drift detection

# Tests for data drift detection
import unittest
import pandas as pd
import numpy as np
from monitoring.data_drift import run_drift_tests

class TestRunDriftTests(unittest.TestCase):
    def setUp(self):
        # Create simple reference and current dataframes
        self.reference_data = pd.DataFrame({
            'num_col': np.random.normal(0, 1, 1000),
            'cat_col': np.random.choice(['A', 'B', 'C'], 1000, p=[0.7, 0.2, 0.1])
        })
        # Current data with drift in both columns
        self.current_data = pd.DataFrame({
            'num_col': np.random.normal(1, 1, 1000),  # mean shifted
            'cat_col': np.random.choice(['A', 'B', 'C'], 1000, p=[0.4, 0.4, 0.2])
        })

    def test_run_drift_tests_returns_expected_keys(self):
        result = run_drift_tests(self.reference_data, self.current_data, html_report=False)
        self.assertIn('is_drift', result)
        self.assertIn('details', result)
        self.assertIn('data_summary', result)
        self.assertIn('reference_meta', result)
        self.assertIn('current_meta', result)

    def test_drift_detected(self):
        result = run_drift_tests(self.reference_data, self.current_data, html_report=False)
        # At least one column should have drift
        drift_columns = [col for col, det in result['details'].items() if det['is_drift']]
        self.assertTrue(len(drift_columns) > 0, "Drift should be detected in at least one column.")

    def test_no_drift_when_data_identical(self):
        result = run_drift_tests(self.reference_data, self.reference_data.copy(), html_report=False)
        drift_columns = [col for col, det in result['details'].items() if det['is_drift']]
        self.assertEqual(len(drift_columns), 0, "No drift should be detected when data is identical.")

if __name__ == "__main__":
    unittest.main()
