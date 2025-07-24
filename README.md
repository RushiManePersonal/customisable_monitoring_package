# Monitoring Package Documentation

## Overview
The `monitoring` package provides modular, extensible tools for data drift detection across tabular datasets. It supports multiple statistical tests, per-column configuration, weighted scoring, and generates structured output and interactive HTML reports.

---

## Installation

```bash
pip install .
```
Or add to your requirements and install via pip.

---

## Usage Example

```python
import monitoring.data_drift as data_drift
from monitoring.data_drift import run_drift_tests
from monitoring.drift_report import generate_html_report
import pandas as pd

current = pd.read_csv('trial/df_2016.csv')
previous = pd.read_csv('trial/df_2017.csv')

result = run_drift_tests(previous, current)

# Print results
import json
print(json.dumps(result, indent=2, default=str))

# Save results
with open('drift_results.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)

# Generate HTML report
generate_html_report(result, output_path='drift_report.html')
```

---

## Main Modules

### `monitoring.data_drift`
- **`run_drift_tests(reference_data, current_data, columns_to_test=None, columns_to_exclude=None, test_config=None)`**
  - Runs drift tests for specified columns.
  - Smart test selection based on column type.
  - Supports per-column test config, thresholds, and weights.
  - Returns structured results for each column and test.

### `monitoring.stat_tests`
- Contains individual test modules:
  - `ks_stat_test.py` (Kolmogorov-Smirnov)
  - `psi_test.py` (Population Stability Index)
  - `chi2_test.py` (Chi-Square)
  - `wasserstein_test.py` (Wasserstein Distance)
  - `anderson_darling_test.py` (Anderson-Darling)
  - `kl_divergence_test.py` (KL Divergence)
  - `ttest.py` (T-Test)
  - `mannwhitneyu_test.py` (Mann-Whitney U)
  - `levene_test.py` (Levene's Test)
  - `f_test.py` (F-Test)
  - `jensen_shannon_test.py` (Jensen-Shannon)
  - `mmd_test.py` (Maximum Mean Discrepancy)
  - `emd_test.py` (Earth Mover's Distance)
- Each test exposes a `run` method for use in drift detection.

### `monitoring.drift_report`
- **`generate_html_report(result, output_path='drift_report.html')`**
  - Generates an interactive HTML report using Jinja2.
  - Supports expandable details and column plots.

### `monitoring/drift_report_template.html`
- Jinja2 template for HTML report.
- Interactive, expandable details for test outputs.
- Plots per column (if provided).

### `monitoring/drift_report_assets.js`
- JavaScript for expand/collapse details in the report.

---

## Customization
- **Columns:** Use `columns_to_test` or `columns_to_exclude` in `run_drift_tests`.
- **Test config:** Pass `test_config` dict for per-column tests, thresholds, and weights.
- **Plots:** Add paths to images in `result['plots'][col]` for each column before report generation.

---

## Output
- **Structured dict:** Per-column, per-test results, drift status, statistics, thresholds, weights.
- **HTML report:** Interactive, expandable, with optional plots.
- **JSON file:** For programmatic use or archiving.

---

## Extending
- Add new statistical tests in `monitoring/stat_tests` and update selection logic in `data_drift.py`.
- Customize HTML report via `drift_report_template.html` and assets.

---

## Requirements
- `numpy`, `scipy`, `pandas`, `jinja2`, `setuptools`

---

## License
MIT

---

## Support
For issues or feature requests, open an issue in your project repository.
