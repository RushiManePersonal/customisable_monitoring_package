from monitoring.stat_tests.ks_stat_test import KSStatTest
from monitoring.stat_tests.psi_test import PSITest
from monitoring.stat_tests.chi2_test import Chi2Test
from monitoring.stat_tests.wasserstein_test import WassersteinTest
from monitoring.stat_tests.anderson_darling_test import AndersonDarlingTest
from monitoring.stat_tests.kl_divergence_test import KLDivergenceTest
from monitoring.stat_tests.ttest import TTest
from monitoring.stat_tests.mannwhitneyu_test import MannWhitneyUTest
from monitoring.stat_tests.levene_test import LeveneTest
from monitoring.stat_tests.jensen_shannon_test import JensenShannonTest
from monitoring.stat_tests.mmd_test import MMDTest
import logging
import json
import concurrent.futures
import pandas as pd
import numpy as np
from monitoring.drift_report import generate_html_report
import time
from datetime import datetime
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
def run_drift_tests(reference_data, current_data, columns_to_test=None, columns_to_exclude=None, test_config=None, html_report=True, html_report_path='drift_report.html'):
    # Compute summary statistics for each column in reference and current data
    def get_summary(df):
        summary = {}
        for col in df.columns:
            col_data = df[col]
            if pd.api.types.is_numeric_dtype(col_data):
                summary[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'missing': int(col_data.isnull().sum()),
                    'mode': None,
                    'unique': None
                }
            else:
                mode_val = col_data.mode().iloc[0] if not col_data.mode().empty else None
                summary[col] = {
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'missing': int(col_data.isnull().sum()),
                    'mode': str(mode_val) if mode_val is not None else None,
                    'unique': int(col_data.nunique())
                }
        return summary

    
    def log(msg):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{now}] [data_drift] {msg}')

    program_start = time.time()
    log('Starting run_drift_tests...')
    if columns_to_test and columns_to_exclude:
        raise ValueError("Set only one: columns_to_test or columns_to_exclude.")
    
    
    columns = list(reference_data.columns)
    if columns_to_test:
        columns = [col for col in columns if col in columns_to_test]
    elif columns_to_exclude:
        columns = [col for col in columns if col not in columns_to_exclude]

    default_thresholds = {
        'ks': 0.05, 'psi': 0.2, 'chi2': 0.05, 'wasserstein': 0.1,
        'anderson_darling': 0.05, 'kl_divergence': 0.1, 'ttest': 0.05,
        'mannwhitneyu': 0.05, 'levene': 0.05,
        'jensen_shannon': 0.1, 'mmd': 0.05
    }
    
    
    test_counts = {col: len(test_config.get(col, [])) if test_config else 0 for col in columns}
    numerical_tests = ['ks', 'wasserstein', 'psi'] 
    categorical_tests = ['chi2'] 

    def process_column(col):
        col_tests_result = {}
        drift_votes = 0
        total_tests = 0
        local_structured_log = []
        col_plot_data = {} 

        tests_to_run = test_config.get(col, []) if test_config and col in test_config else []
        if not tests_to_run:
            selected_tests = numerical_tests if pd.api.types.is_numeric_dtype(reference_data[col]) else categorical_tests
            num_tests = len(selected_tests)
            tests_to_run = [{'test': t, 'weight': 1.0 / num_tests if num_tests > 0 else 1.0} for t in selected_tests]
            tests_to_run = [{'test': t, 'weight': 1.0 / num_tests if num_tests > 0 else 1.0} for t in selected_tests]
        else:
            num_tests = test_counts.get(col, 1)
            # num_tests = test_counts.get(col, 1)

        for test_cfg in tests_to_run:
            test_name = test_cfg['test']
            threshold = test_cfg.get('threshold', default_thresholds.get(test_name))
            weight = test_cfg.get('weight', 1.0 / num_tests if num_tests > 0 else 1.0)
            
            test_class_map = {
                'ks': KSStatTest, 'psi': PSITest, 'chi2': Chi2Test, 'wasserstein': WassersteinTest,
                'anderson_darling': AndersonDarlingTest, 'kl_divergence': KLDivergenceTest, 'ttest': TTest,
                'mannwhitneyu': MannWhitneyUTest, 'levene': LeveneTest,
                'jensen_shannon': JensenShannonTest, 'mmd': MMDTest
            }

            res = None
            if test_name in test_class_map:
                test_instance = test_class_map[test_name]()
                # The run method now accepts **kwargs, so `plot=False` is fine
                res = test_instance.run(reference_data[col], current_data[col], threshold=threshold, plot=False)
            else:
                res = {'error': f'Unknown test: {test_name}'}

            if res is None: res = {}
            
            if 'plot_data' in res and isinstance(res.get('plot_data'), dict):
                col_plot_data[test_name] = res.pop('plot_data')

            drift = False
            if 'error' in res:
                # If a test returns an error (e.g., wrong data type), it cannot detect drift.
                drift = False
            elif 'drift' in res:
                # Use the 'drift' flag provided by the test if available.
                drift = res['drift']
            elif 'p_value' in res and threshold is not None:
                # Fallback for p-value based tests, ensuring p_value is not NaN.
                p_val = res['p_value']
                if p_val is not None and not np.isnan(p_val):
                    drift = p_val < threshold

            col_tests_result[test_name] = {
                'result': res,
                'drift': drift,
                'weight': weight,
                'threshold': threshold
            }
            local_structured_log.append({
                'column': col, 'test': test_name, 'result': res,
                'drift_detected': drift, 'threshold': threshold, 'weight': weight
            })
            if drift:
                drift_votes += 1
            total_tests += 1
        
        col_drift = drift_votes > (total_tests / 2) if total_tests > 0 else False
        
        column_summary = {'tests': col_tests_result, 'is_drift': col_drift}
        return col, column_summary, col_plot_data, local_structured_log
 
    structured_log = []
    results = {}
    plot_data_all = {}
    process_start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_column, col): col for col in columns}
        for future in concurrent.futures.as_completed(futures):
            col, col_summary, col_plot_data, local_log = future.result()
            results[col] = col_summary
            plot_data_all[col] = col_plot_data
            structured_log.extend(local_log)
    log(f'All columns processed in {time.time() - process_start:.3f} seconds')
 
    is_drift = any(col_result['is_drift'] for col_result in results.values())
 
    log_object = {
        'reference_meta': {'shape': reference_data.shape, 'columns': list(reference_data.columns)},
        'current_meta': {'shape': current_data.shape, 'columns': list(current_data.columns)},
        'reference_meta': {'shape': reference_data.shape, 'columns': list(reference_data.columns)},
        'current_meta': {'shape': current_data.shape, 'columns': list(current_data.columns)},
        'tests': structured_log
    }
    log_write_start = time.time()
    with open('drift_structured_log.json', 'w', encoding='utf-8') as log_file:
        json.dump(log_object, log_file, indent=2, default=str)
    log(f'drift_structured_log.json written in {time.time() - log_write_start:.3f} seconds')
 
    # Ensure all bools in results are standard Python bool (not numpy.bool_)
    def convert_bools(obj):
        if isinstance(obj, dict):
            return {k: convert_bools(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_bools(v) for v in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    data_summary = {
        'columns': list(reference_data.columns),
        'ref': get_summary(reference_data),
        'cur': get_summary(current_data)
    }
    final_result = {
        'is_drift': bool(is_drift),
        'details': convert_bools(results),
        'data_summary': data_summary,
        'reference_meta': {'shape': reference_data.shape, 'columns': list(reference_data.columns)},
        'current_meta': {'shape': current_data.shape, 'columns': list(current_data.columns)}
    }

    html_report_start = time.time()
    if html_report:
        test_class_map = {
            'ks': KSStatTest, 'psi': PSITest, 'chi2': Chi2Test, 'wasserstein': WassersteinTest,
            'anderson_darling': AndersonDarlingTest, 'kl_divergence': KLDivergenceTest, 'ttest': TTest,
            'mannwhitneyu': MannWhitneyUTest, 'levene': LeveneTest,
            'jensen_shannon': JensenShannonTest, 'mmd': MMDTest
        }
        # Prepare a copy for HTML report with plot_html, but do not add plot_html to the returned JSON
        html_result = json.loads(json.dumps(final_result))  # deep copy
        for col, test_plots_data in plot_data_all.items():
            for test_name, pdata in test_plots_data.items():
                TestClass = test_class_map.get(test_name)
                if TestClass and hasattr(TestClass, 'make_plot_from_data'):
                    try:
                        fig = TestClass.make_plot_from_data(pdata)
                        html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
                        html_result['details'][col]['tests'][test_name]['plot_html'] = html_str
                    except Exception as e:
                        error_msg = f'Plot error: {e}'
                        html_result['details'][col]['tests'][test_name]['plot_html'] = error_msg
                        logger.error(f"Failed to generate plot for {col}/{test_name}: {e}")

        try:
            log('[drift_report] Generating HTML report...')
            generate_html_report(html_result, output_path=html_report_path)
            log(f'[drift_report] HTML report generated at {html_report_path}')
            log(f'HTML report step took {time.time() - html_report_start:.3f} seconds')
            final_result['html_report_path'] = html_report_path
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            final_result['html_report_error'] = str(e)

    # Remove plot_html from the returned JSON if present (defensive, should not be present)
    for col in final_result.get('details', {}):
        for test in final_result['details'][col].get('tests', {}):
            final_result['details'][col]['tests'][test].pop('plot_html', None)
    total_time = time.time() - program_start
    log(f'Program terminated. Total run_drift_tests time: {total_time:.3f} seconds')
    return final_result