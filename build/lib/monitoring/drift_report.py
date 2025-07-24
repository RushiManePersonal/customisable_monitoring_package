import json
from jinja2 import Environment, FileSystemLoader
import os
import numpy as np 

def generate_html_report(result, output_path='drift_report.html'):
    """
    Generates the final HTML report from the drift test results.
    """

    import time
    from datetime import datetime
    def log(msg):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{now}] [drift_report] {msg}')

    start_time = time.time()
    log('Starting HTML report generation...')
    env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    template = env.get_template('drift_report_template.html')
    log('Template loaded.')

    sanitize_start = time.time()
    # Sanitize the result dictionary to convert NumPy types to standard Python types.
    for col, details in result.get('details', {}).items():
        for test_name, test_info in details.get('tests', {}).items():
            if 'drift' in test_info and isinstance(test_info['drift'], np.bool_):
                test_info['drift'] = bool(test_info['drift'])
            if 'result' in test_info and isinstance(test_info['result'], dict):
                for key, value in test_info['result'].items():
                    if isinstance(value, np.bool_):
                        test_info['result'][key] = bool(value)
                    elif isinstance(value, np.integer):
                        test_info['result'][key] = int(value)
                    elif isinstance(value, np.floating):
                        test_info['result'][key] = float(value)
                    elif isinstance(value, np.ndarray):
                        test_info['result'][key] = value.tolist()
    log(f'Result sanitization took {time.time() - sanitize_start:.3f} seconds')

    render_start = time.time()
    html = template.render(result=result)
    log(f'Template rendering took {time.time() - render_start:.3f} seconds')

    write_start = time.time()
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    log(f'File write took {time.time() - write_start:.3f} seconds')
    log(f'HTML report saved to {output_path}')
    total_time = time.time() - start_time
    log(f'Total drift_report.py time: {total_time:.3f} seconds')
    # Print final program termination time and total time
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{end_time}] [drift_report] Program terminated. Total time: {total_time:.3f} seconds')

def drop_unnamed_column(df):
    """Automatically drops the 'Unnamed: 0' column from DataFrames if present."""
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

# Example usage: wherever a DataFrame is loaded or passed in, call drop_unnamed_column(df)