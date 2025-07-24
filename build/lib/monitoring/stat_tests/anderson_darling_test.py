# monitoring/stat_tests/anderson_darling_test.py

import numpy as np
import pandas as pd
from scipy.stats import anderson_ksamp
import plotly.graph_objects as go

def _calculate_ad(reference_data: pd.Series, current_data: pd.Series):
    """
    Performs the k-sample Anderson-Darling test.
    Returns the full result object from scipy.
    """
    if reference_data.empty or current_data.empty:
        return None
    
    try:
        return anderson_ksamp([reference_data, current_data])
    except ValueError:
        return None

class AndersonDarlingTest:
    def __init__(self, params=None):
        self.params = params

    def run(self, reference_data, current_data, threshold=0.05, **kwargs):
        """
        Runs the Anderson-Darling test. Drift is determined if the test statistic
        exceeds the critical value for the given significance level (threshold).
        """
        ref_series = pd.Series(reference_data)
        curr_series = pd.Series(current_data)

        if not pd.api.types.is_numeric_dtype(ref_series):
            return {
                'statistic': np.nan, 'critical_values': [], 'drift': False,
                'plot_data': {}, 'error': 'Anderson-Darling test is only applicable to numerical data.'
            }

        ref_series = ref_series.dropna()
        curr_series = curr_series.dropna()
        
        ad_result = _calculate_ad(ref_series, curr_series)
        
        drift_detected = False
        statistic = np.nan
        critical_values_list = [] # Initialize as empty list
        significance_levels = [0.25, 0.10, 0.05, 0.025, 0.01] 

        if ad_result:
            statistic = ad_result.statistic
            # ✅ --- START OF PRIMARY FIX --- ✅
            # Convert the NumPy array to a standard Python list
            critical_values_list = ad_result.critical_values.tolist()
            # ✅ ---  END OF PRIMARY FIX  --- ✅
            
            try:
                sig_level_index = significance_levels.index(threshold)
                critical_value_at_threshold = critical_values_list[sig_level_index]
                drift_detected = statistic > critical_value_at_threshold
            except (ValueError, IndexError):
                return {
                    'statistic': statistic, 'critical_values': critical_values_list,
                    'drift': False, 'plot_data': {},
                    'error': f"Invalid threshold '{threshold}'. Must be one of {significance_levels}."
                }
        
        plot_data = {
            'reference': ref_series.tolist(), 'current': curr_series.tolist(),
            'column_name': ref_series.name or "Value", 'statistic': statistic,
            'critical_values': critical_values_list, 'significance_levels': significance_levels,
            'drift_detected': drift_detected, 'threshold': threshold
        }

        result = {
            'statistic': statistic,
            'critical_values': critical_values_list, # Now it's a list
            'drift': drift_detected,
            'plot_data': plot_data
        }
        return result

    @staticmethod
    def make_plot_from_data(plot_data):
        # This method remains unchanged and is correct.
        try:
            if 'error' in plot_data or not plot_data or not plot_data.get('reference'):
                error_msg = plot_data.get('error', 'Test not applicable or not enough data.')
                raise ValueError(error_msg)
            column_name = plot_data.get('column_name', 'Value')
            statistic = plot_data.get('statistic')
            drift_status = "Yes" if plot_data.get('drift_detected') else "No"
            threshold = plot_data.get('threshold')
            ref_data = plot_data.get('reference', [])
            curr_data = plot_data.get('current', [])
            ref_sorted = np.sort(ref_data)
            curr_sorted = np.sort(curr_data)
            ref_ecdf = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
            curr_ecdf = np.arange(1, len(curr_sorted) + 1) / len(curr_sorted)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ref_sorted, y=ref_ecdf, mode='lines', name='Reference ECDF', line=dict(color='#1f77b4', shape='hv')))
            fig.add_trace(go.Scatter(x=curr_sorted, y=curr_ecdf, mode='lines', name='Current ECDF', line=dict(color='#d62728', shape='hv')))
            stat_str = f"{statistic:.4f}" if statistic is not None and not np.isnan(statistic) else "N/A"
            title = (f"<b>Anderson-Darling Test: '{column_name}'</b><br>"f"Statistic: {stat_str} | Drift Detected at {threshold*100}% significance: {drift_status}")
            fig.update_layout(title_text=title, xaxis_title=f"Value of {column_name}", yaxis_title="Cumulative Probability", height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            return fig
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f'A-D Plot generation error: {e}', x=0.5, y=0.5, showarrow=False)
            return fig