# monitoring/stat_tests/ttest.py

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import plotly.graph_objects as go
import plotly.figure_factory as ff

def _calculate_ttest(reference_data: pd.Series, current_data: pd.Series):
    """
    Performs a two-sample t-test, returning the statistic and p-value.
    Handles empty data gracefully.
    """
    if reference_data.empty or current_data.empty:
        return np.nan, np.nan
    
    # equal_var=False performs Welch's T-test, which is more robust
    # and does not assume equal population variances.
    stat, p_value = ttest_ind(reference_data, current_data, nan_policy='omit', equal_var=False)
    return stat, p_value


class TTest:
    def __init__(self, params=None):
        self.params = params

    def run(self, reference_data, current_data, threshold=0.05, **kwargs):
        """
        Runs Welch's T-test, but first checks if the data is numerical.
        If not, returns a structured message indicating the test was skipped.
        Ignores any extra keyword arguments like 'plot'.
        """
        ref_series = pd.Series(reference_data)

        # ✅ --- CRITICAL FIX: DATA TYPE VALIDATION --- ✅
        # Check if the data type is appropriate for a T-test before proceeding.
        if not pd.api.types.is_numeric_dtype(ref_series):
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'drift': False,  # No drift as test is not applicable
                'plot_data': {}, # Must return an empty dict for downstream consistency
                'error': 'T-test is only applicable to numerical data.'
            }
        # ✅ --- END OF FIX --- ✅
        
        # If the check passes, proceed with the original logic.
        ref_series = ref_series.dropna()
        curr_series = pd.Series(current_data).dropna()
        
        stat, p_value = _calculate_ttest(ref_series, curr_series)
        
        if p_value is None or np.isnan(p_value):
            drift_detected = False
        else:
            drift_detected = p_value < threshold

        plot_data = {
            'reference': ref_series.tolist(),
            'current': curr_series.tolist(),
            'column_name': ref_series.name or "Value",
            'p_value': p_value,
            'statistic': stat,
            'drift_detected': drift_detected,
            'ref_mean': ref_series.mean(),
            'curr_mean': curr_series.mean()
        }

        result = {
            'statistic': stat,
            'p_value': p_value,
            'drift': drift_detected,
            'plot_data': plot_data
        }
        return result

    @staticmethod
    def make_plot_from_data(plot_data):
        """
        Generates an interactive Plotly Figure to visualize the T-test results.
        This plot shows both the distributions and their respective means.
        """
        try:
            # Gracefully handle the case where a plot is requested for an errored test
            if not plot_data:
                fig = go.Figure()
                fig.add_annotation(text="Test not applicable (e.g., non-numeric data).", x=0.5, y=0.5, showarrow=False)
                return fig

            column_name = plot_data.get('column_name', 'Value')
            p_value = plot_data.get('p_value')
            drift_status = "Yes" if plot_data.get('drift_detected') else "No"
            ref_mean = plot_data.get('ref_mean')
            curr_mean = plot_data.get('curr_mean')

            ref_data = plot_data.get('reference', [])
            curr_data = plot_data.get('current', [])

            if not ref_data or not curr_data:
                raise ValueError("Not enough data to create plot.")

            # Create a distplot (density plot with histogram)
            hist_data = [ref_data, curr_data]
            group_labels = ['Reference', 'Current']
            colors = ['#1f77b4', '#d62728']

            fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, colors=colors, show_rug=False)

            # Add vertical lines and annotations for the means
            fig.add_vline(x=ref_mean, line_width=2, line_dash="dash", line_color=colors[0])
            fig.add_vline(x=curr_mean, line_width=2, line_dash="dash", line_color=colors[1])

            fig.add_annotation(
                x=ref_mean, y=0, text=f"Mean: {ref_mean:.2f}",
                showarrow=True, arrowhead=1, ax=0, ay=-60,
                bordercolor=colors[0], borderwidth=1, bgcolor="#ffffff"
            )
            fig.add_annotation(
                x=curr_mean, y=0, text=f"Mean: {curr_mean:.2f}",
                showarrow=True, arrowhead=1, ax=0, ay=-90,
                bordercolor=colors[1], borderwidth=1, bgcolor="#ffffff"
            )
            
            p_value_str = f"{p_value:.4f}" if p_value is not None and not np.isnan(p_value) else "N/A"
            title = (f"<b>T-Test: Comparison of Means for '{column_name}'</b><br>"
                     f"P-value: {p_value_str} | Drift Detected: {drift_status}")

            fig.update_layout(
                title_text=title,
                xaxis_title=f"Value of {column_name}",
                yaxis_title="Density",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            return fig

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f'T-Test Plot generation error: {e}', x=0.5, y=0.5, showarrow=False)
            return fig