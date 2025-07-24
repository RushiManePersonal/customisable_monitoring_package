# monitoring/stat_tests/ks_stat_test.py

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _calculate_ks(reference_data: pd.Series, current_data: pd.Series):
    """
    Performs the two-sample Kolmogorov-Smirnov test.
    Returns the KS statistic and the p-value.
    """
    if reference_data.empty or current_data.empty:
        return np.nan, np.nan
    
    statistic, p_value = ks_2samp(reference_data, current_data)
    return statistic, p_value


class KSStatTest:
    @staticmethod
    def make_plot_from_data(plot_data):
        """
        Given plot_data (dict with 'reference', 'current', 'column_name'), generate and return a Plotly Figure.
        """
        import plotly.graph_objects as go
        import numpy as np
        ref = np.array(plot_data.get('reference', []))
        curr = np.array(plot_data.get('current', []))
        column_name = plot_data.get('column_name', 'Value')
        threshold = plot_data.get('threshold', 0.05)
        # Defensive: handle empty data
        if len(ref) == 0 or len(curr) == 0:
            fig = go.Figure()
            fig.add_annotation(text='No data for plot', x=0.5, y=0.5, showarrow=False)
            return fig
        # Histogram
        hist_ref = go.Histogram(x=ref, name='Reference', opacity=0.5, marker_color='blue', histnorm='probability density')
        hist_curr = go.Histogram(x=curr, name='Current', opacity=0.5, marker_color='red', histnorm='probability density')
        # CDFs
        all_values = np.sort(np.unique(np.concatenate([ref, curr])))
        ref_cdf = np.searchsorted(np.sort(ref), all_values, side='right') / len(ref)
        curr_cdf = np.searchsorted(np.sort(curr), all_values, side='right') / len(curr)
        cdf_ref = go.Scatter(x=all_values, y=ref_cdf, mode='lines', name='Reference CDF', line=dict(color='blue'))
        cdf_curr = go.Scatter(x=all_values, y=curr_cdf, mode='lines', name='Current CDF', line=dict(color='red'))
        # Subplots
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Data Distributions (Histogram)", "Empirical CDFs"))
        fig.add_trace(hist_ref, row=1, col=1)
        fig.add_trace(hist_curr, row=1, col=1)
        fig.add_trace(cdf_ref, row=2, col=1)
        fig.add_trace(cdf_curr, row=2, col=1)
        fig.update_xaxes(title_text=str(column_name), row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_xaxes(title_text=str(column_name), row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)
        fig.update_layout(height=700, showlegend=True, barmode='overlay', title_text=f"KS Test Diagnostic Plot: {column_name}")
        return fig
    def __init__(self, params=None):
        self.params = params

    def run(self, reference_data, current_data, threshold=0.1, **kwargs):
        """
        Runs the Kolmogorov-Smirnov test. Drift is determined if the KS statistic
        is greater than the threshold.
        
        Default threshold of 0.1 is a common rule of thumb for the KS statistic.
        """
        ref_series = pd.Series(reference_data)

        # ✅ Data type validation: KS test is for continuous numerical data.
        if not pd.api.types.is_numeric_dtype(ref_series):
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'drift': False,
                'plot_data': {},
                'error': 'KS test is only applicable to numerical data.'
            }

        ref_series = ref_series.dropna()
        curr_series = pd.Series(current_data).dropna()
        
        statistic, p_value = _calculate_ks(ref_series, curr_series)
        
        # ✅ Drift decision based on the statistic, not the p-value.
        # A larger statistic means a larger difference between distributions.
        if statistic is None or np.isnan(statistic):
            drift_detected = False
        else:
            drift_detected = statistic > threshold

        plot_data = {
            'reference': ref_series.tolist(),
            'current': curr_series.tolist(),
            'column_name': ref_series.name or "Value",
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected,
            'threshold': threshold
        }

        result = {
            'statistic': statistic,
            'p_value': p_value,
            'drift': drift_detected,
            'plot_data': plot_data
        }
        return result

    @staticmethod
    def make_plot_from_data(plot_data):
        """
        Generates an interactive Plotly Figure for the KS test, highlighting the
        KS statistic on the CDF plot.
        """
        try:
            if not plot_data:
                fig = go.Figure()
                fig.add_annotation(text="Test not applicable (e.g., non-numeric data).", x=0.5, y=0.5, showarrow=False)
                return fig

            column_name = plot_data.get('column_name', 'Value')
            statistic = plot_data.get('statistic')
            drift_status = "Yes" if plot_data.get('drift_detected') else "No"
            threshold = plot_data.get('threshold')

            ref_data = plot_data.get('reference', [])
            curr_data = plot_data.get('current', [])

            if not ref_data or not curr_data:
                raise ValueError("Not enough data to create plot.")

            # Create subplots: one for histogram, one for CDF
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    "Data Distributions",
                    f"Empirical CDFs<br>KS Statistic: {statistic:.4f} | Drift: {drift_status}"
                ),
                vertical_spacing=0.15
            )

            # 1. Histogram plot
            fig.add_trace(go.Histogram(x=ref_data, name='Reference', opacity=0.7, marker_color='#1f77b4', histnorm='probability density'), row=1, col=1)
            fig.add_trace(go.Histogram(x=curr_data, name='Current', opacity=0.7, marker_color='#d62728', histnorm='probability density'), row=1, col=1)

            # 2. CDF plot
            all_values = np.sort(np.unique(np.concatenate([ref_data, curr_data])))
            ref_cdf = np.searchsorted(np.sort(ref_data), all_values, side='right') / len(ref_data)
            curr_cdf = np.searchsorted(np.sort(curr_data), all_values, side='right') / len(curr_data)

            fig.add_trace(go.Scatter(x=all_values, y=ref_cdf, mode='lines', name='Reference CDF', line=dict(color='#1f77b4', shape='hv')), row=2, col=1)
            fig.add_trace(go.Scatter(x=all_values, y=curr_cdf, mode='lines', name='Current CDF', line=dict(color='#d62728', shape='hv')), row=2, col=1)
            
            # ✅ Add visualization for the KS statistic
            # Find the point of maximum distance
            max_dist_idx = np.argmax(np.abs(ref_cdf - curr_cdf))
            max_dist_x = all_values[max_dist_idx]
            ref_y_at_max = ref_cdf[max_dist_idx]
            curr_y_at_max = curr_cdf[max_dist_idx]
            
            # Add a line segment representing the KS statistic
            fig.add_trace(go.Scatter(
                x=[max_dist_x, max_dist_x],
                y=[ref_y_at_max, curr_y_at_max],
                mode='lines',
                line=dict(color='black', width=3, dash='dash'),
                name='KS Statistic'
            ), row=2, col=1)

            # Update axes and layout
            fig.update_xaxes(title_text=column_name, row=1, col=1)
            fig.update_yaxes(title_text="Density", row=1, col=1)
            fig.update_xaxes(title_text=column_name, row=2, col=1)
            fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)
            
            fig.update_layout(
                height=800,
                showlegend=True,
                barmode='overlay',
                title_text=f"<b>Kolmogorov-Smirnov Test: {column_name}</b>",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            return fig

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f'KS Test Plot generation error: {e}', x=0.5, y=0.5, showarrow=False)
            return fig