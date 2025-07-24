# monitoring/stat_tests/wasserstein_test.py

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# The helper function can be simplified as it's only used for calculation now
def _calculate_wasserstein(reference_data, current_data):
    """Calculates raw and normalized Wasserstein distance."""
    ref = pd.Series(reference_data).dropna()
    curr = pd.Series(current_data).dropna()

    if ref.empty or curr.empty:
        return 0.0, 0.0

    # Raw distance
    wd_raw = stats.wasserstein_distance(ref, curr)
    
    # Normalize by the standard deviation of the reference data
    ref_std = np.std(ref)
    norm_factor = ref_std if ref_std > 0 else 1.0
    wd_norm_value = wd_raw / norm_factor
    
    return wd_raw, wd_norm_value


class WassersteinTest:
    def __init__(self, params=None):
        self.params = params

    def run(self, reference_data, current_data, threshold=0.1, column_name="Value", plot=False):
        """
        Runs the Wasserstein distance test.
        NOTE: The 'plot' parameter is ignored here, as plotting is handled
        in the main thread using make_plot_from_data.
        """
        ref_series = pd.Series(reference_data).dropna()
        curr_series = pd.Series(current_data).dropna()

        raw_dist, norm_dist = _calculate_wasserstein(ref_series, curr_series)
        
        drift_detected = norm_dist >= threshold

        # This dictionary is what the main thread needs to generate the plot.
        plot_data = {
            'reference': ref_series.tolist(),
            'current': curr_series.tolist(),
            'column_name': column_name,
            'threshold': threshold,
            'norm_dist': norm_dist,
            'drift_detected': drift_detected
        }

        result = {
            'wasserstein_distance': raw_dist,
            'wasserstein_distance_norm': norm_dist,
            'drift': drift_detected,
            'plot_data': plot_data
        }

        return result

    @staticmethod
    def make_plot_from_data(plot_data):
        """
        Generates a Plotly Figure from the provided plot_data dictionary.
        This method is called in the main thread of data_drift.py.
        """
        try:
            ref = plot_data.get('reference', [])
            curr = plot_data.get('current', [])
            column_name = plot_data.get('column_name', 'Value')
            norm_dist = plot_data.get('norm_dist', 0.0)
            drift_detected = 'Yes' if plot_data.get('drift_detected') else 'No'

            # Handle empty data case
            if not ref or not curr:
                fig = go.Figure()
                fig.add_annotation(text=f"Not enough data to plot for column '{column_name}'", x=0.5, y=0.5, showarrow=False)
                return fig

            # Create subplots: one for histogram, one for CDF
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    "Data Distributions (Histogram)",
                    f"Cumulative Distributions (CDF)<br>Wasserstein Distance (norm): {norm_dist:.4f} | Drift: {drift_detected}"
                )
            )

            # 1. Histogram
            fig.add_trace(go.Histogram(x=ref, name='Reference', opacity=0.7, marker_color='blue', histnorm='probability density'), row=1, col=1)
            fig.add_trace(go.Histogram(x=curr, name='Current', opacity=0.7, marker_color='red', histnorm='probability density'), row=1, col=1)

            # 2. CDFs
            all_values = np.sort(np.unique(np.concatenate([ref, curr])))
            ref_cdf = np.searchsorted(np.sort(ref), all_values, side='right') / len(ref)
            curr_cdf = np.searchsorted(np.sort(curr), all_values, side='right') / len(curr)

            fig.add_trace(go.Scatter(x=all_values, y=ref_cdf, mode='lines', name='Reference CDF', line=dict(color='blue', shape='hv')), row=2, col=1)
            fig.add_trace(go.Scatter(x=all_values, y=curr_cdf, mode='lines', name='Current CDF', line=dict(color='red', shape='hv')), row=2, col=1)

            # Update axes and layout
            fig.update_xaxes(title_text=str(column_name), row=1, col=1)
            fig.update_yaxes(title_text="Density", row=1, col=1)
            fig.update_xaxes(title_text=str(column_name), row=2, col=1)
            fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)
            
            fig.update_layout(
                height=800,
                showlegend=True,
                barmode='overlay',
                title_text=f"<b>Wasserstein Test: {column_name}</b>",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.data[0].showlegend = True
            fig.data[1].showlegend = True
            fig.data[2].showlegend = False # Hide CDF legend items to avoid clutter
            fig.data[3].showlegend = False

            return fig
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f'Plot generation error: {e}', x=0.5, y=0.5, showarrow=False)
            return fig