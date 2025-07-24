import numpy as np
import pandas as pd
from scipy.stats import entropy
import plotly.graph_objects as go

def _calculate_kl_and_bins(reference_data: pd.Series, current_data: pd.Series, feature_type: str, n_bins: int):
    """
    Consolidates the logic for binning data and calculating the KL Divergence.
    KL(P || Q) where P=current, Q=reference.
    """
    epsilon = 1e-8 # A small constant to avoid division by zero

    # Bin data based on feature type
    if feature_type == "categorical":
        all_categories = pd.unique(np.concatenate([reference_data.dropna(), current_data.dropna()]))
        ref_counts = reference_data.value_counts().reindex(all_categories, fill_value=0)
        curr_counts = current_data.value_counts().reindex(all_categories, fill_value=0)
    else: # Numerical
        all_data = np.concatenate([reference_data.dropna(), current_data.dropna()])
        bins = np.unique(np.quantile(all_data, np.linspace(0, 1, n_bins + 1)))

        # If all values are same
        if len(bins) < 2: bins = np.array([all_data.min(), all_data.max() + epsilon])
        
        ref_counts = np.histogram(reference_data.dropna(), bins=bins)[0]
        curr_counts = np.histogram(current_data.dropna(), bins=bins)[0]
        bins_or_cats = bins

    # Normalize counts to get probability distributions, adding epsilon for stability
    ref_perc = (ref_counts + epsilon) / (np.sum(ref_counts) + epsilon * len(ref_counts))
    curr_perc = (curr_counts + epsilon) / (np.sum(curr_counts) + epsilon * len(curr_counts))
    
    if feature_type == "categorical":
        bins_or_cats = all_categories

    # Calculate KL Divergence: KL(current || reference)
    kl_divergence = entropy(pk=curr_perc, qk=ref_perc)
    
    return kl_divergence, ref_perc, curr_perc, bins_or_cats


class KLDivergenceTest:
    def __init__(self, params=None):
        self.params = params

    def run(self, reference_data, current_data, threshold=0.1, n_bins=10, **kwargs):
        """
        Runs the Kullback-Leibler Divergence test. Automatically determines data type.
        """
        ref_series = pd.Series(reference_data)
        curr_series = pd.Series(current_data)
        
        feature_type = "numerical" if pd.api.types.is_numeric_dtype(ref_series) else "categorical"

        kl_divergence, ref_perc, curr_perc, bins_or_cats = _calculate_kl_and_bins(
            ref_series, curr_series, feature_type, n_bins
        )
        
        # A higher KL divergence value indicates more drift
        drift_detected = kl_divergence >= threshold

        plot_data = {
            'column_name': ref_series.name or "Value",
            'feature_type': feature_type,
            'kl_divergence': kl_divergence,
            'drift_detected': drift_detected,
            'threshold': threshold,
            'ref_perc': ref_perc.tolist() if hasattr(ref_perc, 'tolist') else ref_perc,
            'curr_perc': curr_perc.tolist() if hasattr(curr_perc, 'tolist') else curr_perc,
            'bins_or_cats': bins_or_cats.tolist() if hasattr(bins_or_cats, 'tolist') else bins_or_cats,
        }

        result = {
            'kl_divergence': kl_divergence,
            'drift': drift_detected,
            'plot_data': plot_data
        }
        return result

    @staticmethod
    def make_plot_from_data(plot_data):
        """
        Generates an interactive Plotly bar chart showing the two distributions
        that were used to calculate KL Divergence.
        """
        try:
            if not plot_data or not plot_data.get('bins_or_cats'):
                raise ValueError("Not enough data to create plot.")

            column_name = plot_data.get('column_name', 'Value')
            kl_divergence = plot_data.get('kl_divergence', 0.0)
            drift_status = "Yes" if plot_data.get('drift_detected') else "No"
            feature_type = plot_data.get('feature_type')
            
            ref_perc = plot_data.get('ref_perc')
            curr_perc = plot_data.get('curr_perc')
            bins_or_cats = plot_data.get('bins_or_cats')

            fig = go.Figure()
            
            if feature_type == "categorical":
                x_labels = [str(c) for c in bins_or_cats] # Ensure labels are strings
                fig.add_trace(go.Bar(name='Reference', x=x_labels, y=ref_perc, marker_color='#1f77b4'))
                fig.add_trace(go.Bar(name='Current', x=x_labels, y=curr_perc, marker_color='#d62728'))
                fig.update_xaxes(tickangle=-45)
            else: # Numerical
                bin_centers = (np.array(bins_or_cats[:-1]) + np.array(bins_or_cats[1:])) / 2
                fig.add_trace(go.Bar(name='Reference', x=bin_centers, y=ref_perc, marker_color='#1f77b4'))
                fig.add_trace(go.Bar(name='Current', x=bin_centers, y=curr_perc, marker_color='#d62728'))

            # Handle infinite values for display
            kl_str = f"{kl_divergence:.4f}" if np.isfinite(kl_divergence) else "inf"

            fig.update_layout(
                title_text=(
                    f"<b>Kullback-Leibler Divergence: '{column_name}'</b><br>"
                    f"KL Divergence (Current || Reference): {kl_str} | Drift Detected: {drift_status}"
                ),
                xaxis_title=f"Bins / Categories for {column_name}",
                yaxis_title="Proportion of Observations",
                barmode='group',
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            return fig

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f'KL Divergence Plot generation error: {e}', x=0.5, y=0.5, showarrow=False)
            return fig