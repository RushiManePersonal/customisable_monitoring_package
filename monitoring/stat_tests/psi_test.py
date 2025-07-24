import numpy as np
import pandas as pd
import plotly.graph_objects as go

# A private helper function to handle all calculations in one place.
def _calculate_psi_and_bins(reference_data: pd.Series, current_data: pd.Series, feature_type: str, n_bins: int):
    """
    Calculates PSI and returns the binned data for plotting.
    This function consolidates the logic from the old get_binned_data and psi_stat_test.
    """
    # Define a small constant to avoid division by zero or log(0)
    epsilon = 1e-8

    if feature_type == "categorical":
        # Combine all unique categories from both datasets
        all_categories = pd.unique(np.concatenate([reference_data.dropna(), current_data.dropna()]))
        
        # Calculate value counts for each dataset, ensuring all categories are present
        ref_counts = reference_data.value_counts().reindex(all_categories, fill_value=0)
        curr_counts = current_data.value_counts().reindex(all_categories, fill_value=0)

        # Calculate percentages
        ref_perc = (ref_counts + epsilon) / max(len(reference_data), 1)
        curr_perc = (curr_counts + epsilon) / max(len(current_data), 1)
        
        bins_or_cats = ref_perc.index.astype(str) # Use index for labels
    else: # Numerical
        # Use quantiles on the combined data to define robust bins
        all_data = np.concatenate([reference_data.dropna(), current_data.dropna()])
        bins = np.unique(np.quantile(all_data, np.linspace(0, 1, n_bins + 1)))

        # If bins are not unique (e.g., all values are the same), create a single bin
        if len(bins) < 2:
            bins = np.array([all_data.min(), all_data.max() + epsilon])

        ref_hist = np.histogram(reference_data.dropna(), bins=bins)[0]
        curr_hist = np.histogram(current_data.dropna(), bins=bins)[0]

        ref_perc = (ref_hist + epsilon) / max(len(reference_data), 1)
        curr_perc = (curr_hist + epsilon) / max(len(current_data), 1)

        bins_or_cats = bins

    # Calculate PSI value
    psi_values = (ref_perc - curr_perc) * np.log(ref_perc / curr_perc)
    psi_value = np.sum(psi_values)

    return psi_value, ref_perc, curr_perc, bins_or_cats


class PSITest:
    def __init__(self, params=None):
        self.params = params

    def run(self, reference_data, current_data, threshold=0.2, n_bins=10, plot=False, **kwargs):
        """
        Runs the Population Stability Index (PSI) test.
        Automatically determines if the feature is numerical or categorical.
        """
        ref_series = pd.Series(reference_data)
        curr_series = pd.Series(current_data)

        # Automatically determine feature type based on data type
        if pd.api.types.is_numeric_dtype(ref_series):
            feature_type = "numerical"
        else:
            feature_type = "categorical"
            # For categorical, n_bins is irrelevant, but we pass it for consistency
            n_bins = len(pd.unique(np.concatenate([ref_series.dropna(), curr_series.dropna()])))


        psi_value, ref_perc, curr_perc, bins_or_cats = _calculate_psi_and_bins(ref_series, curr_series, feature_type, n_bins)

        # Standard PSI thresholds: < 0.1 no drift, 0.1-0.2 moderate drift, > 0.2 significant drift
        drift_detected = psi_value >= threshold

        # Package all data needed for plotting later
        plot_data = {
            'reference': ref_series.dropna().tolist(),
            'current': curr_series.dropna().tolist(),
            'column_name': ref_series.name or "Value",
            'feature_type': feature_type,
            'n_bins': n_bins,
            'psi_value': psi_value,
            'drift_detected': drift_detected,
            'ref_perc': ref_perc.tolist() if hasattr(ref_perc, 'tolist') else ref_perc,
            'curr_perc': curr_perc.tolist() if hasattr(curr_perc, 'tolist') else curr_perc,
            'bins_or_cats': bins_or_cats.tolist() if hasattr(bins_or_cats, 'tolist') else bins_or_cats,
        }

        result = {
            'psi': psi_value,
            'drift': drift_detected,
            'plot_data': plot_data
        }

        return result

    @staticmethod
    def make_plot_from_data(plot_data):
        """
        Generates an interactive Plotly Figure from the provided plot_data dictionary.
        This static method is called in the main thread of data_drift.py.
        """
        try:
            column_name = plot_data.get('column_name', 'Value')
            psi_value = plot_data.get('psi_value', 0.0)
            drift_status = "Yes" if plot_data.get('drift_detected') else "No"
            feature_type = plot_data.get('feature_type')
            
            ref_perc = plot_data.get('ref_perc')
            curr_perc = plot_data.get('curr_perc')
            bins_or_cats = plot_data.get('bins_or_cats')

            if ref_perc is None or curr_perc is None or bins_or_cats is None:
                raise ValueError("Binned data for plotting is missing.")

            fig = go.Figure()
            
            if feature_type == "categorical":
                x_labels = bins_or_cats
                fig.add_trace(go.Bar(name='Reference', x=x_labels, y=ref_perc, marker_color='#1f77b4'))
                fig.add_trace(go.Bar(name='Current', x=x_labels, y=curr_perc, marker_color='#d62728'))
            else: # Numerical
                # Bins define the edges, so we calculate the center for plotting
                bin_centers = (np.array(bins_or_cats[:-1]) + np.array(bins_or_cats[1:])) / 2
                fig.add_trace(go.Bar(name='Reference', x=bin_centers, y=ref_perc, marker_color='#1f77b4'))
                fig.add_trace(go.Bar(name='Current', x=bin_centers, y=curr_perc, marker_color='#d62728'))

            # Update layout for a professional look
            fig.update_layout(
                title_text=f"<b>Population Stability Index (PSI): {column_name}</b><br>PSI Value: {psi_value:.4f} | Drift Detected: {drift_status}",
                xaxis_title=f"Bins / Categories for {column_name}",
                yaxis_title="Proportion of Observations",
                barmode='group', # 'group' is often clearer for direct comparison than 'overlay'
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            return fig

        except Exception as e:
            # Return a figure with an error message if anything goes wrong
            fig = go.Figure()
            fig.add_annotation(text=f'PSI Plot generation error: {e}', x=0.5, y=0.5, showarrow=False)
            return fig