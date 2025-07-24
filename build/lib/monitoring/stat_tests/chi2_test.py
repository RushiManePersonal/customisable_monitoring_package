# monitoring/stat_tests/chi2_test.py

import pandas as pd
import numpy as np
from scipy.stats import chisquare
import plotly.graph_objects as go

def _calculate_chi2(reference_data: pd.Series, current_data: pd.Series):
    """
    Calculates the Chi-Squared statistic and p-value.
    
    Returns the statistic, p-value, and the frequency tables needed for plotting.
    """
    # Get all unique categories from both datasets to create a common index
    all_categories = pd.unique(np.concatenate([reference_data, current_data]))
    
    # Calculate observed frequencies in the current dataset
    f_obs = current_data.value_counts().reindex(all_categories, fill_value=0)
    
    # Calculate expected frequencies based on the reference distribution,
    # scaled to the size of the current dataset.
    ref_proportions = reference_data.value_counts(normalize=True).reindex(all_categories, fill_value=0)
    f_exp = ref_proportions * len(current_data)

    # Chi-squared test is not defined for an expected frequency of 0.
    # We replace 0s with a very small number to avoid errors, which has a negligible
    # impact on the final statistic for those categories.
    f_exp[f_exp == 0] = 1e-8
    
    statistic, p_value = chisquare(f_obs=f_obs, f_exp=f_exp)
    
    return statistic, p_value, f_obs, f_exp, all_categories


class Chi2Test:
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
        # Defensive: handle empty data
        if len(ref) == 0 or len(curr) == 0:
            fig = go.Figure()
            fig.add_annotation(text='No data for plot', x=0.5, y=0.5, showarrow=False)
            return fig
        keys = np.unique(np.concatenate([ref, curr]))
        import pandas as pd
        ref_counts = pd.Series(ref).value_counts(sort=False).reindex(keys, fill_value=0)
        curr_counts = pd.Series(curr).value_counts(sort=False).reindex(keys, fill_value=0)
        k_norm = curr.shape[0] / (ref.shape[0] if ref.shape[0] else 1)
        f_exp = ref_counts * k_norm
        f_obs = curr_counts
        fig = go.Figure()
        fig.add_bar(x=keys, y=f_obs, name="Current (Observed)", marker_color='red', opacity=0.5)
        fig.add_bar(x=keys, y=f_exp, name="Expected (Reference)", marker_color='blue', opacity=0.5)
        fig.update_layout(title_text="Chi-square Test Diagnostic Plot", barmode='overlay', height=400)
        fig.update_xaxes(title_text=str(column_name), tickangle=45)
        fig.update_yaxes(title_text="Counts")
        return fig
    def __init__(self, params=None):
        self.params = params

    def run(self, reference_data, current_data, threshold=0.05, **kwargs):
        """
        Runs the Chi-Squared test for categorical data.
        If data is numerical, it returns an error.
        """
        ref_series = pd.Series(reference_data)

        # ✅ Data type validation: Chi-squared is for categorical data.
        if pd.api.types.is_numeric_dtype(ref_series):
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'drift': False,
                'plot_data': {},
                'error': 'Chi-Squared test is for categorical data. Use KS or PSI for numerical.'
            }
        
        # Ensure data is treated as strings for consistency
        ref_series = ref_series.astype(str).dropna()
        curr_series = pd.Series(current_data).astype(str).dropna()

        if ref_series.empty or curr_series.empty:
            return {
                'statistic': 0, 'p_value': 1.0, 'drift': False,
                'plot_data': {'error': 'Not enough data to run test.'}
            }

        statistic, p_value, f_obs, f_exp, categories = _calculate_chi2(ref_series, curr_series)
        
        # Drift is detected if the p-value is below the threshold.
        drift_detected = p_value < threshold

        plot_data = {
            'column_name': ref_series.name or "Value",
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected,
            'threshold': threshold,
            'observed': f_obs.tolist(),
            'expected': f_exp.tolist(),
            'categories': categories.tolist()
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
        Generates an interactive bar chart comparing observed vs. expected frequencies.
        """
        try:
            if 'error' in plot_data:
                raise ValueError(plot_data['error'])
            if not plot_data or not plot_data.get('categories'):
                raise ValueError("Not enough data to create plot.")

            column_name = plot_data.get('column_name', 'Value')
            p_value = plot_data.get('p_value')
            statistic = plot_data.get('statistic')
            drift_status = "Yes" if plot_data.get('drift_detected') else "No"
            
            categories = plot_data.get('categories')
            observed = plot_data.get('observed')
            expected = plot_data.get('expected')

            fig = go.Figure(data=[
                go.Bar(name='Current (Observed)', x=categories, y=observed, marker_color='#d62728'),
                go.Bar(name='Reference (Expected)', x=categories, y=expected, marker_color='#1f77b4')
            ])
            
            p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"
            stat_str = f"{statistic:.4f}" if statistic is not None else "N/A"

            fig.update_layout(
                title_text=(
                    f"<b>Chi-Squared Test: '{column_name}'</b><br>"
                    f"P-value: {p_value_str} | Statistic: {stat_str} | Drift: {drift_status}"
                ),
                xaxis_title=f"Categories for {column_name}",
                yaxis_title="Frequency",
                barmode='group', # Grouped bars are easier to compare
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_xaxes(tickangle=-45)

            return fig

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f'Chi-Squared Plot generation error: {e}', x=0.5, y=0.5, showarrow=False)
            return fig