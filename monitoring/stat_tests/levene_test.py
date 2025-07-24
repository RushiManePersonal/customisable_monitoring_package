# Levene's test for equal variances
from scipy.stats import levene

class LeveneTest:
    def __init__(self, params=None):
        self.params = params


    def run(self, reference_data, current_data, threshold=0.05, plot=False, **kwargs):
        try:
            statistic, p_value = levene(reference_data, current_data)
            drift = p_value < threshold if p_value is not None else False
            plot_data = {
                'reference': list(reference_data),
                'current': list(current_data),
                'statistic': statistic,
                'p_value': p_value,
                'threshold': threshold,
                'drift': drift
            }
            return {
                'statistic': statistic,
                'p_value': p_value,
                'drift': drift,
                'plot_data': plot_data
            }
        
        except Exception as e:
            return {'error': str(e), 'drift': False}

    @staticmethod
    def make_plot_from_data(plot_data):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(text='Levene plot not implemented', x=0.5, y=0.5, showarrow=False)
        return fig
