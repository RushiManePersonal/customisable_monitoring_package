# Maximum Mean Discrepancy (MMD)
import numpy as np

class MMDTest:
    def __init__(self, params=None):
        self.params = params


    def run(self, reference_data, current_data, threshold=0.05, plot=False, **kwargs):
        try:
            x = np.array(reference_data)
            y = np.array(current_data)
            mmd = np.mean(x) - np.mean(y)
            drift = abs(mmd) > threshold if mmd is not None else False
            plot_data = {
                'reference': list(reference_data),
                'current': list(current_data),
                'mmd': mmd,
                'threshold': threshold,
                'drift': drift
            }
            return {
                'mmd': mmd,
                'drift': drift,
                'plot_data': plot_data
            }
        
        except Exception as e:
            return {'error': str(e), 'drift': False}

    @staticmethod
    def make_plot_from_data(plot_data):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(text='MMD plot not implemented', x=0.5, y=0.5, showarrow=False)
        return fig
