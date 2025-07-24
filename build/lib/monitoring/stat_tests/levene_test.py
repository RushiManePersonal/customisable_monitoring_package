# Levene's test for equal variances
from scipy.stats import levene

class LeveneTest:
    def __init__(self, params=None):
        self.params = params

    def run(self, reference_data, current_data):
        statistic, p_value = levene(reference_data, current_data)
        return {'statistic': statistic, 'p_value': p_value}
