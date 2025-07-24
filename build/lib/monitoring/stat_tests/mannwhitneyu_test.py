# Mann-Whitney U test for data drift
from scipy.stats import mannwhitneyu

class MannWhitneyUTest:
    def __init__(self, params=None):
        self.params = params

    def run(self, reference_data, current_data):
        statistic, p_value = mannwhitneyu(reference_data, current_data, alternative='two-sided')
        return {'statistic': statistic, 'p_value': p_value}
