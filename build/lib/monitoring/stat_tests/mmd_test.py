# Maximum Mean Discrepancy (MMD) for data drift
import numpy as np

class MMDTest:
    def __init__(self, params=None):
        self.params = params

    def run(self, reference_data, current_data):
        x = np.array(reference_data)
        y = np.array(current_data)
        mmd = np.mean(x) - np.mean(y)
        return {'mmd': mmd}
