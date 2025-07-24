import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class PredictionDriftResult:
    def __init__(self, 
                 overlap_periods: List[Any], 
                 drift_metrics: Dict[str, Dict[str, float]], 
                 details: Dict[str, Any],
                 mean_previous: Optional[List[float]] = None,
                 mean_current: Optional[List[float]] = None):
        self.overlap_periods = overlap_periods
        self.drift_metrics = drift_metrics  # {period: {metric: value}}
        self.details = details  # Any additional info
        self.mean_previous = mean_previous or []
        self.mean_current = mean_current or []


def compute_prediction_drift(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    time_col: str = 'period',
    id_col: Optional[str] = None,
    pred_col: str = 'prediction',
    task_type: str = 'regression',
    metrics: Optional[List[str]] = None
) -> PredictionDriftResult:
    """
    Compare predictions from two models for overlapping periods.
    previous, current: DataFrames with columns [time_col, id_col (optional), pred_col]
    task_type: 'regression' or 'classification'
    metrics: list of metrics to compute
    """
    if metrics is None:
        if task_type == 'regression':
            metrics = ['mae', 'rmse', 'corr']
        else:
            metrics = ['class_change_rate', 'agreement']

    # Find overlapping periods
    periods_previous = set(previous[time_col])
    periods_current = set(current[time_col])
    overlap = sorted(list(periods_previous & periods_current))
    
    drift_metrics = {}
    details = {}
    all_mean_previous = []
    all_mean_current = []

    for period in overlap:
        if id_col:
            prev = previous[(previous[time_col] == period)][[id_col, pred_col]].set_index(id_col)
            curr = current[(current[time_col] == period)][[id_col, pred_col]].set_index(id_col)
            merged = prev.join(curr, lsuffix='_previous', rsuffix='_current', how='inner')
            pred_previous = merged[f'{pred_col}_previous']
            pred_current = merged[f'{pred_col}_current']
        else:
            pred_previous = previous[previous[time_col] == period][pred_col].reset_index(drop=True)
            pred_current = current[current[time_col] == period][pred_col].reset_index(drop=True)
            min_len = min(len(pred_previous), len(pred_current))
            pred_previous = pred_previous[:min_len]
            pred_current = pred_current[:min_len]

        period_metrics = {}
        if task_type == 'regression':
            mean_prev = float(np.mean(pred_previous))
            mean_curr = float(np.mean(pred_current))
            all_mean_previous.append(mean_prev)
            all_mean_current.append(mean_curr)

            period_metrics['mae'] = float(np.mean(np.abs(pred_previous - pred_current)))
            period_metrics['rmse'] = float(np.sqrt(np.mean((pred_previous - pred_current) ** 2)))
            if len(pred_previous) > 1 and len(pred_current) > 1:
                period_metrics['corr'] = float(np.corrcoef(pred_previous, pred_current)[0, 1])
            else:
                period_metrics['corr'] = np.nan
        else: # classification
            period_metrics['class_change_rate'] = float(np.mean(pred_previous != pred_current))
            period_metrics['agreement'] = float(np.mean(pred_previous == pred_current))
        
        drift_metrics[period] = period_metrics
        details[period] = {
            'n_samples': len(pred_previous),
            'previous': pred_previous.values.tolist(),
            'current': pred_current.values.tolist(),
        }

    return PredictionDriftResult(
        overlap_periods=overlap, 
        drift_metrics=drift_metrics, 
        details=details,
        mean_previous=all_mean_previous,
        mean_current=all_mean_current
    )