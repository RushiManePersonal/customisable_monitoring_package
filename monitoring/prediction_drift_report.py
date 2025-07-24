import os
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
from .prediction_drift import PredictionDriftResult, compute_prediction_drift
import json
import pandas as pd

def _auto_detect_column(df: pd.DataFrame, candidates):
    """Return the first column in df that matches any of the candidates (case-insensitive)."""
    for cand in candidates:
        for col in df.columns:
            if col.lower() == cand.lower():
                return col
    return None

def _auto_detect_date_column(df: pd.DataFrame):
    # Try to find a column with 'date', 'time', or 'period' in the name
    for col in df.columns:
        if any(x in col.lower() for x in ['date', 'time', 'period', 'month', 'year', 'timestamp', 'datetime', 'week', 'day']):
            return col
    return None

def _parse_date_column(df: pd.DataFrame, date_col: str):
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    except Exception:
        pass
    return df

def load_and_prepare_predictions(csv_path, id_col=None, time_col=None, pred_col=None):
    df = pd.read_csv(csv_path)
    # Auto-detect columns if not provided
    if not time_col:
        time_col = _auto_detect_date_column(df)
    if not pred_col:
        pred_col = _auto_detect_column(df, ['prediction', 'pred', 'score', 'output'])
    if not id_col:
        id_col = _auto_detect_column(df, ['id', 'row_id', 'index'])
    # Parse date column if found
    if time_col:
        df = _parse_date_column(df, time_col)
    return df, id_col, time_col, pred_col

def generate_prediction_drift_report(result: PredictionDriftResult, output_path: str, output_html: bool = True):
    # Fix per-period MAPE calculation for the template
    # This will update result.drift_metrics[period]['mape'] for each period
    if hasattr(result, 'overlap_periods') and hasattr(result, 'details'):
        for period in result.overlap_periods:
            details = result.details.get(period, {})
            prev = details.get('previous', [])
            curr = details.get('current', [])
            min_len = min(len(prev), len(curr))
            if min_len > 0:
                arr_prev = pd.Series(prev[:min_len])
                arr_curr = pd.Series(curr[:min_len])
                mask = arr_prev != 0
                if mask.any():
                    mape = float((abs(arr_curr[mask] - arr_prev[mask]) / abs(arr_prev[mask])).mean() * 100)
                else:
                    mape = 0.0
                if period in result.drift_metrics:
                    result.drift_metrics[period]['mape'] = mape
    elif isinstance(result, dict) and 'overlap_periods' in result and 'details' in result:
        for period in result['overlap_periods']:
            details = result['details'].get(period, {})
            prev = details.get('previous', [])
            curr = details.get('current', [])
            min_len = min(len(prev), len(curr))
            if min_len > 0:
                arr_prev = pd.Series(prev[:min_len])
                arr_curr = pd.Series(curr[:min_len])
                mask = arr_prev != 0
                if mask.any():
                    mape = float((abs(arr_curr[mask] - arr_prev[mask]) / abs(arr_prev[mask])).mean() * 100)
                else:
                    mape = 0.0
                if period in result['drift_metrics']:
                    result['drift_metrics'][period]['mape'] = mape
    # Compute overall MAE, RMSE, and MAPE using all predictions
    all_previous = []
    all_current = []
    for period in getattr(result, 'overlap_periods', getattr(result, 'overlap_periods', [])):
        details = result.details if hasattr(result, 'details') else result.get('details', {})
        if period in details:
            prev = details[period].get('previous', [])
            curr = details[period].get('current', [])
            min_len = min(len(prev), len(curr))
            if min_len > 0:
                all_previous.extend(prev[:min_len])
                all_current.extend(curr[:min_len])
    overall_mae = None
    overall_rmse = None
    overall_mape = None
    if all_previous and all_current:
        import numpy as np
        arr_prev = np.array(all_previous)
        arr_curr = np.array(all_current)
        overall_mae = float(np.mean(np.abs(arr_prev - arr_curr)))
        overall_rmse = float(np.sqrt(np.mean((arr_prev - arr_curr) ** 2)))
        # MAPE: mean absolute percentage error, avoid division by zero
        mask = arr_prev != 0
        if np.any(mask):
            overall_mape = float(np.mean(np.abs(arr_curr[mask] - arr_prev[mask]) / np.abs(arr_prev[mask])) * 100)
        else:
            overall_mape = 0.0
    # Attach to result for template
    if isinstance(result, dict):
        result['overall_mae'] = overall_mae
        result['overall_rmse'] = overall_rmse
        result['overall_mape'] = overall_mape
    else:
        result.overall_mae = overall_mae
        result.overall_rmse = overall_rmse
        result.overall_mape = overall_mape
    """
    Render the prediction drift report as HTML and/or JSON.
    Args:
        result: PredictionDriftResult object
        output_path: Path to save the HTML report (if output_html=True)
        output_html: Whether to generate HTML report (default True)
    """
    # Always save JSON report
    json_path = output_path + ".json"
    # Convert to dict for JSON serialization
    if hasattr(result, '__dict__'):
        result_dict = result.__dict__.copy()
    else:
        result_dict = dict(result)
    # Convert any numpy types to native Python and ensure all dict keys are strings
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
    result_dict = convert(result_dict)
    with open(json_path, 'w', encoding='utf-8') as f_json:
        json.dump(result_dict, f_json, indent=2, default=str)
    # Generate HTML report only if requested
    if output_html:
        env = Environment(
            loader=FileSystemLoader(os.path.dirname(__file__)),
            autoescape=True
        )
        env.filters['tojson'] = lambda v: json.dumps(v)
        # Convert all periods to strings for JSON serialization
        if hasattr(result, 'overlap_periods'):
            result.overlap_periods = [str(p) for p in result.overlap_periods]
            # Convert keys in drift_metrics and details to strings
            if hasattr(result, 'drift_metrics'):
                result.drift_metrics = {str(k): v for k, v in result.drift_metrics.items()}
            if hasattr(result, 'details'):
                result.details = {str(k): v for k, v in result.details.items()}
        # Ensure combined_pred_a and combined_pred_b are present for the template
        if isinstance(result, dict):
            details = result.get('details', {})
            overlap_periods = result.get('overlap_periods', [])
            result['combined_pred_a'] = [details[p]['previous'][0] if p in details and details[p].get('previous') else None for p in overlap_periods]
            result['combined_pred_b'] = [details[p]['current'][0] if p in details and details[p].get('current') else None for p in overlap_periods]
        elif hasattr(result, 'details') and hasattr(result, 'overlap_periods'):
            details = result.details
            overlap_periods = result.overlap_periods
            result.combined_pred_a = [details[p]['previous'][0] if p in details and details[p].get('previous') else None for p in overlap_periods]
            result.combined_pred_b = [details[p]['current'][0] if p in details and details[p].get('current') else None for p in overlap_periods]
        template = env.get_template('prediction_drift_report_template.html')
        # Before rendering, ensure overall metrics are not None (set to 0.0 if None)
        if isinstance(result, dict):
            if result.get('overall_mae') is None:
                result['overall_mae'] = 0.0
            if result.get('overall_rmse') is None:
                result['overall_rmse'] = 0.0
            if result.get('overall_mape') is None:
                result['overall_mape'] = 0.0
        else:
            if getattr(result, 'overall_mae', None) is None:
                result.overall_mae = 0.0
            if getattr(result, 'overall_rmse', None) is None:
                result.overall_rmse = 0.0
            if getattr(result, 'overall_mape', None) is None:
                result.overall_mape = 0.0
        html = template.render(
            result=result,
            generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
