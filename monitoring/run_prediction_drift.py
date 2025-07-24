

from monitoring.prediction_drift_report import (
    load_and_prepare_predictions,
    compute_prediction_drift,
    generate_prediction_drift_report
)

def main():
    # Hardcoded file paths
    preds_a_csv = "C:/codes/trial2/pred1.csv"
    preds_b_csv = "C:/codes/trial2/pred2.csv"
    output_html = "C:/codes/trial2/output_report.html"

    # Load and auto-detect columns for both prediction files
    df_a, id_col_a, time_col_a, pred_col_a = load_and_prepare_predictions(preds_a_csv)
    df_b, id_col_b, time_col_b, pred_col_b = load_and_prepare_predictions(preds_b_csv)

    # Use the detected columns (prefer the ones found in both)
    id_col = id_col_a or id_col_b
    time_col = time_col_a or time_col_b
    pred_col = pred_col_a or pred_col_b

    # Compute prediction drift
    result = compute_prediction_drift(
        preds_a=df_a,
        preds_b=df_b,
        time_col=time_col,
        id_col=id_col,
        pred_col=pred_col,
        task_type='regression'  # or 'classification' as needed
    )

    # Generate HTML report
    generate_prediction_drift_report(result, output_html)
    print(f"Prediction drift report saved to {output_html}")

if __name__ == "__main__":
    main()
