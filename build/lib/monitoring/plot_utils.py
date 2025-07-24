

import os
import plotly.express as px
import plotly.io as pio
import time
from datetime import datetime

def log(msg):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{now}] [plot_utils] {msg}')

def save_column_distributions(reference_df, current_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}
    total_start = time.time()
    log('Starting save_column_distributions...')
    for col in reference_df.columns:
        if col == 'Unnamed: 0':
            log(f'Skipping column {col} (likely index column)')
            continue
        col_start = time.time()
        # Reference distribution (interactive HTML)
        if reference_df[col].dtype.kind in 'biufc':
            fig_ref = px.histogram(reference_df, x=col, nbins=30, title=f'Reference Distribution: {col}')
        else:
            ref_counts = reference_df[col].value_counts().reset_index()
            ref_counts.columns = [col, 'count']
            fig_ref = px.bar(ref_counts, x=col, y='count', title=f'Reference Distribution: {col}')
        ref_plot_path = os.path.join(output_dir, f'{col}_reference.html')
        fig_ref.write_html(ref_plot_path, full_html=True, include_plotlyjs='cdn')

        # Current distribution (interactive HTML)
        if current_df[col].dtype.kind in 'biufc':
            fig_cur = px.histogram(current_df, x=col, nbins=30, title=f'Current Distribution: {col}', color_discrete_sequence=['orange'])
        else:
            cur_counts = current_df[col].value_counts().reset_index()
            cur_counts.columns = [col, 'count']
            fig_cur = px.bar(cur_counts, x=col, y='count', title=f'Current Distribution: {col}', color_discrete_sequence=['orange'])
        cur_plot_path = os.path.join(output_dir, f'{col}_current.html')
        fig_cur.write_html(cur_plot_path, full_html=True, include_plotlyjs='cdn')

        plot_paths[col] = {'reference': ref_plot_path, 'current': cur_plot_path}
        log(f'Column {col} plotted in {time.time() - col_start:.3f} seconds')
    log(f'All columns plotted in {time.time() - total_start:.3f} seconds')
    return plot_paths
