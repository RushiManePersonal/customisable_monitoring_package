# monitoring package init

import pandas as pd

if not hasattr(pd.read_csv, '_is_patched_for_source_path'):
    _original_read_csv = pd.read_csv
    def _read_csv_with_path(*args, **kwargs):
        df = _original_read_csv(*args, **kwargs)
        # Store the path as an attribute
        if args:
            df._source_path = args[0]
        elif 'filepath_or_buffer' in kwargs:
            df._source_path = kwargs['filepath_or_buffer']
        else:
            df._source_path = None
        return df
    _read_csv_with_path._is_patched_for_source_path = True
    pd.read_csv = _read_csv_with_path
