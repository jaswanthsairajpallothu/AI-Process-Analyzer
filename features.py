# features.py
import numpy as np
import pandas as pd

def window_to_features(samples):
    """
    Convert a list of samples (for one PID) into features for ML.
    Samples are dicts with keys: cpu_percent, memory_rss, memory_percent, read_bytes, write_bytes, threads
    Returns a dict of engineered features.
    """
    if not samples:
        return None
    df = pd.DataFrame(samples)
    # ensure numeric columns exist
    cols = ['cpu_percent', 'memory_rss', 'memory_percent', 'read_bytes', 'write_bytes', 'threads', 'open_files']
    for c in cols:
        if c not in df.columns:
            df[c] = 0

    features = {}
    for c in cols:
        vals = df[c].astype(float).values
        features[f'{c}_mean'] = float(np.mean(vals))
        features[f'{c}_std'] = float(np.std(vals))
        features[f'{c}_max'] = float(np.max(vals))
        features[f'{c}_min'] = float(np.min(vals))
        # slope (simple linear regression slope)
        x = np.arange(len(vals))
        if len(vals) > 1 and np.ptp(x) > 0:
            slope = np.polyfit(x, vals, 1)[0]
        else:
            slope = 0.0
        features[f'{c}_slope'] = float(slope)
    # sample count
    features['sample_count'] = len(df)
    # last observed metrics for immediacy
    last = df.iloc[-1].to_dict()
    features['last_cpu'] = float(last.get('cpu_percent', 0))
    features['last_mem_percent'] = float(last.get('memory_percent', 0))
    return features
