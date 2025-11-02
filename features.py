# features.py
import numpy as np
import pandas as pd

def window_to_features(samples):
    """
    Convert a list of samples (metrics over time) for a single PID into a 
    dictionary of statistical and trend features for ML analysis.
    """
    if not samples:
        return None
        
    df = pd.DataFrame(samples)
    
    # Define the core metrics to be analyzed
    cols = ['cpu_percent', 'memory_rss', 'memory_percent', 'read_bytes', 'write_bytes', 'threads', 'open_files']
    
    # Ensure all expected numeric columns exist, defaulting to 0 if a metric was missed
    for c in cols:
        if c not in df.columns:
            df[c] = 0

    features = {}
    for c in cols:
        vals = df[c].astype(float).values
        
        # Calculate statistical features over the window
        features[f'{c}_mean'] = float(np.mean(vals))
        features[f'{c}_std'] = float(np.std(vals))
        features[f'{c}_max'] = float(np.max(vals))
        features[f'{c}_min'] = float(np.min(vals))
        
        # Calculate the slope (trend) using linear regression
        x = np.arange(len(vals))
        if len(vals) > 1 and np.ptp(x) > 0:
            # np.polyfit(x, y, 1) returns the slope and intercept
            slope = np.polyfit(x, vals, 1)[0]
        else:
            slope = 0.0
        features[f'{c}_slope'] = float(slope)
        
    # Add count and immediate values as features
    features['sample_count'] = len(df)
    last = df.iloc[-1].to_dict() # Get the most recent sample
    features['last_cpu'] = float(last.get('cpu_percent', 0))
    features['last_mem_percent'] = float(last.get('memory_percent', 0))
    
    return features
