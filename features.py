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
    # Define which columns are standard metrics and which are cumulative counters
std_cols = ['cpu_percent', 'memory_rss', 'memory_percent', 'threads', 'open_files']
io_cols = ['read_bytes', 'write_bytes']

for c in std_cols + io_cols:
    if c not in df.columns:
        df[c] = 0

features = {}

# Process standard columns normally
for c in std_cols:
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

# Process I/O columns as a *rate* (bytes per second)
if 'timestamp' in df.columns and len(df) > 1:
    # Get time elapsed in seconds, ensure at least 1.0 to avoid ZeroDivisionError
    time_elapsed = max(1.0, df['timestamp'].iloc[-1] - df['timestamp'].iloc[0])

    for c in io_cols:
        vals = df[c].astype(float).values
        # Rate = (last value - first value) / time elapsed
        rate = (vals[-1] - vals[0]) / time_elapsed
        features[f'{c}_rate'] = max(0.0, float(rate)) # Ensure rate is non-negative
else:
    # Not enough data to calculate a rate
    for c in io_cols:
        features[f'{c}_rate'] = 0.0
        
    # Add count and immediate values as features
    features['sample_count'] = len(df)
    last = df.iloc[-1].to_dict() # Get the most recent sample
    features['last_cpu'] = float(last.get('cpu_percent', 0))
    features['last_mem_percent'] = float(last.get('memory_percent', 0))
    
    return features
