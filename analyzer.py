# analyzer.py
import threading
import time
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from collections import deque
from features import window_to_features

RETRAIN_INTERVAL = 30  # seconds
MIN_TRAIN_SAMPLES = 5

class Analyzer:
    """
    Performs real-time anomaly detection using a continuously retrained Isolation Forest.
    """
    def __init__(self, retrain_interval=RETRAIN_INTERVAL):
        self.model = None
        self.retrain_interval = retrain_interval
        self.feature_names = None
        # Thread lock to protect shared resources (model, train_pool)
        self.lock = threading.Lock() 
        # Rolling pool of feature vectors for training. Limits memory to 1000 samples.
        self.train_pool = deque(maxlen=1000)
        self.pool_means = None  # To cache training pool mean
        self.pool_stds = None   # To cache training pool std dev 
        self.running = True
        self._start_retrainer()

    def _start_retrainer(self):
        """Starts the background thread for continuous model retraining."""
        t = threading.Thread(target=self._retrain_loop, daemon=True)
        t.start()

    def _retrain_loop(self):
        """The loop that triggers model training at intervals."""
        while self.running:
            self._train_if_possible()
            time.sleep(self.retrain_interval)

    def _train_if_possible(self):
        """Trains a new Isolation Forest model if enough data is available."""
        with self.lock: # Acquire lock before accessing self.train_pool and self.model
            if len(self.train_pool) < MIN_TRAIN_SAMPLES:
                return
            X = np.array(self.train_pool)
            # IsolationForest is an unsupervised anomaly detection model
            model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
            try:
                model.fit(X)
                self.model = model
# Cache pool statistics for faster explanations
                self.pool_means = X.mean(axis=0)
                self.pool_stds = X.std(axis=0) + 1e-6 # Add epsilon to avoid divide by zero # Update the model reference after successful fit
            except Exception as e:
                print("Training failed:", e)

    def compute_features_from_store(self, store_snapshot):
        """
        Converts raw process data into a feature matrix (X) and populates the training pool.
        """
        rows = []
        metadata = []
        for pid, samples in store_snapshot.items():
            feat = window_to_features(samples)
            if feat:
                # Features are extracted in a consistent, sorted order
                rows.append([feat[k] for k in sorted(feat.keys())])
                metadata.append({'pid': pid, 'name': samples[-1].get('name'), 'last_sample': samples[-1]})
                if self.feature_names is None:
                    self.feature_names = sorted(feat.keys())
                # Add feature vector to train pool under lock
                with self.lock:
                    self.train_pool.append([feat[k] for k in self.feature_names]) 
        if rows:
            X = np.array(rows)
        else:
            X = np.empty((0, len(self.feature_names or [])))
        return X, metadata

    def score(self, feature_vector):
        """
        Converts the raw Isolation Forest decision score (higher = normal) 
        to a 0-1 anomaly score (1 = high anomaly).
        """
        if self.model is None:
            return 0.0
        try:
            raw = self.model.decision_function([feature_vector])[0]
            # Sigmoid-like conversion to map raw score to a 0..1 range
            score = float(1.0 / (1.0 + np.exp(5 * raw))) 
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            return 0.0

    def analyze_snapshot(self, store_snapshot, top_n=10):
        """
        Scores all process features, sorts by anomaly score, and returns top N results.
        """
        X, metadata = self.compute_features_from_store(store_snapshot)
        results = []
        if X.size == 0:
            return results
            
        for i, meta in enumerate(metadata):
            vector = X[i]
            s = self.score(vector.tolist())
            explanation = self.simple_explain(vector)
            r = {'pid': meta['pid'], 'name': meta.get('name'), 'score': s, 'explanation': explanation, 'last_sample': meta['last_sample']}
            results.append(r)
            
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]

    def simple_explain(self, vector):
        """
        Identifies the two features with the largest absolute Z-score deviation 
        from the training pool's mean.
        """
        if self.feature_names is None or self.pool_means is None:
            return "Awaiting model training for explanation"

# Use cached statistics for massive performance gain
        import numpy as np
        means = self.pool_means
        stds = self.pool_stds
        z = (np.array(vector) - means) / stds
        
        # Get indices of the top 2 features with the largest absolute deviation
        idx = np.argsort(-np.abs(z))[:2] 
        
        feats = []
        for i in idx:
            feats.append((self.feature_names[i], float(z[i])))
        return feats
