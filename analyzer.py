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
    def __init__(self, retrain_interval=RETRAIN_INTERVAL):
        self.model = None
        self.retrain_interval = retrain_interval
        self.feature_names = None
        self.lock = threading.Lock()
        # Keep a rolling pool of feature vectors for training
        self.train_pool = deque(maxlen=1000)
        self.running = True
        self._start_retrainer()

    def _start_retrainer(self):
        t = threading.Thread(target=self._retrain_loop, daemon=True)
        t.start()

    def _retrain_loop(self):
        while self.running:
            self._train_if_possible()
            time.sleep(self.retrain_interval)

    def _train_if_possible(self):
        with self.lock:
            if len(self.train_pool) < MIN_TRAIN_SAMPLES:
                return
            X = np.array(self.train_pool)
            # fit new model
            model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
            try:
                model.fit(X)
                self.model = model
                # store feature names only if set
            except Exception as e:
                print("Training failed:", e)

    def compute_features_from_store(self, store_snapshot):
        """
        Convert entire store snapshot into feature matrix and metadata.
        """
        rows = []
        metadata = []
        for pid, samples in store_snapshot.items():
            feat = window_to_features(samples)
            if feat:
                rows.append([feat[k] for k in sorted(feat.keys())])
                metadata.append({'pid': pid, 'name': samples[-1].get('name'), 'last_sample': samples[-1]})
                if self.feature_names is None:
                    self.feature_names = sorted(feat.keys())
                # add to train pool
                with self.lock:
                    self.train_pool.append([feat[k] for k in self.feature_names])
        if rows:
            X = np.array(rows)
        else:
            X = np.empty((0, len(self.feature_names or [])))
        return X, metadata

    def score(self, feature_vector):
        """
        Return anomaly score between 0 and 1 where 1 is most anomalous.
        IsolationForest decision_function returns score; lower means anomalous; we convert.
        """
        if self.model is None:
            return 0.0
        try:
            raw = self.model.decision_function([feature_vector])[0]  # higher = normal
            # convert to 0..1 anomaly score (1 high anomaly)
            # decision_function ranges approx [-0.5, 0.5] depending on parameters; we invert and scale.
            # Use sigmoid-like conversion
            score = float(1.0 / (1.0 + np.exp(5 * raw)))
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            return 0.0

    def analyze_snapshot(self, store_snapshot, top_n=10):
        """
        Analyze current store snapshot; return list of dicts with pid, name, score, explanation.
        """
        X, metadata = self.compute_features_from_store(store_snapshot)
        results = []
        if X.size == 0:
            return results
        # compute scores
        for i, meta in enumerate(metadata):
            vector = X[i] if self.feature_names is not None else X[i]
            s = self.score(vector.tolist())
            # top 2 features with largest deviation from pool mean
            explanation = self.simple_explain(vector)
            r = {'pid': meta['pid'], 'name': meta.get('name'), 'score': s, 'explanation': explanation, 'last_sample': meta['last_sample']}
            results.append(r)
        # sort by score desc
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]

    def simple_explain(self, vector):
        # find features with largest z-score relative to training pool
        if self.feature_names is None or len(self.train_pool) < 2:
            return "Insufficient data for explanation"
        import numpy as np
        pool = np.array(self.train_pool)
        means = pool.mean(axis=0)
        stds = pool.std(axis=0) + 1e-6
        z = (np.array(vector) - means) / stds
        idx = np.argsort(-np.abs(z))[:2]
        feats = []
        for i in idx:
            feats.append((self.feature_names[i], float(z[i])))
        return feats
