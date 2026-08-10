"""
Synthetic Anomaly Injection Engine (INV-011).

Injects four physically motivated synthetic anomaly types into control lake
time series for Protocol E3 evaluation.

Invariants:
    INV-011: Synthetic anomaly definitions & 10 injections per type per lake.
    INV-012: Deterministic random seed = 2023.
"""

import os
import sys
import json
import random
import numpy as np
from typing import Dict, Any, List, Tuple

source_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anomaly_config.json')


class SyntheticInjector:
    """Synthetic Anomaly Injector per INV-011 / Decision 003."""
    
    def __init__(self, seed: int = 2023, config_path: str = CONFIG_PATH):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {
                "anomaly_types": [
                    {"id": 1, "name": "sudden_extent", "channel": "CH-01", "column_idx": 0, "magnitude": 0.20, "duration_windows": 1},
                    {"id": 2, "name": "gradual_extent", "channel": "CH-01", "column_idx": 0, "magnitude": 0.15, "duration_windows": 3},
                    {"id": 3, "name": "sar_backscatter", "channel": "CH-05", "column_idx": 8, "magnitude": 3.0, "duration_windows": 6},
                    {"id": 4, "name": "temperature_spike", "channel": "CH-04", "column_idx": 7, "magnitude": 5.0, "duration_windows": 1}
                ]
            }

    def inject(self, features: np.ndarray, anomaly_type: int, window_idx: int, channel_idx: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Inject a synthetic anomaly into a copy of the feature matrix.
        
        Args:
            features: (T, C) feature matrix
            anomaly_type: 1, 2, 3, or 4
            window_idx: window index where injection begins
            channel_idx: column index in feature matrix
            
        Returns:
            (modified_features, metadata)
        """
        mod_features = features.copy()
        T, C = mod_features.shape
        
        type_cfg = next((t for t in self.config['anomaly_types'] if t['id'] == anomaly_type), None)
        if type_cfg is None:
            mag = 0.20
            dur = 1
            col_idx = channel_idx
            name = f"type_{anomaly_type}"
        else:
            mag = type_cfg['magnitude']
            dur = type_cfg['duration_windows']
            col_idx = type_cfg['column_idx'] if type_cfg['column_idx'] < C else channel_idx
            name = type_cfg['name']

        start = max(0, min(window_idx, T - 1))
        end = min(start + dur, T)

        if anomaly_type == 1:
            # Sudden extent step change (+20%)
            mod_features[start:end, col_idx] *= (1.0 + mag)
        elif anomaly_type == 2:
            # Gradual linear ramp (+15% over dur windows)
            ramp_len = end - start
            ramp = np.linspace(0.0, mag, ramp_len)
            for idx, r in enumerate(ramp):
                mod_features[start + idx, col_idx] *= (1.0 + r)
        elif anomaly_type == 3:
            # SAR backscatter step change (+3 dB)
            mod_features[start:end, col_idx] += mag
        elif anomaly_type == 4:
            # Temperature spike (+5.0 °C)
            mod_features[start:end, col_idx] += mag

        metadata = {
            "anomaly_type": anomaly_type,
            "name": name,
            "window_idx": start,
            "duration_windows": dur,
            "channel_idx": col_idx,
            "magnitude": mag
        }

        return mod_features, metadata

    def generate_injections(self, features: np.ndarray, lake_id: str, n_injections: int = 10) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Generate all 4 anomaly types × n_injections per control lake.
        
        Args:
            features: (T, C) feature matrix
            lake_id: String lake ID
            n_injections: Number of injections per type (default 10 per INV-011)
            
        Returns:
            List of (modified_features, metadata) tuples
        """
        # Seed generator deterministically per lake for reproducibility (INV-012)
        lake_hash = sum(ord(c) for c in lake_id)
        local_rng = random.Random(self.seed + lake_hash)
        
        T, C = features.shape
        injections = []

        for type_cfg in self.config['anomaly_types']:
            a_type = type_cfg['id']
            dur = type_cfg['duration_windows']
            valid_max_start = max(1, T - dur - 1)
            
            # Select n_injections random window indices
            window_indices = [local_rng.randint(5, valid_max_start) for _ in range(n_injections)]
            
            for w_idx in window_indices:
                mod_feat, meta = self.inject(features, a_type, w_idx, type_cfg['column_idx'])
                meta['lake_id'] = lake_id
                injections.append((mod_feat, meta))

        return injections
