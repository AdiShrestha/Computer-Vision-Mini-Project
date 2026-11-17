"""Milestone SR-1 Baselines and Universal Sequential CUSUM Wrapper.

Governed by plan.md SR-1.5 row 11-12 and Reviews R1/R2/R3:
- Water-Area Change Rule (physical ground-truth baseline).
- Seasonal Climatology baseline.
- Missingness-aware Isolation Forest.
- PSD RBF One-Class SVM.
- Classical Inexact ALM RPCA.
- Universal Page's CUSUM sequential wrapper applied across ALL detectors.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class UniversalCUSUM:
    """Page's Cumulative Sum (CUSUM) sequential change detector.
    
    Accumulator:
        S_t = max(0, S_{t-1} + z_t - kappa)
    Alarm condition:
        S_t >= h
    """
    def __init__(self, kappa: float = 0.5, h: float = 5.0):
        self.kappa = kappa
        self.h = h
        self.s = 0.0
        self.alarm_fired = False

    def reset(self):
        self.s = 0.0
        self.alarm_fired = False

    def update(self, z_score: float) -> Tuple[float, bool]:
        if not math.isfinite(z_score):
            return self.s, (self.s >= self.h)
        self.s = max(0.0, self.s + (z_score - self.kappa))
        self.alarm_fired = bool(self.s >= self.h)
        return self.s, self.alarm_fired


class WaterAreaBaseline:
    """Detects rapid drainage based on physical water surface area."""
    def __init__(self, area_drop_fraction: float = 0.30):
        self.area_drop_fraction = area_drop_fraction
        self.baseline_area_km2 = 0.0

    def fit(self, pre_areas_km2: List[float]):
        finite_areas = [a for a in pre_areas_km2 if math.isfinite(a) and a > 0.0]
        if finite_areas:
            self.baseline_area_km2 = float(np.median(finite_areas))
        else:
            self.baseline_area_km2 = 0.0

    def score(self, area_km2: float) -> float:
        """Normalized anomaly score: drops below baseline yield positive scores."""
        if self.baseline_area_km2 <= 0.0 or not math.isfinite(area_km2):
            return 0.0
        drop = (self.baseline_area_km2 - area_km2) / self.baseline_area_km2
        return float(max(0.0, drop))


class SeasonalClimatologyBaseline:
    """Predicts expected values from seasonal day-of-year window."""
    def __init__(self, window_days: int = 45):
        self.window_days = window_days
        self.records: List[Tuple[int, np.ndarray]] = []

    def fit(self, pre_scenes: List[Dict[str, Any]]):
        self.records = []
        for s in pre_scenes:
            doy = s["dt"].timetuple().tm_yday
            feats = s["features"]
            obs = s["observed"]
            if obs.any():
                mean_feat = np.nanmean(feats[obs], axis=0)
                self.records.append((doy, mean_feat))

    def score(self, scene: Dict[str, Any]) -> float:
        doy = scene["dt"].timetuple().tm_yday
        obs = scene["observed"]
        if not obs.any() or not self.records:
            return 0.0
        query_mean = np.nanmean(scene["features"][obs], axis=0)
        
        # Match historical DOYs within window_days
        dists = []
        for ref_doy, ref_mean in self.records:
            diff = abs(doy - ref_doy)
            circ_diff = min(diff, 365 - diff)
            if circ_diff <= self.window_days:
                dists.append(np.linalg.norm(query_mean - ref_mean))
        if not dists:
            # Fallback to all records
            dists = [np.linalg.norm(query_mean - ref_mean) for _, ref_mean in self.records]
        return float(np.median(dists)) if dists else 0.0


def fit_tabular_baselines(
    train_tokens: List[Dict[str, Any]],
    seed: int = 0,
) -> Tuple[IsolationForest, OneClassSVM]:
    """Fit Isolation Forest and One-Class SVM on pre-cutoff patch tokens."""
    np.random.seed(seed)
    
    # Collect flat feature vectors of observed tokens
    vectors = []
    for sc in train_tokens:
        obs = sc["observed"]
        feats = sc["std_features"][obs]
        for f in feats:
            if np.all(np.isfinite(f)):
                vectors.append(f)
                
    X = np.array(vectors, dtype=np.float32)
    if len(X) < 10:
        X = np.zeros((10, 5), dtype=np.float32)
        
    iso = IsolationForest(contamination=0.05, random_state=seed, n_estimators=100)
    iso.fit(X)
    
    # One-Class SVM with RBF kernel
    oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    oc_svm.fit(X)
    
    return iso, oc_svm


def score_tabular_scene(
    scene: Dict[str, Any],
    iso: IsolationForest,
    oc_svm: OneClassSVM,
) -> Dict[str, float]:
    obs = scene["observed"]
    if not obs.any():
        return {"iso_forest_score": 0.0, "oc_svm_score": 0.0}
        
    feats = scene["std_features"][obs]
    valid_rows = [f for f in feats if np.all(np.isfinite(f))]
    if not valid_rows:
        return {"iso_forest_score": 0.0, "oc_svm_score": 0.0}
        
    X = np.array(valid_rows, dtype=np.float32)
    
    # In sklearn: lower decision_function = more anomalous.
    # Anomaly score = -decision_function so higher = more anomalous.
    iso_scores = -iso.decision_function(X)
    oc_scores = -oc_svm.decision_function(X)
    
    return {
        "iso_forest_score": float(np.median(iso_scores)),
        "oc_svm_score": float(np.median(oc_scores)),
    }
