"""Analytical fixture baseline operators; research fitting and panel scoring remain blocked."""
from __future__ import annotations
import dataclasses
from datetime import date
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .features import CHANNEL_MODALITIES, MultiModalPanel


# ---------------------------------------------------------------------------
# 1. Seasonal Climatology Baseline
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ClimatologyBaseline:
    """Seasonal climatology baseline tracking day-of-year means and standard deviations."""
    training_lake_ids: Tuple[str, ...]
    temporal_cutoff: Optional[str]
    global_means: Tuple[float, ...]
    global_scales: Tuple[float, ...]
    channel_names: Tuple[str, ...]
    channel_schema_hash: str
    doy_means: np.ndarray
    doy_scales: np.ndarray
    state_hash: str

    @classmethod
    def fit_fixture(
        cls,
        panels: Sequence[MultiModalPanel],
        training_lake_ids: Sequence[str],
        temporal_cutoff: Optional[str] = None,
        eps: float = 1e-7,
    ) -> ClimatologyBaseline:
        """Fit climatology parameters strictly on training split panels."""
        if any(p.origin != "fixture" for p in panels):
            raise ValueError("baseline fixtures require explicit fixture origin")
        train_set = set(training_lake_ids)
        if not train_set:
            raise ValueError("training_lake_ids must be non-empty")

        valid_panels = [
            p for p in panels
            if p.lake_id in train_set and (temporal_cutoff is None or p.end_date <= temporal_cutoff)
        ]
        if not valid_panels:
            raise ValueError("No valid training panels available to fit ClimatologyBaseline")
        channel_names = valid_panels[0].channel_names
        schema_hash = valid_panels[0].channel_schema_hash
        if any(p.channel_names != channel_names or p.channel_schema_hash != schema_hash for p in valid_panels):
            raise ValueError("climatology fit panels have different channel schemas")
        n_channels = len(channel_names)

        # 1. Compute global training channel statistics
        global_means = []
        global_scales = []
        for c in range(n_channels):
            obs_vals = []
            for p in valid_panels:
                vals = p.values[:, c][p.mask[:, c]]
                if vals.size > 0:
                    obs_vals.append(vals)
            if obs_vals:
                stacked = np.concatenate(obs_vals)
                mu = float(np.mean(stacked))
                sigma = float(np.std(stacked))
                if not np.isfinite(sigma):
                    raise ValueError("nonfinite fitted scale")
                if sigma == 0.0:
                    sigma = 1.0
            else:
                raise ValueError("missing training channel")
            global_means.append(mu)
            global_scales.append(sigma)

        # 2. Compute DOY bins (1 to 366)
        doy_means = np.zeros((367, n_channels), dtype=np.float64)
        doy_scales = np.zeros((367, n_channels), dtype=np.float64)

        # Pre-initialize with global fallback
        for d in range(367):
            doy_means[d, :] = global_means
            doy_scales[d, :] = global_scales

        # Collect observations per DOY
        # Map: (doy, channel) -> list of values
        doy_obs: Dict[Tuple[int, int], List[float]] = {}
        for p in valid_panels:
            for t_idx, d_str in enumerate(p.dates):
                d_obj = date.fromisoformat(d_str.split("T")[0])
                doy = d_obj.timetuple().tm_yday
                for c in range(n_channels):
                    if p.mask[t_idx, c]:
                        key = (doy, c)
                        if key not in doy_obs:
                            doy_obs[key] = []
                        doy_obs[key].append(float(p.values[t_idx, c]))

        for (doy, c), vals in doy_obs.items():
            if vals:
                arr = np.array(vals, dtype=np.float64)
                mu_doy = float(np.mean(arr))
                sigma_doy = float(np.std(arr))
                doy_means[doy, c] = mu_doy
                if sigma_doy > 0.0 and np.isfinite(sigma_doy):
                    doy_scales[doy, c] = sigma_doy
                else:
                    doy_scales[doy, c] = global_scales[c]

        # Cryptographic state hash
        hash_payload = {
            "training_lake_ids": sorted(list(train_set)),
            "temporal_cutoff": temporal_cutoff,
            "channel_names": list(channel_names),
            "channel_schema_hash": schema_hash,
            "global_means": list(global_means),
            "global_scales": list(global_scales),
            "doy_means_digest": hashlib.sha256(doy_means.tobytes()).hexdigest(),
            "doy_scales_digest": hashlib.sha256(doy_scales.tobytes()).hexdigest(),
        }
        state_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode("utf-8")).hexdigest()

        return cls(
            training_lake_ids=tuple(sorted(train_set)),
            temporal_cutoff=temporal_cutoff,
            global_means=tuple(global_means),
            global_scales=tuple(global_scales),
            channel_names=channel_names,
            channel_schema_hash=schema_hash,
            doy_means=doy_means,
            doy_scales=doy_scales,
            state_hash=state_hash,
        )

    def predict_window(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        dates: Sequence[str],
        eps: float = 1e-7,
    ) -> float:
        """Compute window anomaly score via observed seasonal climatology deviation."""
        if values.shape[0] != len(dates):
            raise ValueError(f"values rows {values.shape[0]} does not match dates count {len(dates)}")
        if values.ndim != 2 or values.shape[1] != len(self.channel_names) or mask.shape != values.shape:
            raise ValueError("climatology input shape differs from its fitted channel schema")

        doys = np.array([date.fromisoformat(d.split("T")[0]).timetuple().tm_yday for d in dates], dtype=np.int32)
        mu_window = self.doy_means[doys, :]   # (T, C)
        sig_window = self.doy_scales[doys, :] # (T, C)

        valid_entries = mask & np.isfinite(values)
        total_obs = int(np.sum(valid_entries))
        if total_obs == 0:
            return None

        v = values[valid_entries]
        mu = mu_window[valid_entries]
        sig = sig_window[valid_entries]
        diff = np.abs(v - mu) / (sig + eps)
        score = float(np.sum(diff) / (total_obs + eps))
        return score

    def predict_panel(self, panel: MultiModalPanel) -> float:
        if panel.origin != "fixture":
            from .admission import blocked
            blocked("research baseline scoring")
        if panel.channel_names != self.channel_names or panel.channel_schema_hash != self.channel_schema_hash:
            raise ValueError("panel channel schema differs from fitted climatology")
        return self.predict_window(panel.values, panel.mask, panel.dates)

    def predict_status(self, panel):
        value = self.predict_panel(panel)
        return {'value': value, 'status': 'ESTIMATED' if value is not None else 'NOT_ESTIMABLE',
                'reason': None if value is not None else 'INSUFFICIENT_MEASUREMENT_OR_FITTED_SUPPORT',
                'origin': 'fixture', 'research_admissible': False}

    @classmethod
    def fit(cls, *args, **kwargs):
        from .admission import blocked
        blocked("research baseline fitting")


# ---------------------------------------------------------------------------
# 2. Lake-Area Trend Heuristic
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class LakeAreaTrendBaseline:
    """Lake-area expansion heuristic monitoring optical lake area trend."""
    training_lake_ids: Tuple[str, ...]
    temporal_cutoff: Optional[str]
    min_separation_days: int
    state_hash: str

    @classmethod
    def fit_fixture(
        cls,
        panels: Sequence[MultiModalPanel],
        training_lake_ids: Sequence[str],
        temporal_cutoff: Optional[str] = None,
        min_separation_days: int = 30,
    ) -> LakeAreaTrendBaseline:
        """Initialize and seal area trend baseline with training isolation provenance."""
        if any(p.origin != "fixture" for p in panels):
            raise ValueError("baseline fixtures require explicit fixture origin")
        train_set = set(training_lake_ids)
        if not train_set:
            raise ValueError("training_lake_ids must be non-empty")

        hash_payload = {
            "training_lake_ids": sorted(list(train_set)),
            "temporal_cutoff": temporal_cutoff,
            "min_separation_days": min_separation_days,
            "channel": "optical_candidate_roi_area_km2",
        }
        state_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode("utf-8")).hexdigest()

        return cls(
            training_lake_ids=tuple(sorted(train_set)),
            temporal_cutoff=temporal_cutoff,
            min_separation_days=min_separation_days,
            state_hash=state_hash,
        )

    def predict_window(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        dates: Optional[Sequence[str]] = None,
        channel_names: Optional[Sequence[str]] = None,
    ) -> float:
        """Compute relative lake area expansion Delta A / A from earliest to latest valid optical observation."""
        try:
            if channel_names is None:
                return None
            area_index = tuple(channel_names).index("optical_candidate_roi_area_km2")
        except ValueError:
            return None
        area_mask = mask[:, area_index] & np.isfinite(values[:, area_index])
        valid_indices = np.flatnonzero(area_mask)
        if len(valid_indices) < 2:
            return None

        idx_earliest = int(valid_indices[0])
        idx_latest = int(valid_indices[-1])

        if dates is not None:
            d_earliest = date.fromisoformat(dates[idx_earliest].split("T")[0])
            d_latest = date.fromisoformat(dates[idx_latest].split("T")[0])
            separation = (d_latest - d_earliest).days
        else:
            separation = idx_latest - idx_earliest

        if separation < self.min_separation_days:
            return None

        a_earliest = float(values[idx_earliest, area_index])
        a_latest = float(values[idx_latest, area_index])

        if a_earliest <= 0.0:
            return None

        delta_rel = (a_latest - a_earliest) / a_earliest
        return float(max(0.0, delta_rel))

    def predict_panel(self, panel: MultiModalPanel) -> float:
        if panel.origin != "fixture":
            from .admission import blocked
            blocked("research baseline scoring")
        return self.predict_window(panel.values, panel.mask, panel.dates, panel.channel_names)

    def predict_status(self, panel):
        value = self.predict_panel(panel)
        return {'value': value, 'status': 'ESTIMATED' if value is not None else 'NOT_ESTIMABLE',
                'reason': None if value is not None else 'INSUFFICIENT_MEASUREMENT_OR_FITTED_SUPPORT',
                'origin': 'fixture', 'research_admissible': False}

    @classmethod
    def fit(cls, *args, **kwargs):
        from .admission import blocked
        blocked("research baseline fitting")


# ---------------------------------------------------------------------------
# 3. Weather-Only Anomaly Baseline
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class WeatherOnlyBaseline:
    """Atmospheric anomaly baseline tracking standardized weather deviations."""
    training_lake_ids: Tuple[str, ...]
    temporal_cutoff: Optional[str]
    weather_means: Tuple[float, ...]
    weather_scales: Tuple[float, ...]
    weather_channel_names: Tuple[str, ...]
    channel_schema_hash: str
    state_hash: str

    @classmethod
    def fit_fixture(
        cls,
        panels: Sequence[MultiModalPanel],
        training_lake_ids: Sequence[str],
        temporal_cutoff: Optional[str] = None,
    ) -> WeatherOnlyBaseline:
        """Fit weather means and scales strictly on training panels."""
        if any(p.origin != "fixture" for p in panels):
            raise ValueError("baseline fixtures require explicit fixture origin")
        train_set = set(training_lake_ids)
        if not train_set:
            raise ValueError("training_lake_ids must be non-empty")

        valid_panels = [
            p for p in panels
            if p.lake_id in train_set and (temporal_cutoff is None or p.end_date <= temporal_cutoff)
        ]
        if not valid_panels:
            raise ValueError("No valid training panels available to fit WeatherOnlyBaseline")
        channel_names = valid_panels[0].channel_names
        schema_hash = valid_panels[0].channel_schema_hash
        if any(p.channel_names != channel_names or p.channel_schema_hash != schema_hash for p in valid_panels):
            raise ValueError("weather fit panels have different channel schemas")
        weather_names = tuple(name for name in channel_names if CHANNEL_MODALITIES[name] == "weather")
        if not weather_names:
            raise ValueError("declared panel schema has no weather channels")

        means = []
        scales = []
        indices = [channel_names.index(name) for name in weather_names]
        for c in indices:
            obs = []
            for p in valid_panels:
                v = p.values[:, c][p.mask[:, c]]
                if v.size > 0:
                    obs.append(v)
            if obs:
                stacked = np.concatenate(obs)
                mu = float(np.mean(stacked))
                sigma = float(np.std(stacked))
                if not np.isfinite(sigma):
                    raise ValueError("nonfinite fitted scale")
                if sigma == 0.0:
                    sigma = 1.0
            else:
                raise ValueError("missing training channel")
            means.append(mu)
            scales.append(sigma)

        hash_payload = {
            "training_lake_ids": sorted(list(train_set)),
            "temporal_cutoff": temporal_cutoff,
            "weather_channel_names": list(weather_names),
            "channel_schema_hash": schema_hash,
            "weather_means": list(means),
            "weather_scales": list(scales),
        }
        state_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode("utf-8")).hexdigest()

        return cls(
            training_lake_ids=tuple(sorted(train_set)),
            temporal_cutoff=temporal_cutoff,
            weather_means=tuple(means),
            weather_scales=tuple(scales),
            weather_channel_names=weather_names,
            channel_schema_hash=schema_hash,
            state_hash=state_hash,
        )

    def predict_window(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        dates: Optional[Sequence[str]] = None,
        eps: float = 1e-7,
        channel_names: Optional[Sequence[str]] = None,
    ) -> float:
        """Compute standardized anomaly distance over observed weather channels."""
        if channel_names is None:
            raise ValueError("weather prediction requires its explicit channel schema")
        if tuple(name for name in channel_names if CHANNEL_MODALITIES.get(name) == "weather") != self.weather_channel_names:
            raise ValueError("weather channels differ from fitted baseline schema")
        indices = [tuple(channel_names).index(name) for name in self.weather_channel_names]
        w_values = values[:, indices]
        w_mask = mask[:, indices] & np.isfinite(w_values)

        total_obs = int(np.sum(w_mask))
        if total_obs == 0:
            return None

        means_arr = np.array(self.weather_means, dtype=np.float64)
        scales_arr = np.array(self.weather_scales, dtype=np.float64)

        means_grid = np.broadcast_to(means_arr, w_values.shape)
        scales_grid = np.broadcast_to(scales_arr, w_values.shape)

        v = w_values[w_mask]
        mu = means_grid[w_mask]
        sig = scales_grid[w_mask]

        diff = (v - mu) / (sig + eps)
        sq_dist = diff ** 2
        score = float(np.sqrt(np.sum(sq_dist) / (total_obs + eps)))
        return score

    def predict_panel(self, panel: MultiModalPanel) -> float:
        if panel.origin != "fixture":
            from .admission import blocked
            blocked("research baseline scoring")
        if panel.channel_schema_hash != self.channel_schema_hash:
            raise ValueError("panel channel schema differs from fitted weather baseline")
        return self.predict_window(panel.values, panel.mask, panel.dates, panel.channel_names)

    def predict_status(self, panel):
        value = self.predict_panel(panel)
        return {'value': value, 'status': 'ESTIMATED' if value is not None else 'NOT_ESTIMABLE',
                'reason': None if value is not None else 'INSUFFICIENT_MEASUREMENT_OR_FITTED_SUPPORT',
                'origin': 'fixture', 'research_admissible': False}

    @classmethod
    def fit(cls, *args, **kwargs):
        from .admission import blocked
        blocked("research baseline fitting")


# ---------------------------------------------------------------------------
# 4. Robust PCA Residual Baseline
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RobustPCABaseline:
    """Linear subspace PCA baseline computing normalized reconstruction error."""
    training_lake_ids: Tuple[str, ...]
    temporal_cutoff: Optional[str]
    k_components: int
    channel_names: Tuple[str, ...]
    channel_schema_hash: str
    window_length: int
    channel_means: Tuple[float, ...]
    mean_window: np.ndarray   # shape (1980,)
    components: np.ndarray    # shape (k_eff, 1980)
    state_hash: str

    @classmethod
    def fit_fixture(
        cls,
        panels: Sequence[MultiModalPanel],
        training_lake_ids: Sequence[str],
        temporal_cutoff: Optional[str] = None,
        k_components: int = 8,
    ) -> RobustPCABaseline:
        """Fit PCA subspace strictly on training panels."""
        if any(p.origin != "fixture" for p in panels):
            raise ValueError("baseline fixtures require explicit fixture origin")
        train_set = set(training_lake_ids)
        if not train_set:
            raise ValueError("training_lake_ids must be non-empty")

        valid_panels = [
            p for p in panels
            if p.lake_id in train_set and (temporal_cutoff is None or p.end_date <= temporal_cutoff)
        ]
        if not valid_panels:
            raise ValueError("No valid training panels available to fit RobustPCABaseline")
        channel_names = valid_panels[0].channel_names
        schema_hash = valid_panels[0].channel_schema_hash
        window_length = len(valid_panels[0].dates)
        expected_shape = valid_panels[0].values.shape
        if any(p.channel_names != channel_names or p.channel_schema_hash != schema_hash or
               p.values.shape != expected_shape for p in valid_panels):
            raise ValueError("PCA fit panels have different channel schemas or schedule shapes")
        n_channels = len(channel_names)

        # 1. Compute training channel means for missing value imputation
        channel_means_list = []
        for c in range(n_channels):
            obs = []
            for p in valid_panels:
                v = p.values[:, c][p.mask[:, c]]
                if v.size > 0:
                    obs.append(v)
            if obs:
                channel_means_list.append(float(np.mean(np.concatenate(obs))))
            else:
                raise ValueError("missing training channel")

        ch_means_arr = np.array(channel_means_list, dtype=np.float64)

        # 2. Impute and flatten training panels to shape (N, 1980)
        imputed_windows = []
        for p in valid_panels:
            valid_m = p.mask & np.isfinite(p.values)
            imputed = np.where(valid_m, p.values, ch_means_arr)
            imputed_windows.append(imputed.reshape(-1))

        x_train = np.stack(imputed_windows, axis=0)  # (N, 1980)
        mean_window = np.mean(x_train, axis=0)       # (1980,)
        x_centered = x_train - mean_window

        # 3. Truncated SVD for principal components
        # Full SVD: X = U * S * Vt
        n_samples, n_features = x_centered.shape
        max_k = min(n_samples, n_features, k_components)
        if max_k > 0 and n_samples > 1:
            _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
            components = vt[:max_k, :]  # (k_eff, 1980)
        else:
            components = np.zeros((0, n_features), dtype=np.float64)

        hash_payload = {
            "training_lake_ids": sorted(list(train_set)),
            "temporal_cutoff": temporal_cutoff,
            "channel_names": list(channel_names),
            "channel_schema_hash": schema_hash,
            "window_length": window_length,
            "k_components": k_components,
            "channel_means": list(channel_means_list),
            "mean_window_digest": hashlib.sha256(mean_window.tobytes()).hexdigest(),
            "components_digest": hashlib.sha256(components.tobytes()).hexdigest(),
        }
        state_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode("utf-8")).hexdigest()

        return cls(
            training_lake_ids=tuple(sorted(train_set)),
            temporal_cutoff=temporal_cutoff,
            k_components=k_components,
            channel_names=channel_names,
            channel_schema_hash=schema_hash,
            window_length=window_length,
            channel_means=tuple(channel_means_list),
            mean_window=mean_window,
            components=components,
            state_hash=state_hash,
        )

    def predict_window(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        dates: Optional[Sequence[str]] = None,
        eps: float = 1e-7,
    ) -> float:
        """Reconstruct window and return normalized mean squared reconstruction error over observed entries."""
        values = np.asarray(values)
        mask = np.asarray(mask)
        if values.shape != (self.window_length, len(self.channel_names)) or mask.shape != values.shape:
            raise ValueError("PCA input schedule/channel shape differs from fitted transform")
        ch_means_arr = np.array(self.channel_means, dtype=np.float64)
        valid_m = mask & np.isfinite(values)
        x_imputed = np.where(valid_m, values, ch_means_arr).reshape(-1)

        x_diff = x_imputed - self.mean_window
        if self.components.shape[0] > 0:
            # z = x_diff @ components.T -> (k_eff,)
            z = x_diff @ self.components.T
            # recon = mean_window + z @ components
            recon = self.mean_window + z @ self.components
        else:
            recon = self.mean_window

        recon_2d = recon.reshape(self.window_length, len(self.channel_names))
        valid_entries = mask & np.isfinite(values)
        total_obs = int(np.sum(valid_entries))
        if total_obs == 0:
            return None

        v = values[valid_entries]
        recon_valid = recon_2d[valid_entries]
        squared_error = (v - recon_valid) ** 2
        score = float(np.sum(squared_error) / (total_obs + eps))
        return score

    def predict_panel(self, panel: MultiModalPanel) -> float:
        if panel.origin != "fixture":
            from .admission import blocked
            blocked("research baseline scoring")
        if panel.channel_names != self.channel_names or panel.channel_schema_hash != self.channel_schema_hash:
            raise ValueError("panel channel schema differs from fitted PCA baseline")
        return self.predict_window(panel.values, panel.mask, panel.dates)

    def predict_status(self, panel):
        value = self.predict_panel(panel)
        return {'value': value, 'status': 'ESTIMATED' if value is not None else 'NOT_ESTIMABLE',
                'reason': None if value is not None else 'INSUFFICIENT_MEASUREMENT_OR_FITTED_SUPPORT',
                'origin': 'fixture', 'research_admissible': False}

    @classmethod
    def fit(cls, *args, **kwargs):
        from .admission import blocked
        blocked("research baseline fitting")
