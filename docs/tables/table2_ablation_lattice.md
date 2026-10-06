# Table 2: 2^N Sensor Ablation Lattice Comparison

| Configuration ID | Active Modalities | Optical | SAR | ERA5 | Valid Windows | Mean Recon MSE | Sensitivity Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `full` | optical+sar+era5 | Yes | Yes | Yes | 11 | 1.3465 | 1.3465 |
| `opt_sar` | optical+sar | Yes | Yes | No | 11 | 0.8834 | 0.8834 |
| `opt_era5` | optical+era5 | Yes | No | Yes | 11 | 1.6750 | 1.6750 |
| `sar_era5` | sar+era5 | No | Yes | Yes | 12 | 1.3138 | 1.3138 |
| `opt_only` | optical | Yes | No | No | 11 | 1.8522 | 1.8522 |
| `sar_only` | sar | No | Yes | No | 12 | 0.1671 | 0.1671 |
| `era5_only` | era5 | No | No | Yes | 12 | 1.7521 | 1.7521 |
