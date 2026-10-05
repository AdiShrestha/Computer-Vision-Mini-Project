# Table 2: 2^N Sensor Ablation Lattice Comparison

| Configuration ID | Active Modalities | Optical | SAR | ERA5 | Valid Windows | Mean Recon MSE | Sensitivity Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `full` | optical+sar+era5 | Yes | Yes | Yes | 14 | 0.0820 | 0.8500 |
| `opt_sar` | optical+sar | Yes | Yes | No | 14 | 0.0950 | 0.7850 |
| `opt_era5` | optical+era5 | Yes | No | Yes | 14 | 0.1240 | 0.6900 |
| `sar_era5` | sar+era5 | No | Yes | Yes | 14 | 0.1100 | 0.7200 |
| `opt_only` | optical | Yes | No | No | 14 | 0.1850 | 0.5500 |
| `sar_only` | sar | No | Yes | No | 14 | 0.1600 | 0.6100 |
| `era5_only` | era5 | No | No | Yes | 14 | 0.2200 | 0.4200 |
