# Legacy source inspection index

All source Python files were read in full and parsed; signals below are review prompts, not automatic findings. Scientific findings and their interpretation are in plan.md. Generated arrays and old outputs are not research evidence.

## source/config/__init__.py

SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

0 lines. Definitions: 



## source/data/__init__.py

SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

0 lines. Definitions: 



## source/data/acquisition/__init__.py

SHA-256: `1984f5370685744e3fea9a9f0934077f7f9088d73018a169008bc5052e3407de`

32 lines. Definitions: 



## source/data/acquisition/acquire_era5.py

SHA-256: `a52363ed97f4a1252064dfcc7d2ff8a2eb1a1b9451c9d5f9fd045b67097b0eb9`

13 lines. Definitions: acquire

- Line 6: `def acquire(lake_id=None, start_date="2016-01-01", end_date="2024-10-31",`

## source/data/acquisition/acquire_itslive.py

SHA-256: `e389c812a0d50e3988b79709a807a0ba13f74aeeb6a4b4aaca2f3bd30e5358ed`

13 lines. Definitions: acquire

- Line 6: `def acquire(lake_id=None, start_date="2016-01-01", end_date="2024-10-31",`

## source/data/acquisition/acquire_landsat.py

SHA-256: `97f7fd7f02e416cefeb83ac6c8ce7cc319ddb36e41f1d52e4a8c9eb37b15f5d9`

13 lines. Definitions: acquire

- Line 6: `def acquire(lake_id=None, start_date="2016-01-01", end_date="2024-10-31",`

## source/data/acquisition/acquire_modis.py

SHA-256: `50e010cb95aee561b476c4f91fcd9e86ddf8947ac2833f09da2c386ad91a2a19`

13 lines. Definitions: acquire

- Line 6: `def acquire(lake_id=None, start_date="2016-01-01", end_date="2024-10-31",`

## source/data/acquisition/acquire_sentinel1.py

SHA-256: `84a8f4840da4045475a73202a1f40e90c1a699cd7dff67c97c80b3a7b946d84d`

13 lines. Definitions: acquire

- Line 6: `def acquire(lake_id=None, start_date="2016-01-01", end_date="2024-10-31",`

## source/data/acquisition/acquire_sentinel2.py

SHA-256: `13b6532491d8ab645bc42694fa9cda1b15359c26e20b7d3c001792251fdfca3f`

13 lines. Definitions: acquire

- Line 6: `def acquire(lake_id=None, start_date="2016-01-01", end_date="2024-10-31",`

## source/data/acquisition/common.py

SHA-256: `449158d8260235645f45fbfb6323b247f59a6b4e708f4f537ef4cad1affe1ee3`

59 lines. Definitions: AcquisitionError, AcquisitionBlockedError, AuthenticationError, NetworkReachabilityError, RateLimitError, check_endpoint_reachability, verify_fail_closed_preconditions

- Line 14: `class AcquisitionError(Exception):`
- Line 19: `class AcquisitionBlockedError(AcquisitionError):`
- Line 24: `class AuthenticationError(AcquisitionBlockedError):`
- Line 29: `class NetworkReachabilityError(AcquisitionBlockedError):`
- Line 34: `class RateLimitError(AcquisitionBlockedError):`
- Line 39: `def check_endpoint_reachability(host: str, port: int = 443, timeout: float = 5.0) -> bool:`
- Line 45: `    except (socket.timeout, socket.gaierror, OSError) as e:`
- Line 50: `def verify_fail_closed_preconditions(service_name: str, host: str) -> None:`
- Line 52: `    Assert network reachability before attempting acquisition.`

## source/data/acquisition/era5.py

SHA-256: `9ba0ec8b7d817d4602a5aac67f68a844abd41ab42fe981ee2902c3cc9ff9891c`

75 lines. Definitions: ERA5AcquisitionAdapter, __init__, initialize, request_lake_climate_data

- Line 16: `class ERA5AcquisitionAdapter:`
- Line 17: `    def __init__(self, key_file: Optional[Path] = None):`
- Line 21: `    def initialize(self) -> None:`
- Line 32: `        except Exception as e:`
- Line 38: `    def request_lake_climate_data(`
- Line 71: `        except Exception as e:`

## source/data/acquisition/modis.py

SHA-256: `82f2c5f3d79495974127edee4136035a8f3edb0cfdcdab8926b582bb9b0b378d`

72 lines. Definitions: ModisLSTAcquisitionAdapter, __init__, initialize, acquire_lake_series

- Line 16: `class ModisLSTAcquisitionAdapter:`
- Line 17: `    def __init__(self, project_id: Optional[str] = None):`
- Line 21: `    def initialize(self) -> None:`
- Line 31: `        except Exception as e:`
- Line 37: `    def acquire_lake_series(`
- Line 68: `        except Exception as e:`

## source/data/acquisition/sentinel1.py

SHA-256: `28b0602fb8653a5b4b2b9dfdcb14e956c7dd82553fc289e54f332173c33a5d3b`

77 lines. Definitions: Sentinel1AcquisitionAdapter, __init__, initialize, acquire_lake_series

- Line 16: `class Sentinel1AcquisitionAdapter:`
- Line 17: `    def __init__(self, project_id: Optional[str] = None):`
- Line 21: `    def initialize(self) -> None:`
- Line 31: `        except Exception as e:`
- Line 37: `    def acquire_lake_series(`
- Line 73: `        except Exception as e:`

## source/data/acquisition/sentinel2.py

SHA-256: `bcb4ecb553223cfbd876988055ed7e6a316708950af635d9fb35b774eac10f08`

74 lines. Definitions: Sentinel2AcquisitionAdapter, __init__, initialize, acquire_lake_series

- Line 16: `class Sentinel2AcquisitionAdapter:`
- Line 17: `    def __init__(self, project_id: Optional[str] = None):`
- Line 21: `    def initialize(self) -> None:`
- Line 31: `        except Exception as e:`
- Line 37: `    def acquire_lake_series(`
- Line 70: `        except Exception as e:`

## source/data/acquisition/topography.py

SHA-256: `e32dd1f2331898ad504a8c3b2f06d3edfe6065233c6b50c19339346ee4de0c44`

68 lines. Definitions: TopographyAcquisitionAdapter, __init__, initialize, acquire_lake_topography

- Line 16: `class TopographyAcquisitionAdapter:`
- Line 17: `    def __init__(self, project_id: Optional[str] = None):`
- Line 21: `    def initialize(self) -> None:`
- Line 31: `        except Exception as e:`
- Line 37: `    def acquire_lake_topography(`
- Line 64: `        except Exception as e:`

## source/data/channels/__init__.py

SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

0 lines. Definitions: 



## source/data/channels/assemble_features.py

SHA-256: `889e17b86d23a1073933154edc1f000a203fec659fd6d775cdcdc668763545f9`

201 lines. Definitions: load_lake_registry, date_range_daily, load_csv_as_dict, safe_float, assemble_lake_features, compute_normalization_stats, assemble_all

- Line 26: `def load_lake_registry():`
- Line 32: `def date_range_daily(start_str, end_str):`
- Line 44: `def load_csv_as_dict(csv_path, date_col='date'):`
- Line 58: `def safe_float(val):`
- Line 64: `    except (ValueError, TypeError):`
- Line 68: `def assemble_lake_features(lake_id, dates, raw_root):`
- Line 121: `def compute_normalization_stats(feature_dir, registry, dates):`
- Line 155: `def assemble_all():`

## source/data/channels/channel_registry.py

SHA-256: `0a43021d3c3aac3ffe00e1d78169e16ff18f9364c7f20c11844526eff65869cd`

16 lines. Definitions: 



## source/data/channels/extract_extent.py

SHA-256: `49ba5f43560f41f735862e60f2852021a833e66ddde9924746bf220760f7f691`

42 lines. Definitions: extract

- Line 10: `def extract(lake_id: str, window_start: str, window_end: str,`

## source/data/channels/extract_meteorological.py

SHA-256: `84ca3ecda30a967aa943d8281c0fb26f3446932c84e6326b4401631bcbc1af19`

40 lines. Definitions: extract

- Line 10: `def extract(lake_id: str, window_start: str, window_end: str,`

## source/data/channels/extract_sar.py

SHA-256: `83e9bbf3600557426cad3aed5520daf37858c9cb51c37981ac9d439ef8e739e1`

54 lines. Definitions: extract

- Line 10: `def extract(lake_id: str, window_start: str, window_end: str,`
- Line 24: `            coherence = float(np.clip(0.75 + (vv_db / 100.0), 0.0, 1.0))`

## source/data/channels/extract_spectral.py

SHA-256: `bbdd3b94f4f2c03d6ef7539c6f3d746d3081a5486426cc24231c8b442420f95d`

43 lines. Definitions: extract

- Line 10: `def extract(lake_id: str, window_start: str, window_end: str,`

## source/data/channels/extract_temperature.py

SHA-256: `363a6208269f33d27bb1af36fd1c6f389bc2e310a05b1b0dc900265613916309`

37 lines. Definitions: extract

- Line 10: `def extract(lake_id: str, window_start: str, window_end: str,`
- Line 22: `            climatology_c = 2.5`
- Line 23: `            temp_anomaly = float(temp_c - climatology_c)`

## source/data/channels/extract_velocity.py

SHA-256: `73657126d762ac62ee3b53cfd882c7ebc926d8070b652dfa6106b11d6ebb0746`

39 lines. Definitions: extract

- Line 10: `def extract(lake_id: str, window_start: str, window_end: str,`

## source/data/insar/__init__.py

SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

0 lines. Definitions: 



## source/data/insar/insar_feasibility.py

SHA-256: `c67d42b614d96021f6582e3ec540b2742ac47138a817445c32fe26c0e327b2d3`

94 lines. Definitions: assess_coherence, generate_interferogram, assess_feasibility

- Line 12: `def assess_coherence(lake_id: str, slc_dir: str, config: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:`
- Line 24: `def generate_interferogram(master_path: str, slave_path: str, dem_path: str, output_dir: str) -> Dict[str, Any]:`
- Line 25: `    """Mock/wrapper for SNAP/ISCE2 interferogram generation."""`
- Line 34: `def assess_feasibility(config: Dict[str, Any] = None, registry: Dict[str, Any] = None) -> Dict[str, Any]:`

## source/data/loaders/__init__.py

SHA-256: `89f9365720248a19607fcd0c959ccb23d05ef82b4a35854d94703640ff80fb31`

4 lines. Definitions: 



## source/data/loaders/lake_dataset.py

SHA-256: `1f324eba4b30fc478a7df69e62f7c9b69d6cb9fb6afd9cca290c9af38a13137e`

350 lines. Definitions: load_registry, get_lakes_by_role, GlacialLakeDataset, create_data_loaders, create_inference_loader, __init__, _compute_norm_stats, get_norm_stats, __len__, __getitem__

- Line 15: `import random`
- Line 30: `def load_registry(registry_path: str) -> dict:`
- Line 36: `def get_lakes_by_role(registry: dict) -> Dict[str, List[str]]:`
- Line 51: `class GlacialLakeDataset(Dataset):`
- Line 76: `    def __init__(`
- Line 140: `    def _compute_norm_stats(self):`
- Line 152: `            assert role not in EVALUATION_ROLES, (`
- Line 176: `    def get_norm_stats(self) -> Dict[str, np.ndarray]:`
- Line 184: `    def __len__(self) -> int:`
- Line 187: `    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:`
- Line 194: `            assert role not in EVALUATION_ROLES, (`
- Line 212: `def create_data_loaders(`
- Line 230: `        val_split_seed: Random seed for split (default 7, INV-012)`
- Line 255: `    rng = random.Random(val_split_seed)`
- Line 308: `def create_inference_loader(`

## source/data/preprocessing/__init__.py

SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

0 lines. Definitions: 



## source/data/preprocessing/cloud_mask_s2.py

SHA-256: `9971bea7b650b55c48d623ed817d02c33a53648626b12ffabd28969d6c442a0e`

46 lines. Definitions: apply_cloud_mask

- Line 8: `def apply_cloud_mask(`

## source/data/preprocessing/common.py

SHA-256: `78472a9a0524477cb92efd3cb02604f983decaa6445868a33e99e512f453747e`

118 lines. Definitions: build_time_windows, composite_within_window, generate_quality_mask, reproject_to_utm, resample_to_grid

- Line 15: `def build_time_windows(start_date: str, end_date: str,`
- Line 52: `def composite_within_window(scenes: List[np.ndarray], method: str = 'median') -> np.ndarray:`
- Line 74: `def generate_quality_mask(array: np.ndarray, source_qa: Optional[np.ndarray] = None) -> np.ndarray:`
- Line 100: `def reproject_to_utm(array: np.ndarray, src_crs: str, dst_crs: str, resolution: float = 10.0) -> np.ndarray:`
- Line 101: `    """Mock/wrapper for spatial reprojection to target UTM CRS."""`
- Line 106: `def resample_to_grid(array: np.ndarray, target_shape: Tuple[int, int], method: str = 'bilinear') -> np.ndarray:`

## source/data/preprocessing/preprocess_era5.py

SHA-256: `e54fd0119e9eda3e17948e0c479b17585b7846c82f44dc82ce963458dad1c99c`

49 lines. Definitions: preprocess

- Line 14: `def preprocess(lake_id: str, raw_dir: str, output_dir: str, config: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:`
- Line 28: `    for w_start, w_end in windows[:5]:`
- Line 32: `        dummy_data = np.random.uniform(-15.0, 25.0, size=(180, 3)).astype(np.float32)`
- Line 33: `        dummy_qa = generate_quality_mask(dummy_data)`
- Line 35: `        np.savez_compressed(out_path, data=dummy_data, quality=dummy_qa, metadata={`
- Line 39: `        quality_flags[w_start] = {"valid_samples": int(np.sum(dummy_qa == 1))}`

## source/data/preprocessing/preprocess_itslive.py

SHA-256: `a3f9d0be457c695ab6b0e26ed6ab9a445a9ff6002b913ef67a5e59a727e92933`

49 lines. Definitions: preprocess

- Line 14: `def preprocess(lake_id: str, raw_dir: str, output_dir: str, config: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:`
- Line 28: `    for w_start, w_end in windows[:5]:`
- Line 32: `        dummy_data = np.random.uniform(0.0, 150.0, size=(50, 50, 2)).astype(np.float32)`
- Line 33: `        dummy_qa = generate_quality_mask(dummy_data)`
- Line 35: `        np.savez_compressed(out_path, data=dummy_data, quality=dummy_qa, metadata={`
- Line 39: `        quality_flags[w_start] = {"valid_pixels": int(np.sum(dummy_qa == 1))}`

## source/data/preprocessing/preprocess_modis.py

SHA-256: `e8850406ba433714928956e15ff6811d1523bcb16051f5941f4cc5d305844ae0`

49 lines. Definitions: preprocess

- Line 14: `def preprocess(lake_id: str, raw_dir: str, output_dir: str, config: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:`
- Line 28: `    for w_start, w_end in windows[:5]:`
- Line 32: `        dummy_data = np.random.uniform(240.0, 300.0, size=(10, 10, 1)).astype(np.float32)`
- Line 33: `        dummy_qa = generate_quality_mask(dummy_data)`
- Line 35: `        np.savez_compressed(out_path, data=dummy_data, quality=dummy_qa, metadata={`
- Line 39: `        quality_flags[w_start] = {"valid_pixels": int(np.sum(dummy_qa == 1))}`

## source/data/preprocessing/preprocess_optical.py

SHA-256: `e31258ecef8436116f61d1ed60b86dd702821b83d82e69cbac7e121d89a0079f`

52 lines. Definitions: preprocess

- Line 15: `def preprocess(lake_id: str, raw_dir: str, output_dir: str, config: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:`
- Line 31: `    for w_start, w_end in windows[:5]: # Light processing for output generation`
- Line 34: `        # Synthetic / loaded array representation (10m resolution tile mock: 100x100 4-band)`
- Line 35: `        dummy_data = np.random.uniform(0.0, 0.4, size=(100, 100, 4)).astype(np.float32)`
- Line 36: `        dummy_qa = generate_quality_mask(dummy_data)`
- Line 38: `        np.savez_compressed(out_path, data=dummy_data, quality=dummy_qa, metadata={`
- Line 42: `        quality_flags[w_start] = {"cloud_fraction": 0.12, "valid_pixels": int(np.sum(dummy_qa == 1))}`

## source/data/preprocessing/preprocess_sar.py

SHA-256: `90637d88f1aa5a39e73d6c2a2fb9b37aff361d78fe3f79ecd1768cf9bc3554e7`

50 lines. Definitions: preprocess

- Line 15: `def preprocess(lake_id: str, raw_dir: str, output_dir: str, config: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:`
- Line 29: `    for w_start, w_end in windows[:5]:`
- Line 33: `        dummy_data = np.random.uniform(-25.0, 5.0, size=(100, 100, 2)).astype(np.float32)`
- Line 34: `        dummy_qa = generate_quality_mask(dummy_data)`
- Line 36: `        np.savez_compressed(out_path, data=dummy_data, quality=dummy_qa, metadata={`
- Line 40: `        quality_flags[w_start] = {"valid_pixels": int(np.sum(dummy_qa == 1))}`

## source/data/registry/validate_registry.py

SHA-256: `b4f2040413412a5c8dfe66bd469c5e2c27adc9fa6089ef61af0ea74f16573738`

119 lines. Definitions: validate, main

- Line 17: `def validate(registry_path: str, schema_path: str) -> Tuple[bool, List[str]]:`
- Line 38: `    except json.JSONDecodeError as e:`
- Line 44: `    except json.JSONDecodeError as e:`
- Line 95: `def main():`

## source/data/schemas/__init__.py

SHA-256: `f3f540dda9a0ad78ee0f421f826cf864be7858af5c4aaa468ff985c23ee960e3`

31 lines. Definitions: 



## source/data/schemas/data_manifest_schema.py

SHA-256: `a43c10db1658aa400958498a3defe6bf260e25f8c38abd8d208a34ab98708461`

321 lines. Definitions: EvidenceClass, _validate_percentage, LakeDataEntry, ChannelProvenanceEntry, GapStatistics, DatasetManifest, __post_init__, validate_record, to_dict, from_dict, __post_init__, validate_record, to_dict, from_dict, __post_init__, validate_record, to_dict, from_dict, __post_init__, validate_record, to_dict, to_json, from_dict, from_json

- Line 19: `class EvidenceClass(str, Enum):`
- Line 27: `def _validate_percentage(value: Any, field_name: str) -> float:`
- Line 37: `class LakeDataEntry:`
- Line 54: `    def __post_init__(self) -> None:`
- Line 57: `    def validate_record(self) -> "LakeDataEntry":`
- Line 97: `    def to_dict(self) -> Dict[str, Any]:`
- Line 109: `    def from_dict(cls, value: Mapping[str, Any]) -> "LakeDataEntry":`
- Line 117: `class ChannelProvenanceEntry:`
- Line 130: `    def __post_init__(self) -> None:`
- Line 133: `    def validate_record(self) -> "ChannelProvenanceEntry":`
- Line 148: `    def to_dict(self) -> Dict[str, Any]:`
- Line 158: `    def from_dict(cls, value: Mapping[str, Any]) -> "ChannelProvenanceEntry":`
- Line 166: `class GapStatistics:`
- Line 175: `    def __post_init__(self) -> None:`
- Line 178: `    def validate_record(self) -> "GapStatistics":`
- Line 198: `    def to_dict(self) -> Dict[str, Any]:`
- Line 208: `    def from_dict(cls, value: Mapping[str, Any]) -> "GapStatistics":`
- Line 216: `class DatasetManifest:`
- Line 233: `    def __post_init__(self) -> None:`
- Line 236: `    def validate_record(self) -> "DatasetManifest":`
- Line 262: `            raise SchemaValidationError("evidence_class must be an EvidenceClass value")`
- Line 265: `    def to_dict(self) -> Dict[str, Any]:`
- Line 281: `    def to_json(self, *, indent: Optional[int] = 2) -> str:`
- Line 287: `    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetManifest":`
- Line 309: `        except (TypeError, ValueError) as exc:`
- Line 314: `    def from_json(cls, payload: str) -> "DatasetManifest":`
- Line 319: `        except json.JSONDecodeError as exc:`

## source/data/schemas/feature_schema_v2.py

SHA-256: `739aaa9d5a3d6673e56ae9713de81f65d7e56e07fd42f7c080c6c3513df0d8ca`

393 lines. Definitions: FeatureChannel, FeatureSchemaV2, __post_init__, transform_fit_scope, missingness_indicator_id, __post_init__, validate_schema, default_multisensor_schema, channel_count, temporal_channels, static_channels, compute_schema_hash, validate_matrix

- Line 26: `class FeatureChannel:`
- Line 41: `    def __post_init__(self) -> None:`
- Line 79: `    def transform_fit_scope(self) -> str:`
- Line 83: `    def missingness_indicator_id(self) -> str:`
- Line 88: `class FeatureSchemaV2:`
- Line 92: `    def __post_init__(self) -> None:`
- Line 95: `    def validate_schema(self) -> None:`
- Line 107: `    def default_multisensor_schema(cls) -> "FeatureSchemaV2":`
- Line 299: `    def channel_count(self) -> int:`
- Line 303: `    def temporal_channels(self) -> List[FeatureChannel]:`
- Line 307: `    def static_channels(self) -> List[FeatureChannel]:`
- Line 310: `    def compute_schema_hash(self) -> str:`
- Line 323: `    def validate_matrix(self, matrix_dict: Dict[str, Any]) -> List[str]:`
- Line 352: `            except (TypeError, ValueError):`

## source/data/schemas/provenance_schema.py

SHA-256: `e172a33b2cf9c4acc0ecc9f0604de6881d91550b645ed2d841a70666af29aab4`

277 lines. Definitions: SchemaValidationError, _require_exact_fields, _require_nonempty_string, validate_sha256, validate_utc_timestamp, DownloadedFileRecord, ExecutionEnvironmentRecord, AcquisitionProvenanceRecord, __post_init__, validate_record, from_dict, __post_init__, validate_record, from_dict, __post_init__, validate_record, to_dict, to_json, from_dict, from_json

- Line 16: `class SchemaValidationError(ValueError):`
- Line 20: `def _require_exact_fields(`
- Line 35: `def _require_nonempty_string(value: Any, field_name: str) -> str:`
- Line 41: `def validate_sha256(value: Any, field_name: str = "checksum") -> str:`
- Line 50: `def validate_utc_timestamp(value: Any, field_name: str) -> datetime:`
- Line 55: `    except ValueError as exc:`
- Line 63: `class DownloadedFileRecord:`
- Line 68: `    def __post_init__(self) -> None:`
- Line 71: `    def validate_record(self) -> "DownloadedFileRecord":`
- Line 81: `    def from_dict(cls, value: Mapping[str, Any]) -> "DownloadedFileRecord":`
- Line 90: `class ExecutionEnvironmentRecord:`
- Line 96: `    def __post_init__(self) -> None:`
- Line 99: `    def validate_record(self) -> "ExecutionEnvironmentRecord":`
- Line 111: `    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionEnvironmentRecord":`
- Line 125: `class AcquisitionProvenanceRecord:`
- Line 154: `    def __post_init__(self) -> None:`
- Line 157: `    def validate_record(self) -> "AcquisitionProvenanceRecord":`
- Line 179: `        except (TypeError, ValueError) as exc:`
- Line 243: `    def to_dict(self) -> Dict[str, Any]:`
- Line 247: `    def to_json(self, *, indent: Optional[int] = 2) -> str:`
- Line 253: `    def from_dict(cls, value: Mapping[str, Any]) -> "AcquisitionProvenanceRecord":`
- Line 270: `    def from_json(cls, payload: str) -> "AcquisitionProvenanceRecord":`
- Line 275: `        except json.JSONDecodeError as exc:`

## source/evaluation/__init__.py

SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

0 lines. Definitions: 



## source/evaluation/ablation.py

SHA-256: `73a59513297d876ff898422e0dd49403b3b33d54c2c8d2f8361db93cc5cf4a52`

246 lines. Definitions: apply_channel_mask, AblationExperiment, __init__, verify_no_retraining, run_config, _get_ablated_embeddings, _score_a_from_normalized, ablated_scorer_fn

- Line 73: `def apply_channel_mask(features: np.ndarray, keep_cols: List[int]) -> np.ndarray:`
- Line 95: `class AblationExperiment:`
- Line 102: `    def __init__(`
- Line 116: `    def verify_no_retraining(self) -> bool:`
- Line 122: `    def run_config(`
- Line 184: `        def ablated_scorer_fn(modified_features: np.ndarray) -> np.ndarray:`
- Line 225: `    def _get_ablated_embeddings(self, masked_normed: np.ndarray) -> np.ndarray:`
- Line 235: `    def _score_a_from_normalized(self, masked_normed: np.ndarray) -> np.ndarray:`

## source/evaluation/ablation_figures.py

SHA-256: `8a871a565dbb4e69e3d5ee9163b4e71ccd48c4783ea15d6652ea550032c1930a`

160 lines. Definitions: generate_ablation_figures

- Line 28: `def generate_ablation_figures():`
- Line 36: `    assert os.path.isfile(ablation_summary_path), f"ABORT: {ablation_summary_path} not found"`
- Line 39: `    assert summary.get('ablation_version') == 'C05-02', (`

## source/evaluation/data_quality.py

SHA-256: `9211830094bae67f16aed4ba56f8cf192ad101a0c12c5e339fae40e76ce3cfe1`

131 lines. Definitions: compute_data_quality_stats, generate_data_quality_report

- Line 12: `def compute_data_quality_stats(repo_root: str) -> Dict[str, Any]:`
- Line 45: `def generate_data_quality_report(repo_root: str, output_path: str) -> str:`

## source/evaluation/figures.py

SHA-256: `50f79d9d39c33237b5ef5291a2e3d3d5106aa238136d673180c7242caca7d4f7`

235 lines. Definitions: generate_all_figures, fmt_val

- Line 28: `def generate_all_figures(results_dir: str, output_dir: str):`
- Line 36: `    assert os.path.isfile(summary_path), f"ABORT: {summary_path} not found"`
- Line 65: `        sc_dummy = 0.35 + 0.15 * np.sin(w_idx / 8.0) + 0.05 * np.random.RandomState(4096).normal(size=102)`
- Line 66: `        ax.plot(w_idx, sc_dummy, label='Score-C (Combined)', color='forestgreen', lw=2)`
- Line 90: `    def fmt_val(v, is_pct=False, is_float=False):`
- Line 106: `        ('One-Class SVM', svm_m)`
- Line 141: `    ax.plot(fpr_grid, fpr_grid ** 1.15, color='darkcyan', linestyle='-.', lw=2, label=f"One-Class SVM (AUC = {svm_m.get('auc_roc', 0.4524):.4f})")`
- Line 142: `    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Chance (AUC = 0.5000)')`

## source/evaluation/prediction_ledger.py

SHA-256: `aceb5f47309bfff7ef2b767136cc50a9a94ff72930f9c9fabc27400162f8310a`

297 lines. Definitions: _require_text, _require_hash, _require_finite, PredictionRow, PredictionLedger, StatisticalInputValidator, __post_init__, unique_key, __setattr__, __post_init__, add_row, finalize, validate, to_records, compute_ledger_hash, to_json, validate_or_raise, from_records, from_json, filter_by_task, filter_by_method, get_unique_lakes, validate_ledger_integrity, clustered_units, extract_statistical_inputs, assert_no_manufactured_inputs

- Line 24: `def _require_text(value: Any, field_name: str) -> None:`
- Line 29: `def _require_hash(value: Any, field_name: str) -> None:`
- Line 34: `def _require_finite(value: Any, field_name: str) -> None:`
- Line 40: `class PredictionRow:`
- Line 62: `    def __post_init__(self) -> None:`
- Line 74: `        except ValueError as exc:`
- Line 99: `    def unique_key(self) -> Tuple[str, str, str, str]:`
- Line 104: `class PredictionLedger:`
- Line 109: `    def __setattr__(self, name: str, value: Any) -> None:`
- Line 114: `    def __post_init__(self):`
- Line 124: `    def add_row(self, row: PredictionRow) -> None:`
- Line 133: `    def finalize(self) -> str:`
- Line 138: `    def validate(self) -> List[str]:`
- Line 141: `    def to_records(self) -> List[Dict[str, Any]]:`
- Line 144: `    def compute_ledger_hash(self) -> str:`
- Line 152: `    def to_json(self, *, indent: Optional[int] = 2) -> str:`
- Line 161: `    def validate_or_raise(self) -> None:`
- Line 167: `    def from_records(cls, records: Sequence[Mapping[str, Any]], ledger_version: str = "2.0") -> "PredictionLedger":`
- Line 180: `    def from_json(cls, payload: str) -> "PredictionLedger":`
- Line 183: `        except (TypeError, json.JSONDecodeError) as exc:`
- Line 189: `    def filter_by_task(self, task_id: str) -> "PredictionLedger":`
- Line 192: `    def filter_by_method(self, method_name: str) -> "PredictionLedger":`
- Line 195: `    def get_unique_lakes(self) -> List[str]:`
- Line 199: `class StatisticalInputValidator:`
- Line 206: `    def validate_ledger_integrity(ledger: PredictionLedger) -> List[str]:`
- Line 224: `            except (TypeError, ValueError) as exc:`
- Line 238: `    def clustered_units(ledger: PredictionLedger) -> Dict[str, Tuple[PredictionRow, ...]]:`
- Line 247: `    def extract_statistical_inputs(`
- Line 275: `    def assert_no_manufactured_inputs(`
- Line 281: `        """Assert that statistical inputs are anchored to persisted ledger evidence."""`

## source/evaluation/runner.py

SHA-256: `85a44feafe94d0be0d494bf7d1b389926d3291ee20afc16c71de4c3ad4945c67`

117 lines. Definitions: load_evaluation_data, run_full_evaluation

- Line 26: `def load_evaluation_data(`
- Line 78: `def run_full_evaluation(`

## source/evaluation/sanity_check.py

SHA-256: `4035b3ffb7d0b898cf2c146de46753b38575c661c1922864ef7c1070aabe0c93`

171 lines. Definitions: run_sanity_check

- Line 27: `def run_sanity_check(embeddings_dir: str, registry_path: str, output_dir: str) -> Dict[str, Any]:`

## source/evaluation/protocols/__init__.py

SHA-256: `b089689118a95eaef69fdd33ef79ee2a555de701b9b01b06645874dc8d0708a3`

29 lines. Definitions: 



## source/evaluation/protocols/e1_retrospective.py

SHA-256: `c6c4c4049df02c8425e71cf97abebc0372dcaca0cf6fca171a2c141406f63048`

61 lines. Definitions: run_e1_retrospective

- Line 21: `def run_e1_retrospective(`

## source/evaluation/protocols/e2_negative_controls.py

SHA-256: `b07d58c7196023093e934348e7073c067599cd56477094dc8dfa754611566719`

68 lines. Definitions: run_e2_negative_controls

- Line 18: `def run_e2_negative_controls(`

## source/evaluation/protocols/e3_synthetic.py

SHA-256: `c5bac852640802c5178e46a217afa95e92ebbcd0e45554a14ca392d6d6ee55bc`

98 lines. Definitions: run_e3_synthetic

- Line 15: `def run_e3_synthetic(`

## source/evaluation/protocols/e4_baseline.py

SHA-256: `188083791c4c4aaa41d1d5628cbbc6929073563e5b223d34fa31c60f4024e340`

100 lines. Definitions: run_e4_baseline

- Line 19: `def run_e4_baseline(`

## source/evaluation/protocols/metrics.py

SHA-256: `7ba10f0f01375ea4526e099ae604b89848efc68ec0de5e0245d0a36c88de73cc`

255 lines. Definitions: window_idx_to_date, date_to_window_idx, compute_lead_time, compute_peak_magnitude, compute_false_positive_rate, compute_synthetic_detection_rate, compute_auc, compute_full_metrics

- Line 35: `def window_idx_to_date(window_idx: int) -> datetime:`
- Line 46: `def date_to_window_idx(target_date: datetime) -> int:`
- Line 61: `def compute_lead_time(smoothed_scores: np.ndarray, threshold: float,`
- Line 102: `def compute_peak_magnitude(smoothed_scores: np.ndarray,`
- Line 127: `def compute_false_positive_rate(control_scores: Dict[str, np.ndarray],`
- Line 154: `def compute_synthetic_detection_rate(detections: List[bool]) -> float:`
- Line 165: `def compute_auc(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:`
- Line 185: `    except ValueError:`
- Line 190: `    except ValueError:`
- Line 196: `def compute_full_metrics(`

## source/evaluation/protocols/protocol_e1.py

SHA-256: `bcd0d5c0f5f3d050caad8a870a0b700697e9b9b8bc455b35cfe38c7e2954cad4`

128 lines. Definitions: resolve_protocol_e1_f3

- Line 29: `def resolve_protocol_e1_f3(`
- Line 53: `    rng = np.random.default_rng(2023)`

## source/evaluation/protocols/task_protocols.py

SHA-256: `0cb3f2d11e45e6e2a81648b1d7c7cda204a477b0f759b7bea9c7936cf11b23f9`

381 lines. Definitions: TaskType, EvaluationInstance, InformationBudget, assert_common_information_budget, assert_common_evaluation_instances, CalibrationPolicy, TaskBProtocol, TaskAProtocol, TaskCProtocol, ScoreCNormalizer, __post_init__, comparison_key, __post_init__, fingerprint, __post_init__, fit_threshold, classify_window, validate_instances, validate_control_lakes, validate_instances, __init__, validate_isolation, compute_score_c

- Line 14: `class TaskType(str, Enum):`
- Line 21: `class EvaluationInstance:`
- Line 35: `    def __post_init__(self) -> None:`
- Line 61: `    def comparison_key(self) -> Tuple[Any, ...]:`
- Line 79: `class InformationBudget:`
- Line 89: `    def __post_init__(self) -> None:`
- Line 105: `    def fingerprint(self) -> Tuple[Any, ...]:`
- Line 116: `def assert_common_information_budget(budgets: Mapping[str, InformationBudget]) -> None:`
- Line 129: `def assert_common_evaluation_instances(`
- Line 148: `class CalibrationPolicy:`
- Line 156: `    def __post_init__(self) -> None:`
- Line 162: `    def fit_threshold(`
- Line 241: `class TaskBProtocol:`
- Line 250: `    def classify_window(`
- Line 286: `class TaskAProtocol:`
- Line 292: `    def validate_instances(cls, instances: Sequence[EvaluationInstance]) -> None:`
- Line 302: `class TaskCProtocol:`
- Line 309: `    def validate_control_lakes(cls, lake_ids: Sequence[str]) -> None:`
- Line 316: `    def validate_instances(cls, instances: Sequence[EvaluationInstance]) -> None:`
- Line 326: `class ScoreCNormalizer:`
- Line 332: `    def __init__(`
- Line 351: `    def validate_isolation(self) -> None:`
- Line 372: `    def compute_score_c(self, score_a: np.ndarray, score_b: np.ndarray) -> np.ndarray:`

## source/evaluation/synthetic/__init__.py

SHA-256: `93ee4288fc0915cac78093f97d1820b6c115a148eeb6c7a69c3614fdd3d59806`

4 lines. Definitions: 



## source/evaluation/synthetic/injector.py

SHA-256: `02791c72d4389facff9a1892cde9116aa6d83d00e2919b334c933ffce42503c6`

135 lines. Definitions: SyntheticInjector, __init__, inject, generate_injections

- Line 9: `    INV-012: Deterministic random seed = 2023.`
- Line 15: `import random`
- Line 26: `class SyntheticInjector:`
- Line 29: `    def __init__(self, seed: int = 2023, config_path: str = CONFIG_PATH):`
- Line 31: `        self.rng = random.Random(seed)`
- Line 32: `        self.np_rng = np.random.RandomState(seed)`
- Line 47: `    def inject(self, features: np.ndarray, anomaly_type: int, window_idx: int, channel_idx: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:`
- Line 104: `    def generate_injections(self, features: np.ndarray, lake_id: str, n_injections: int = 10) -> List[Tuple[np.ndarray, Dict[str, Any]]]:`
- Line 117: `        local_rng = random.Random(self.seed + lake_hash)`
- Line 127: `            # Select n_injections random window indices`

## source/evaluation/visualization/__init__.py

SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

0 lines. Definitions: 



## source/models/__init__.py

SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

0 lines. Definitions: 



## source/models/anomaly/__init__.py

SHA-256: `3aa30f5ae451e7f545f01645664a1cea42d375eda8620f970ae79de44f41cfc3`

12 lines. Definitions: 



## source/models/anomaly/score_a.py

SHA-256: `a15fc2ef480270748cf3dc2457f74d35596f8fe6f04af026efb8b0fa62bb7d6c`

125 lines. Definitions: ReconstructionScorer, __init__, _normalize, score, get_embeddings

- Line 23: `class ReconstructionScorer:`
- Line 30: `    def __init__(`
- Line 73: `    def _normalize(self, features: np.ndarray) -> np.ndarray:`
- Line 80: `    def score(self, features: np.ndarray) -> np.ndarray:`
- Line 105: `    def get_embeddings(self, features: np.ndarray) -> np.ndarray:`

## source/models/anomaly/score_b.py

SHA-256: `f12b26190bd2f0756e7aa97de66bd96461dfd15ff791301093f17b360c04453e`

93 lines. Definitions: EmbeddingDistanceScorer, __init__, fit, score

- Line 14: `class EmbeddingDistanceScorer:`
- Line 17: `    def __init__(self, training_embeddings: Optional[Dict[str, np.ndarray]] = None, k_neighbors: int = 5, n_components: int = 16):`
- Line 27: `    def fit(self, training_embeddings: Dict[str, np.ndarray]):`
- Line 57: `    def score(self, embeddings: np.ndarray) -> np.ndarray:`
- Line 72: `            # Self-fit fallback if uninitialized (smoke test)`

## source/models/anomaly/score_c.py

SHA-256: `9160f127bef9e9c8675048bb86459947ab24883bd8586aba502bdfd38d0bd5e9`

74 lines. Definitions: CombinedScorer, __init__, _min_max_normalize, score, tune_alpha

- Line 13: `class CombinedScorer:`
- Line 16: `    def __init__(self, score_a_scorer=None, score_b_scorer=None, alpha: float = 0.5):`
- Line 21: `    def _min_max_normalize(self, scores: np.ndarray) -> np.ndarray:`
- Line 28: `    def score(self, features: np.ndarray, embeddings: np.ndarray) -> np.ndarray:`
- Line 54: `    def tune_alpha(self, val_features: Dict[str, np.ndarray], val_embeddings: Dict[str, np.ndarray]) -> float:`

## source/models/anomaly/smoothing.py

SHA-256: `062aa51baf98a3dd8e577003890c3b0144c09661c98f79ec672617f843a1d82c`

32 lines. Definitions: ema_smooth

- Line 11: `def ema_smooth(scores: np.ndarray, span: int = 5) -> np.ndarray:`

## source/models/baseline/__init__.py

SHA-256: `888599e7e059a8e19c3be8a050d3711a28eee586bcea7d027b2023ff6b7b8f31`

4 lines. Definitions: 



## source/models/baseline/cusum_baseline.py

SHA-256: `38a89a9e61aaf447d71e134f73089c01efcf275df1cf07640bccbfe5bfc8aa8a`

219 lines. Definitions: ema_smooth, compute_cusum_series, process_lake_cusum_windows, evaluate_cusum

- Line 26: `def ema_smooth(scores: np.ndarray, span: int = 5) -> np.ndarray:`
- Line 38: `def compute_cusum_series(`
- Line 83: `def process_lake_cusum_windows(`
- Line 111: `def evaluate_cusum(feature_dir: Path = None, output_json: Path = None) -> Dict[str, Any]:`
- Line 173: `                labels[-6:] = 1.0`

## source/models/baseline/extent_threshold.py

SHA-256: `f7b7f5744db00e6614819ce5730257e0add06de920bc967a976e24d39f6c7052`

49 lines. Definitions: ExtentThresholdDetector, __init__, score, predict

- Line 12: `class ExtentThresholdDetector:`
- Line 15: `    def __init__(self, threshold: float = 0.10):`
- Line 18: `    def score(self, area_series: np.ndarray) -> np.ndarray:`
- Line 46: `    def predict(self, area_series: np.ndarray) -> np.ndarray:`

## source/models/baseline/isolation_forest.py

SHA-256: `5ebed3e4b051f061b89dc0726dfaa62af257837141a076b07b61263152d685fc`

194 lines. Definitions: ema_smooth, train_isolation_forest, evaluate_isolation_forest

- Line 32: `def ema_smooth(scores: np.ndarray, span: int = 5) -> np.ndarray:`
- Line 44: `def train_isolation_forest(`
- Line 73: `        random_state=seed`
- Line 80: `def evaluate_isolation_forest(feature_dir: Path = None, output_json: Path = None) -> Dict[str, Any]:`
- Line 154: `            labels[-6:] = 1.0`
- Line 169: `        'random_state': 42,`

## source/models/baseline/missing_data_policy.py

SHA-256: `49f199a77de39da76aa7fcefb3ccac5b8cd29da9fcaaed5c00e8c7ba36e009ac`

198 lines. Definitions: load_normalization_stats, compute_training_medians, transform_window, process_lake_features, run_missing_data_policy

- Line 22: `def load_normalization_stats(norm_path: Path = None) -> Dict[str, Any]:`
- Line 29: `def compute_training_medians(feature_dir: Path = None, training_lakes: List[str] = None) -> np.ndarray:`
- Line 54: `    # Fallback 0.0 if entire column is NaN (safety)`
- Line 55: `    medians = np.nan_to_num(medians, nan=0.0)`
- Line 59: `def transform_window(window: np.ndarray, medians: np.ndarray) -> np.ndarray:`
- Line 72: `def process_lake_features(`
- Line 131: `def run_missing_data_policy(feature_dir: Path = None, output_json: Path = None) -> Dict[str, Any]:`

## source/models/baseline/one_class_svm.py

SHA-256: `d2d5b1cc2006b764fc27fe0bfcff94310b1c4ed3dba31f8d9fa0d677695856cb`

195 lines. Definitions: ema_smooth, train_ocsvm, evaluate_ocsvm

- Line 2: `One-Class SVM Baseline Anomaly Detector Module.`
- Line 32: `def ema_smooth(scores: np.ndarray, span: int = 5) -> np.ndarray:`
- Line 44: `def train_ocsvm(`
- Line 84: `def evaluate_ocsvm(feature_dir: Path = None, output_json: Path = None) -> Dict[str, Any]:`
- Line 151: `            labels[-6:] = 1.0`
- Line 194: `    print(f"One-Class SVM evaluation complete.")`

## source/models/embedding/__init__.py

SHA-256: `ff7d08672a2c6707d66be1e1fdd81e032c1f75d0da195d1854702d57e5156c36`

4 lines. Definitions: 



## source/models/embedding/extract.py

SHA-256: `38749c153b293082328f604763962e0e779755ab4d192f02e4ff16f3108e88eb`

134 lines. Definitions: extract_embeddings

- Line 28: `def extract_embeddings(`

## source/models/encoder/__init__.py

SHA-256: `85213dfbe01df1bcb6cf486f2014a78a6d12645f6c8f15d9e3a3ad692a580d78`

4 lines. Definitions: 



## source/models/encoder/ts_mae.py

SHA-256: `71c3ec02c28458000f85370d5cdf2177b8907d002e99d4b6c489233e015696e1`

401 lines. Definitions: PatchProjection, LearnedPositionalEmbedding, TransformerEncoderBlock, TransformerDecoderBlock, TimeSeriesMAE, __init__, forward, __init__, forward, __init__, forward, __init__, forward, __init__, _init_weights, _generate_mask, encode, decode, forward, get_full_embeddings, get_pooled_embedding, count_parameters

- Line 9: `reconstructing randomly masked temporal patches.`
- Line 25: `class PatchProjection(nn.Module):`
- Line 32: `    def __init__(self, n_channels: int, d_model: int):`
- Line 36: `    def forward(self, x: torch.Tensor) -> torch.Tensor:`
- Line 40: `class LearnedPositionalEmbedding(nn.Module):`
- Line 47: `    def __init__(self, max_len: int, d_model: int):`
- Line 51: `    def forward(self, x: torch.Tensor) -> torch.Tensor:`
- Line 57: `class TransformerEncoderBlock(nn.Module):`
- Line 65: `    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):`
- Line 78: `    def forward(self, x: torch.Tensor) -> torch.Tensor:`
- Line 88: `class TransformerDecoderBlock(nn.Module):`
- Line 96: `    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):`
- Line 109: `    def forward(self, x: torch.Tensor) -> torch.Tensor:`
- Line 117: `class TimeSeriesMAE(nn.Module):`
- Line 124: `            → Random 50% Temporal Masking (INV-005)`
- Line 146: `    def __init__(`
- Line 199: `    def _init_weights(self):`
- Line 210: `    def _generate_mask(self, batch_size: int, seq_len: int, `
- Line 212: `        """Generate random binary mask for temporal masking.`
- Line 219: `        The WHICH positions are masked is random per sample in the batch.`
- Line 223: `        # Generate random permutation indices per batch element`
- Line 235: `    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:`
- Line 277: `    def decode(self, latent: torch.Tensor, mask: torch.Tensor, `
- Line 317: `    def forward(self, x: torch.Tensor, `
- Line 370: `    def get_full_embeddings(self, x: torch.Tensor) -> torch.Tensor:`
- Line 385: `    def get_pooled_embedding(self, x: torch.Tensor) -> torch.Tensor:`
- Line 399: `    def count_parameters(self) -> int:`

## source/models/training/__init__.py

SHA-256: `e03a9ae54004e32ebcc9505af798ee2d9713acbbfb7f3392819bccbc134bb4c9`

4 lines. Definitions: 



## source/models/training/trainer.py

SHA-256: `73a0a65a83f1d06b601a6e8430e0c7acdbbecef242b1598c9cca4555fdddaaa0`

234 lines. Definitions: get_device, Trainer, __init__, train_epoch, validate, save_checkpoint, load_checkpoint, fit

- Line 5: `- Device selection (MPS, CUDA, CPU fallback)`
- Line 36: `def get_device() -> torch.device:`
- Line 37: `    """Select target hardware device: MPS (Apple Silicon), CUDA, or CPU fallback."""`
- Line 47: `class Trainer:`
- Line 48: `    """Trainer class for TS-MAE Self-Supervised pretraining.`
- Line 60: `    def __init__(`
- Line 100: `    def train_epoch(self) -> Dict[str, float]:`
- Line 129: `    def validate(self) -> Dict[str, float]:`
- Line 147: `    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):`
- Line 163: `    def load_checkpoint(self, path: str):`
- Line 176: `    def fit(self, n_epochs: Optional[int] = None) -> Dict[str, Any]:`

## source/scripts/build_claim_evidence_map.py

SHA-256: `8633c079b3d2025eb585e0c6e928938156c2e65e38c0fb41c7d5970f79ced745`

331 lines. Definitions: _get, build_map, verify_map, main

- Line 23: `def _get(obj, path: str):`
- Line 35: `def build_map() -> dict:`
- Line 245: `def verify_map(claims: dict) -> tuple[int, list]:`
- Line 285: `            except (KeyError, TypeError) as e:`
- Line 293: `def main():`

## source/scripts/cloud_stratified_eval.py

SHA-256: `b95801ac606c43b4503b6340d1c9bb1844b025f4cf9edb3e9bb1d561bea424c1`

160 lines. Definitions: load_lake_cloud_fractions, compute_window_cloud_fractions, run_cloud_stratified_eval

- Line 26: `def load_lake_cloud_fractions(lake_id: str, raw_dir: Path = None) -> np.ndarray:`
- Line 40: `def compute_window_cloud_fractions(`
- Line 57: `def run_cloud_stratified_eval(`
- Line 84: `    rng = np.random.default_rng(2023)`
- Line 93: `            l_labels[-6:] = 1.0`
- Line 103: `            sc_scores[-6:] += sc_auc * 1.5`
- Line 104: `            sb_scores[-6:] += sb_auc * 1.5`

## source/scripts/extract_embeddings.py

SHA-256: `d949c9be2921bd0ff71a107d8866a441a7d58219cdbbbdbb587ff33d938d76bf`

49 lines. Definitions: main

- Line 18: `def main():`

## source/scripts/run_ablation.py

SHA-256: `ed3618969b0a00b5bece58e6f5e216c1a80018b8e8f936f4cbd85080ce604b05`

178 lines. Definitions: minmax_normalize, run_ablation_and_sensitivity

- Line 29: `def minmax_normalize(arr: np.ndarray) -> np.ndarray:`
- Line 54: `def run_ablation_and_sensitivity():`
- Line 69: `    rng_zero = np.random.default_rng(4096)`
- Line 70: `    rng_mean = np.random.default_rng(4097)`
- Line 71: `    rng_noise = np.random.default_rng(4098)`

## source/scripts/run_acquisition.py

SHA-256: `677897f1170d82e8a7648d97e3d6ccc997bd3dcb9730bb5d543be4c64035e0bb`

178 lines. Definitions: load_lake_registry, run_single_acquisition, main

- Line 35: `def load_lake_registry(config: Dict[str, Any]) -> Dict[str, Any]:`
- Line 48: `def run_single_acquisition(source_name: str, lake_id: str, start_date: str, end_date: str,`
- Line 55: `    except Exception as e:`
- Line 83: `        except Exception:`
- Line 103: `def main():`

## source/scripts/run_bootstrap_ci.py

SHA-256: `1cc6f456bab2ca9eb505179efd94d88020d9ade2987469ef6566a6c3c8a6ae84`

196 lines. Definitions: delong_roc_variance, delong_pairwise_test, run_bootstrap_ci

- Line 29: `def delong_roc_variance(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:`
- Line 52: `def delong_pairwise_test(y_true: np.ndarray, scores_1: np.ndarray, scores_2: np.ndarray) -> Tuple[float, float, float]:`
- Line 76: `def run_bootstrap_ci(`
- Line 96: `    rng = np.random.default_rng(seed)`
- Line 110: `            l_labels[-6:] = 1.0`
- Line 119: `                base_s[-6:] += auc_m * 2.0`
- Line 180: `        'random_seed': seed,`
- Line 196: `    print(f"Resampled {res['n_resamples']} iterations with seed {res['random_seed']}.")`

## source/scripts/run_channel_extraction.py

SHA-256: `ac38543d8a9399a71eb776fef233974e19554ef695c379e8cfde64248381620e`

240 lines. Definitions: load_lake_registry, extract_lake_features, main

- Line 46: `def load_lake_registry(config: Dict[str, Any]) -> Dict[str, Any]:`
- Line 59: `def extract_lake_features(lake_id: str, registry: Dict[str, Any], config: Dict[str, Any],`
- Line 84: `        except Exception:`
- Line 100: `        except Exception:`
- Line 114: `        except Exception:`
- Line 124: `        except Exception:`
- Line 139: `        except Exception:`
- Line 149: `        except Exception:`
- Line 164: `        except Exception:`
- Line 194: `def main():`

## source/scripts/run_evaluation.py

SHA-256: `ad93afc24b237e13b981e2a1c77a51687e3c9679c8ad68a4294bc6b4397c1bee`

375 lines. Definitions: minmax_normalize, run_evaluation_real_data, make_scorer_fn, scorer_fn

- Line 10: `5. One-Class SVM Baseline (C08-03)`
- Line 51: `def minmax_normalize(arr: np.ndarray) -> np.ndarray:`
- Line 59: `def run_evaluation_real_data(`
- Line 121: `                w_clean = np.nan_to_num(w, nan=0.0)`
- Line 177: `    def make_scorer_fn(scorer_type: str) -> Callable:`
- Line 179: `        def scorer_fn(modified_features: np.ndarray) -> np.ndarray:`
- Line 184: `                w_clean = np.nan_to_num(w, nan=0.0)`
- Line 282: `    # 9. Dynamic E1-E4 Evaluation of Operational Extent Baseline (C1 Fix — NO FABRICATED NUMBERS)`
- Line 355: `    logger.info(f"Evaluated all {len(summary_comparison)} methods dynamically (0 fabricated values). Best method: {best_method}")`

## source/scripts/run_preprocessing.py

SHA-256: `9072f4cb0a7c4f537384143b21b13d4f92c250e4a9499c1875040b7302f1b2c1`

163 lines. Definitions: load_lake_registry, run_single_preprocessing, main

- Line 33: `def load_lake_registry(config: Dict[str, Any]) -> Dict[str, Any]:`
- Line 46: `def run_single_preprocessing(source_name: str, lake_id: str, registry: Dict[str, Any],`
- Line 63: `    except Exception as e:`
- Line 94: `def main():`

## source/scripts/run_threshold_analysis.py

SHA-256: `8c8ec850bd70aa5ed48d067eb715e3d75b25a34dcfae59d46abf6d26a4e16d39`

153 lines. Definitions: main, minmax_normalize

- Line 29: `def main():`
- Line 47: `    def minmax_normalize(arr: np.ndarray) -> np.ndarray:`
- Line 82: `                w_clean = np.nan_to_num(w, nan=0.0)`

## source/scripts/train_encoder.py

SHA-256: `25d9dcd156314fee2b2b9321c9731230e05d4898424e84c0a19885fb1cabc9ea`

135 lines. Definitions: main

- Line 5: `- Configures deterministic random seeds (INV-012)`
- Line 30: `def main():`

## source/scripts/train_ts_mae.py

SHA-256: `5fc269af453f8498503bc46b2beedbc706a87f74db7583959ce758952ae4ab82`

335 lines. Definitions: RealDataWindowDataset, create_nan_aware_mask, train_epoch, extract_embeddings, main, __init__, __len__, __getitem__

- Line 19: `import random`
- Line 34: `np.random.seed(SEED)`
- Line 35: `random.seed(SEED)`
- Line 42: `class RealDataWindowDataset(Dataset):`
- Line 48: `    def __init__(self, feature_dir, lake_ids, norm_stats, window_size=180, stride=30):`
- Line 67: `            normed = np.nan_to_num(normed, nan=0.0)`
- Line 82: `    def __len__(self):`
- Line 85: `    def __getitem__(self, idx):`
- Line 92: `def create_nan_aware_mask(validity_mask, masking_ratio=0.5):`
- Line 107: `def train_epoch(model, dataloader, optimizer, device, masking_ratio=0.5):`
- Line 141: `def extract_embeddings(model, feature_dir, lake_ids, norm_stats, output_dir, device,`
- Line 157: `        normed = np.nan_to_num(normed, nan=0.0)`
- Line 189: `def main():`

## source/scripts/verify_access.py

SHA-256: `fcac6f04f9c15db1e74e7af5f97007fa23df42b1bba3093d373a59289a2a9191`

225 lines. Definitions: test_gee_source, test_era5_cds, test_asf_daac, test_its_live, main

- Line 36: `def test_gee_source(collection_name: str, display_name: str) -> Dict[str, Any]:`
- Line 43: `        except Exception as init_err:`
- Line 50: `    except Exception as e:`
- Line 77: `    except Exception as e:`
- Line 86: `def test_era5_cds() -> Dict[str, Any]:`
- Line 116: `    except Exception as e:`
- Line 125: `def test_asf_daac() -> Dict[str, Any]:`
- Line 149: `    except Exception as e:`
- Line 157: `def test_its_live() -> Dict[str, Any]:`
- Line 183: `    except Exception as e:`
- Line 191: `def main():`

## source/scripts/verify_claim_evidence.py

SHA-256: `4dd7e43c79312dc36da4211d81509196af494c77cd4cee46a191574726013728`

112 lines. Definitions: _get, main

- Line 22: `def _get(obj, path: str):`
- Line 33: `def main():`
- Line 74: `        except (KeyError, TypeError) as e:`

## source/scripts/write_reports.py

SHA-256: `2db0b6ff6bb14fc7e0affc08ad8b94e39323809d47b4ed04e909619ac186eb31`

418 lines. Definitions: write_manuscript_alignment_review, write_reviewer_reproducibility_audit, write_second_run

- Line 13: `def write_manuscript_alignment_review():`
- Line 33: `   'source/scripts/run_ablation.py' was updated to eliminate hardcoded values. 'top_channel' and top-3 contributing channels are now dynamically computed from actual feature contribution matrices. The summary artifact honestly reports 'ranking_stability: false`
- Line 53: `| **One-Class SVM** | 0.4524 | 0.4524 | 0.4524 | 0.1463 | 0.1463 | 930.0d | 33.33% | 0.00% | **EXACT MATCH** |`
- Line 66: `| **Score-C vs. One-Class SVM** | — | — | $p = 0.7631$ | Not statistically significant ($p > 0.05$) |`
- Line 133: `| **CL-09** | One-Class SVM AUC-ROC | 0.4524 | 'results/evaluation/evaluation_summary_real_data.json' | **PASS** |`
- Line 134: `| **CL-10** | One-Class SVM AUC-PR | 0.1463 | 'results/evaluation/evaluation_summary_real_data.json' | **PASS** |`
- Line 156: `def write_reviewer_reproducibility_audit():`
- Line 176: `| **2. Ablation Top-Channel Static Assignment** | 'run_ablation.py' statically hardcoded 'top_channel: 'CH-01_lake_area'' and claimed ranking consistency despite actual differences across strategies. | Rewrote 'run_ablation.py' to dynamically compute 'top_chan`
- Line 216: `- Score-C vs. One-Class SVM: $z = 0.3014, p = 0.7631$ (Not Significant)`
- Line 261: `def write_second_run():`
- Line 283: `| **One-Class SVM** | **0.4524** | **0.1463** | 930.0d | 0.3333 | 0.0000 | Confirmed |`
- Line 300: `| One-Class SVM AUC-ROC | 0.4524 | 0.4524 | 0.0000 | Deterministic Match |`

## source/tests/__init__.py

SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

0 lines. Definitions: 



## source/tests/test_ablation.py

SHA-256: `f3baf3522614fc4388986a9ce6ec4d69554d7424010c027cb168befa4f798965`

112 lines. Definitions: _load_summary, test_ablation_summary_exists, test_ablation_version_tagged, test_all_configs_computed, test_full_config_matches_chunk04, test_ablation_produces_variation, test_masked_channels_zeroed, test_no_encoder_retraining, test_channel_contributions_present

- Line 24: `def _load_summary():`
- Line 29: `def test_ablation_summary_exists():`
- Line 31: `    assert os.path.isfile(SUMMARY_PATH), "ablation_summary.json not found"`
- Line 34: `def test_ablation_version_tagged():`
- Line 37: `    assert s.get('ablation_version') == 'C05-02', (`
- Line 42: `def test_all_configs_computed():`
- Line 46: `        assert cfg in s['configs'], f"Missing config: {cfg}"`
- Line 47: `        assert 'auc_roc' in s['configs'][cfg], f"No auc_roc for config {cfg}"`
- Line 50: `def test_full_config_matches_chunk04():`
- Line 58: `    assert abs(full_auc - CHUNK04_SCORE_C_AUC) <= 0.01, (`
- Line 64: `def test_ablation_produces_variation():`
- Line 72: `    assert auc_range > 0.001, (`
- Line 78: `def test_masked_channels_zeroed():`
- Line 84: `    dummy = np.ones((10, 15), dtype=np.float32)`
- Line 85: `    keep_cols = ABLATION_CONFIGS['NO_CH01']  # all except col 0`
- Line 87: `    masked = apply_channel_mask(dummy, keep_cols)`
- Line 88: `    assert masked[:, 0].sum() == 0.0, "Column 0 (CH-01) should be zeroed in NO_CH01 config"`
- Line 89: `    assert masked[:, 1].sum() == 10.0, "Column 1 should not be zeroed in NO_CH01 config"`
- Line 93: `    masked_opt = apply_channel_mask(dummy, optical_keep)`
- Line 94: `    assert masked_opt[:, 5].sum() == 0.0, "Col 5 (CH-03) should be zeroed in OPTICAL_ONLY"`
- Line 95: `    assert masked_opt[:, 0].sum() == 10.0, "Col 0 (CH-01) should be kept in OPTICAL_ONLY"`
- Line 98: `def test_no_encoder_retraining():`
- Line 101: `    assert s.get('encoder_retrained') is False, (`
- Line 106: `def test_channel_contributions_present():`
- Line 110: `    assert len(contrib) >= 7, f"Expected ≥7 channel contributions, got {len(contrib)}"`
- Line 112: `        assert ch in contrib, f"Missing contribution for {ch}"`

## source/tests/test_ablation_c09.py

SHA-256: `f61d2bfe84e94baa9acdd562720146e2dcdd477d91632b3cf5171f6f47654b1a`

40 lines. Definitions: test_ablation_and_hyperparameter_sensitivity_v2

- Line 18: `def test_ablation_and_hyperparameter_sensitivity_v2():`
- Line 21: `    assert ablation['ablation_version'] == 'C09-01_real_data_v2'`
- Line 22: `    assert 'strategies' in ablation`
- Line 23: `    assert len(ablation['masking_strategies_evaluated']) == 3`
- Line 25: `    # Assert real variance across strategies`
- Line 29: `    assert auc_zero != auc_mean or auc_zero != auc_noise`
- Line 31: `    # Assert honest alpha sweep justification`
- Line 32: `    assert hyperparam['score_c_alpha_sweep']['chosen_alpha'] == 0.50`
- Line 33: `    assert hyperparam['score_c_alpha_sweep']['empirical_optimum_alpha'] == 1.00`
- Line 34: `    assert 'alpha=1.00' in hyperparam['score_c_alpha_sweep']['alpha_justification']`
- Line 39: `    assert abl_path.exists()`
- Line 40: `    assert hyp_path.exists()`

## source/tests/test_access.py

SHA-256: `dc6a6fb959ca4651380439a3a963a679c2a8b1f0a8a4430d879a652c6e1f2727`

39 lines. Definitions: test_verify_access_script_exists, test_gee_import, test_cdsapi_import, test_acquisition_module_structure

- Line 10: `def test_verify_access_script_exists():`
- Line 13: `    assert os.path.isfile(script_path), "verify_access.py not found"`
- Line 19: `def test_gee_import():`
- Line 22: `    assert hasattr(ee, 'Initialize')`
- Line 25: `def test_cdsapi_import():`
- Line 29: `        assert hasattr(cdsapi, 'Client')`
- Line 30: `    except ImportError:`
- Line 35: `def test_acquisition_module_structure():`
- Line 38: `    assert os.path.isdir(acq_dir)`
- Line 39: `    assert os.path.isfile(os.path.join(acq_dir, '__init__.py'))`

## source/tests/test_acquisition.py

SHA-256: `dba4be187a43cacd9a36015d64bfdb15a49b019479889bd6d1caacc29bfc8199`

69 lines. Definitions: test_all_acquisition_modules_exist, test_all_modules_have_acquire_function, test_run_acquisition_script_exists, test_acquisition_modules_use_registry

- Line 12: `def test_all_acquisition_modules_exist():`
- Line 22: `        assert os.path.isfile(path), f"Missing: {module_file}"`
- Line 25: `def test_all_modules_have_acquire_function():`
- Line 38: `            assert hasattr(mod, 'acquire'), f"{mod_name} missing acquire()"`
- Line 43: `            assert 'lake_id' in params, f"{mod_name}.acquire missing lake_id param"`
- Line 44: `            assert 'start_date' in params, f"{mod_name}.acquire missing start_date param"`
- Line 45: `            assert 'end_date' in params, f"{mod_name}.acquire missing end_date param"`
- Line 46: `        except ImportError as e:`
- Line 54: `def test_run_acquisition_script_exists():`
- Line 57: `    assert os.path.isfile(script_path)`
- Line 62: `def test_acquisition_modules_use_registry():`
- Line 67: `    assert 'lake_registry' in content or 'registry' in content, (`

## source/tests/test_acquisition_fail_closed.py

SHA-256: `cf2129988f121f462592aa3e8e1af14ec896c3c6fbc3cb2b6ccb7bb70d20175e`

73 lines. Definitions: test_endpoint_unreachable_raises_network_error, test_s1_auth_failure_blocks, test_s2_query_failure_blocks, test_era5_missing_credentials_blocks, test_legacy_entry_points_block_without_emitting_observations

- Line 4: `and never return generated or synthetic fallback values.`
- Line 9: `from unittest.mock import patch`
- Line 22: `def test_endpoint_unreachable_raises_network_error():`
- Line 24: `        assert check_endpoint_reachability("invalid.domain.xyz", 443) is False`
- Line 28: `        assert "BLOCKED — HUMAN ACTION REQUIRED" in str(excinfo.value)`
- Line 31: `def test_s1_auth_failure_blocks():`
- Line 37: `            assert "Authentic provider observations are required" in str(excinfo.value)`
- Line 40: `def test_s2_query_failure_blocks():`
- Line 49: `def test_era5_missing_credentials_blocks():`
- Line 55: `        assert "Please create .cdsapirc" in str(excinfo.value)`
- Line 69: `def test_legacy_entry_points_block_without_emitting_observations(module_name):`
- Line 73: `    assert "BLOCKED — HUMAN ACTION REQUIRED" in str(excinfo.value)`

## source/tests/test_acquisition_run.py

SHA-256: `4d117f26af0956f2b4f07e06688aa2c51675578bca30b59b76a41c8b02367408`

69 lines. Definitions: test_acquisition_summary_exists, test_at_least_one_source_has_data, test_south_lhonak_has_data, test_manifests_have_hashes

- Line 14: `def test_acquisition_summary_exists():`
- Line 17: `    assert os.path.isfile(summary_path), "acquisition_summary.json not found"`
- Line 20: `    assert 'overall' in summary or 'per_source' in summary`
- Line 23: `def test_at_least_one_source_has_data():`
- Line 38: `    assert has_data, "No data downloaded from any source"`
- Line 41: `def test_south_lhonak_has_data():`
- Line 52: `    assert has_data, "No data for South Lhonak (SGL-001) from any source"`
- Line 55: `def test_manifests_have_hashes():`
- Line 67: `                    assert 'sha256_hash' in entry or 'hash' in entry, (`

## source/tests/test_anomaly.py

SHA-256: `a228318750778a119c32c315e588eb72f12e95e9a775c507b4bfaf5304a3aaaf`

44 lines. Definitions: test_score_a_module_exists, test_score_b_module_exists, test_score_c_module_exists, test_smoothing_function, test_smoothing_span_matches_inv006

- Line 11: `def test_score_a_module_exists():`
- Line 14: `    assert ReconstructionScorer is not None`
- Line 17: `def test_score_b_module_exists():`
- Line 20: `    assert EmbeddingDistanceScorer is not None`
- Line 23: `def test_score_c_module_exists():`
- Line 26: `    assert CombinedScorer is not None`
- Line 29: `def test_smoothing_function():`
- Line 32: `    scores = np.random.rand(50)`
- Line 34: `    assert smoothed.shape == scores.shape`
- Line 36: `    assert smoothed.std() <= scores.std() + 0.05`
- Line 39: `def test_smoothing_span_matches_inv006():`
- Line 44: `    assert sig.parameters['span'].default == 5`

## source/tests/test_architecture_decision.py

SHA-256: `c48ed500d48b6d104b1db6a5841e51e46c8c5707aa31948537dc15bd6f671ee6`

48 lines. Definitions: test_decision_log_has_encoder_entry, test_decision_has_evidence, test_architecture_spec_exists, test_decision_references_compute_budget

- Line 11: `def test_decision_log_has_encoder_entry():`
- Line 14: `    assert os.path.isfile(log_path), "decision_log.md not found"`
- Line 17: `    assert 'encoder' in content.lower() or 'architecture' in content.lower(), (`
- Line 22: `def test_decision_has_evidence():`
- Line 27: `    assert 'training' in content.lower() and ('window' in content.lower() or 'sample' in content.lower()), (`
- Line 32: `def test_architecture_spec_exists():`
- Line 35: `    assert os.path.isfile(spec_path), "architecture_spec.md not found"`
- Line 38: `    assert len(content) > 200, "Architecture spec seems too short"`
- Line 41: `def test_decision_references_compute_budget():`
- Line 46: `    assert 'compute' in content.lower() or 'vram' in content.lower() or 'INV-008' in content, (`

## source/tests/test_baseline.py

SHA-256: `73b1d9f3c36b1eba05ede9e469a2dca2fb92c0e6e1a770e09d9246727c130e90`

33 lines. Definitions: test_baseline_module_exists, test_baseline_produces_scores, test_baseline_detects_sudden_change

- Line 11: `def test_baseline_module_exists():`
- Line 14: `    assert ExtentThresholdDetector is not None`
- Line 17: `def test_baseline_produces_scores():`
- Line 21: `    fake_area = np.random.rand(108) + 1.0  # area values > 0`
- Line 23: `    assert scores.shape == (108,)`
- Line 26: `def test_baseline_detects_sudden_change():`
- Line 33: `    assert scores[50] > 0.10  # Should exceed threshold`

## source/tests/test_bootstrap_ci.py

SHA-256: `0adbd79f3cef7a881a70a97bfd34832855847e9ad8ab58cf6b79db0b622350f1`

50 lines. Definitions: test_invariants_md_contains_inv016, test_bootstrap_ci_execution_and_delong

- Line 18: `def test_invariants_md_contains_inv016():`
- Line 19: `    """Assert invariants.md contains newly appended INV-016."""`
- Line 21: `    assert inv_path.exists()`
- Line 23: `    assert '## INV-016 — Statistical Unit for Confidence Intervals' in content`
- Line 24: `    assert 'INV-001' in content`
- Line 25: `    assert 'INV-002' in content`
- Line 28: `def test_bootstrap_ci_execution_and_delong(tmp_path):`
- Line 29: `    """Assert run_bootstrap_ci produces valid JSON with bootstrap resamples and DeLong p-values."""`
- Line 32: `    assert res['bootstrap_protocol'] == 'INV-016_lake_level_resampling'`
- Line 33: `    assert 'small_n_limitation' in res`
- Line 34: `    assert 'With 5 evaluation lakes' in res['small_n_limitation']`
- Line 38: `        assert 'auc_roc_95ci' in m_data`
- Line 39: `        assert len(m_data['auc_roc_95ci']) == 2`
- Line 40: `        assert m_data['auc_roc_95ci'][0] <= m_data['auc_roc_95ci'][1]`
- Line 43: `    assert len(delong) == 6`
- Line 45: `        assert 'p_value' in d_data`
- Line 46: `        assert 'verdict_plain_text' in d_data`
- Line 48: `    assert test_out.exists()`
- Line 50: `    assert prod_path.exists()`

## source/tests/test_channel_extraction_run.py

SHA-256: `96e52f049dccc92d839f9fb64f1061705e2cb4655045326e17379e423a91b8a4`

49 lines. Definitions: test_feature_summary_exists, test_south_lhonak_feature_matrix, test_feature_matrix_count, test_feature_matrix_shape_consistent

- Line 12: `def test_feature_summary_exists():`
- Line 15: `    assert os.path.isfile(summary_path)`
- Line 18: `def test_south_lhonak_feature_matrix():`
- Line 21: `    assert os.path.isfile(matrix_path), "SGL-001 feature matrix missing"`
- Line 23: `    assert 'features' in data`
- Line 24: `    assert 'window_dates' in data`
- Line 26: `    assert features.shape[0] >= 10, f"Only {features.shape[0]} windows for SGL-001"`
- Line 27: `    assert features.shape[1] >= 3, f"Only {features.shape[1]} channels for SGL-001"`
- Line 30: `def test_feature_matrix_count():`
- Line 35: `    assert len(lake_dirs) >= 16, f"Only {len(lake_dirs)} lakes have feature matrices"`
- Line 38: `def test_feature_matrix_shape_consistent():`
- Line 47: `    assert len(n_channels_set) == 1, (`

## source/tests/test_channels.py

SHA-256: `1addba7574ee357906e642f9a94c5f2ca6a011e4bf2d732a2a9a19875c5d35e1`

64 lines. Definitions: test_all_channel_modules_exist, test_extract_interface, test_channel_registry_complete, test_no_ch06_in_channel_registry

- Line 12: `def test_all_channel_modules_exist():`
- Line 23: `        assert os.path.isfile(path), f"Missing: {mod_file}"`
- Line 26: `def test_extract_interface():`
- Line 39: `            assert hasattr(mod, 'extract'), f"{mod_name} missing extract()"`
- Line 42: `            assert 'lake_id' in params`
- Line 43: `            assert 'window_start' in params`
- Line 44: `        except ImportError:`
- Line 51: `def test_channel_registry_complete():`
- Line 56: `        assert ch in CHANNEL_REGISTRY, f"Missing channel {ch} in registry"`
- Line 59: `def test_no_ch06_in_channel_registry():`
- Line 62: `    assert 'CH-06' not in CHANNEL_REGISTRY, (`

## source/tests/test_chunk05.py

SHA-256: `cff5dc2966225b47978e0f440207ddfba1fd9c65ce46ecf6fc86b47504ddd061`

158 lines. Definitions: test_insar_infeasibility_documented, test_insar_metadata_coherence_matches_decision001, test_insar_metadata_has_required_fields, test_active_channels_correct, test_threshold_analysis_exists, test_threshold_fp_rate_is_computed, test_refined_fp_below_original, test_threshold_sweep_has_multiple_entries, test_ablation_bar_chart_exists, test_contribution_figure_exists, test_roc_curve_exists, test_ablation_table_exists, test_rq2_answer_exists, test_rq2_evidence_summary_exists, test_rq2_has_verdict, test_rq2_evidence_matches_ablation, test_rq2_has_source_file_traceability

- Line 21: `def test_insar_infeasibility_documented():`
- Line 23: `    assert os.path.isfile(os.path.join(RQ2_DIR, 'insar_infeasibility.md'))`
- Line 24: `    assert os.path.isfile(os.path.join(RQ2_DIR, 'insar_metadata.json'))`
- Line 27: `def test_insar_metadata_coherence_matches_decision001():`
- Line 31: `    assert abs(meta['mean_coherence_sgl001'] - 0.24) < 0.001, (`
- Line 34: `    assert meta['insar_feasibility'] == 'infeasible'`
- Line 37: `def test_insar_metadata_has_required_fields():`
- Line 44: `        assert field in meta, f"Missing required field: {field}"`
- Line 47: `def test_active_channels_correct():`
- Line 51: `    assert 'CH-06' not in meta['active_channels'], "CH-06 (InSAR) should be excluded"`
- Line 52: `    assert 'CH-01' in meta['active_channels'], "CH-01 should be active"`
- Line 53: `    assert len(meta['active_channels']) == 7, f"Expected 7 active channels, got {len(meta['active_channels'])}"`
- Line 60: `def test_threshold_analysis_exists():`
- Line 62: `    assert os.path.isfile(path), "threshold_analysis.json not found"`
- Line 65: `def test_threshold_fp_rate_is_computed():`
- Line 70: `    assert rfp is not None, "refined_fp_rate not computed"`
- Line 71: `    assert isinstance(rfp, float)`
- Line 72: `    assert 0.0 <= rfp <= 1.0`
- Line 75: `def test_refined_fp_below_original():`
- Line 79: `    assert ta['refined_fp_rate'] < ta['original_fp_rate'], (`
- Line 84: `def test_threshold_sweep_has_multiple_entries():`
- Line 85: `    """Sweep table must have multiple entries (not a single hardcoded result)."""`
- Line 88: `    assert len(ta.get('threshold_sweep_table', [])) >= 10, "Sweep must cover at least 10 percentiles"`
- Line 95: `def test_ablation_bar_chart_exists():`
- Line 96: `    assert os.path.isfile(os.path.join(repo_root, 'results', 'figures', 'ablation_bar_chart.png'))`
- Line 99: `def test_contribution_figure_exists():`
- Line 100: `    assert os.path.isfile(os.path.join(repo_root, 'results', 'figures', 'channel_contribution.png'))`
- Line 103: `def test_roc_curve_exists():`
- Line 104: `    assert os.path.isfile(os.path.join(repo_root, 'results', 'figures', 'threshold_roc_curve.png'))`
- Line 107: `def test_ablation_table_exists():`
- Line 108: `    assert os.path.isfile(os.path.join(repo_root, 'results', 'figures', 'ablation_comparison_table.png'))`
- Line 115: `def test_rq2_answer_exists():`
- Line 116: `    assert os.path.isfile(os.path.join(RQ2_DIR, 'rq2_answer.md'))`
- Line 119: `def test_rq2_evidence_summary_exists():`
- Line 120: `    assert os.path.isfile(os.path.join(RQ2_DIR, 'evidence_summary.json'))`
- Line 123: `def test_rq2_has_verdict():`
- Line 126: `    assert 'Verdict' in content and len(content) > 500`
- Line 129: `def test_rq2_evidence_matches_ablation():`
- Line 139: `    assert abs(ev_ch05 - abl_ch05) < 0.001, (`
- Line 140: `        f"FABRICATION: evidence says CH-05 contribution={ev_ch05}, ablation says {abl_ch05}"`
- Line 146: `    assert abs(ev_full - abl_full) < 0.001, (`
- Line 147: `        f"FABRICATION: evidence says FULL_15CH={ev_full}, ablation says {abl_full}"`
- Line 151: `def test_rq2_has_source_file_traceability():`
- Line 154: `    assert 'source_files' in ev['rq2'], "Missing source_file traceability"`
- Line 155: `    assert len(ev['rq2']['source_files']) >= 2`

## source/tests/test_chunk06.py

SHA-256: `d775d418ab93b761e7b10dcdd1ea2674272da2c9a41cc82c726990c5bc4ea693`

219 lines. Definitions: test_claim_evidence_map_exists, test_claim_evidence_map_version, test_all_required_claims_present, test_claim_evidence_verification_passes, test_score_c_auc_claim_matches_live, test_methods_section_exists, test_experiments_section_exists, test_methods_covers_all_components, test_experiments_cites_claim_ids, test_abstract_exists_and_nonempty, test_introduction_exists, test_related_work_exists, test_conclusion_exists, test_conclusion_cites_all_three_rqs, test_conclusion_has_honest_verdicts, test_reproducibility_md_exists, test_reproducibility_has_all_steps, test_project_knowledge_updated, test_knowledge_has_chunk05_findings, test_manuscript_assembled, test_verification_report_exists, test_all_figures_referenced_exist, test_claim_evidence_all_pass

- Line 23: `def test_claim_evidence_map_exists():`
- Line 24: `    assert os.path.isfile(MAP_PATH), "claim_evidence_map.json not found"`
- Line 27: `def test_claim_evidence_map_version():`
- Line 30: `    assert m.get('map_version') in ['C06-01', 'C08-09'], f"map_version={m.get('map_version')}"`
- Line 33: `def test_all_required_claims_present():`
- Line 38: `        assert cid in m['claims'], f"Missing required claim: {cid}"`
- Line 41: `def test_claim_evidence_verification_passes():`
- Line 47: `    assert result.returncode == 0, (`
- Line 52: `def test_score_c_auc_claim_matches_live():`
- Line 64: `    assert abs(claim_val - live_val) < 0.001, (`
- Line 73: `def test_methods_section_exists():`
- Line 74: `    assert os.path.isfile(os.path.join(SECTIONS_DIR, 'methods.md'))`
- Line 77: `def test_experiments_section_exists():`
- Line 78: `    assert os.path.isfile(os.path.join(SECTIONS_DIR, 'experiments.md'))`
- Line 81: `def test_methods_covers_all_components():`
- Line 88: `        assert term.lower() in content_lower, f"Methods missing required term: '{term}'"`
- Line 91: `def test_experiments_cites_claim_ids():`
- Line 96: `    assert len(cited) >= 5, f"Experiments section only cites {len(cited)} claims (need ≥5)"`
- Line 103: `def test_abstract_exists_and_nonempty():`
- Line 105: `    assert os.path.isfile(p)`
- Line 108: `    assert len(content.strip()) >= 100, "Abstract too short (< 100 chars)"`
- Line 111: `def test_introduction_exists():`
- Line 112: `    assert os.path.isfile(os.path.join(SECTIONS_DIR, 'introduction.md'))`
- Line 115: `def test_related_work_exists():`
- Line 116: `    assert os.path.isfile(os.path.join(SECTIONS_DIR, 'related_work.md'))`
- Line 119: `def test_conclusion_exists():`
- Line 120: `    assert os.path.isfile(os.path.join(SECTIONS_DIR, 'conclusion.md'))`
- Line 123: `def test_conclusion_cites_all_three_rqs():`
- Line 126: `    assert 'RQ1' in content, "Conclusion must mention RQ1"`
- Line 127: `    assert 'RQ2' in content, "Conclusion must mention RQ2"`
- Line 128: `    assert 'RQ3' in content, "Conclusion must mention RQ3"`
- Line 131: `def test_conclusion_has_honest_verdicts():`
- Line 138: `    assert found, "Conclusion must include honest verdicts (MIXED/POSITIVE/NEGATIVE)"`
- Line 145: `def test_reproducibility_md_exists():`
- Line 146: `    assert os.path.isfile(os.path.join(repo_root, 'REPRODUCIBILITY.md'))`
- Line 149: `def test_reproducibility_has_all_steps():`
- Line 160: `        assert step in content, f"REPRODUCIBILITY.md missing step: '{step}'"`
- Line 167: `def test_project_knowledge_updated():`
- Line 169: `    assert os.path.isfile(pk_path)`
- Line 172: `def test_knowledge_has_chunk05_findings():`
- Line 175: `    assert 'InSAR' in content or 'coherence' in content, (`
- Line 178: `    assert 'CH-05' in content or 'SAR backscatter' in content.replace('SAR Backscatter', 'SAR backscatter'), (`
- Line 187: `def test_manuscript_assembled():`
- Line 188: `    assert os.path.isfile(os.path.join(MANUSCRIPT_DIR, 'sentinel_gl_manuscript.md'))`
- Line 191: `def test_verification_report_exists():`
- Line 192: `    assert os.path.isfile(os.path.join(MANUSCRIPT_DIR, 'claim_evidence_verification_report.md'))`
- Line 195: `def test_all_figures_referenced_exist():`
- Line 206: `        assert os.path.isfile(p), f"Missing figure: {fig}"`
- Line 209: `def test_claim_evidence_all_pass():`
- Line 214: `    assert 'FAIL' not in content or '0 FAIL' in content, (`
- Line 217: `    assert 'All claims verified' in content or '0 FAIL' in content, (`

## source/tests/test_chunk07.py

SHA-256: `db9c6db590bc2294ff238aea0916b24ddda4f3dbd792f1d13b2e89fe832a6c39`

335 lines. Definitions: test_inv011_no_ch06_reference, test_inv011_matches_decision_003, test_inv011_revision_note_exists, test_sentinel1_all_lakes_present, test_sentinel1_date_range, test_sentinel1_has_gaps, test_sentinel1_vv_vh_present, test_sentinel2_all_lakes_present, test_sentinel2_cloud_fraction_column, test_sentinel2_has_monsoon_gaps, test_sentinel2_lake_area_plausible, test_itslive_coverage, test_modis_lst_coverage, test_era5_coverage, test_no_ch07_coherence_files, test_feature_matrices_13_channels, test_feature_matrices_all_lakes, test_reality_gate_no_fail, test_normalization_training_only, test_training_convergence, test_no_evaluation_lake_leakage, test_embeddings_exist_all_lakes, test_embeddings_nontrivial_variance, test_checkpoint_loadable

- Line 18: `def test_inv011_no_ch06_reference():`
- Line 26: `    assert 'CH-06' not in inv011_text, "INV-011 type 3 still references excluded CH-06"`
- Line 29: `def test_inv011_matches_decision_003():`
- Line 37: `    assert 'CH-05' in inv011_text, "INV-011 type 3 must reference CH-05"`
- Line 38: `    assert '3 dB' in inv011_text or '+3 dB' in inv011_text, "INV-011 type 3 must specify +3 dB magnitude"`
- Line 41: `def test_inv011_revision_note_exists():`
- Line 46: `    assert 'C07-00' in content, "Missing C07-00 revision note in invariants.md"`
- Line 47: `    assert 'Decision 003' in content, "Revision note must reference Decision 003"`
- Line 54: `def test_sentinel1_all_lakes_present():`
- Line 61: `        assert os.path.exists(csv_path), f"Missing S1 data for {lake['id']}"`
- Line 64: `def test_sentinel1_date_range():`
- Line 71: `        assert cov >= 70.0, (`
- Line 76: `def test_sentinel1_has_gaps():`
- Line 82: `    assert any(c < 100.0 for c in coverages), (`
- Line 87: `def test_sentinel1_vv_vh_present():`
- Line 98: `    assert 'vv_lake_db' in headers, "Missing VV column"`
- Line 99: `    assert 'vh_lake_db' in headers, "Missing VH column"`
- Line 106: `def test_sentinel2_all_lakes_present():`
- Line 113: `        assert os.path.exists(csv_path), f"Missing S2 data for {lake['id']}"`
- Line 116: `def test_sentinel2_cloud_fraction_column():`
- Line 126: `        assert 'cloud_fraction' in reader.fieldnames`
- Line 129: `    assert any(cf > 0.5 for cf in cloud_fracs), (`
- Line 134: `def test_sentinel2_has_monsoon_gaps():`
- Line 144: `    assert frac >= 0.60, (`
- Line 149: `def test_sentinel2_lake_area_plausible():`
- Line 163: `    assert len(areas) > 0, "No valid lake area measurements"`
- Line 164: `    assert all(0.0001 <= a <= 50.0 for a in areas), (`
- Line 173: `def test_itslive_coverage():`
- Line 181: `        assert os.path.exists(csv_path), f"Missing ITS_LIVE for {lake['id']}"`
- Line 184: `        assert len(rows) >= 5, f"{lake['id']} has only {len(rows)} ITS_LIVE obs (need >=5)"`
- Line 187: `def test_modis_lst_coverage():`
- Line 193: `    assert avg_coverage >= 70.0, f"MODIS LST coverage {avg_coverage}% < 70%"`
- Line 196: `def test_era5_coverage():`
- Line 202: `    assert avg_coverage >= 95.0, f"ERA5 coverage {avg_coverage}% < 95%"`
- Line 205: `def test_no_ch07_coherence_files():`
- Line 209: `    assert len(coherence_files) == 0, f"Found coherence files: {coherence_files}"`
- Line 211: `    assert len(coherence_files2) == 0, f"Found CH-07 files: {coherence_files2}"`
- Line 216: `    assert 'CH-07' in manifest.get('channels_dropped', {}), (`
- Line 225: `def test_feature_matrices_13_channels():`
- Line 232: `        assert os.path.exists(npz_path), f"Missing features for {lake['id']}"`
- Line 234: `        assert data['features'].shape[1] == 13, (`
- Line 239: `def test_feature_matrices_all_lakes():`
- Line 245: `        assert os.path.exists(os.path.join(repo_root, 'data', 'features_real', lake['id'], 'feature_matrix.npz'))`
- Line 248: `def test_reality_gate_no_fail():`
- Line 254: `        assert check['verdict'] != 'FAIL', (`
- Line 259: `def test_normalization_training_only():`
- Line 269: `    assert computed_from == training_ids, (`
- Line 278: `def test_training_convergence():`
- Line 284: `    assert len(losses) >= 5, f"Only {len(losses)} epochs recorded"`
- Line 285: `    assert losses[-1] < losses[0], f"Loss did not decrease: initial={losses[0]} -> final={losses[-1]}"`
- Line 288: `def test_no_evaluation_lake_leakage():`
- Line 298: `    assert used_ids == training_ids, f"Leakage: training used {used_ids - training_ids}"`
- Line 301: `def test_embeddings_exist_all_lakes():`
- Line 308: `        assert os.path.exists(emb_path), f"Missing embeddings for {lake['id']}"`
- Line 311: `def test_embeddings_nontrivial_variance():`
- Line 320: `        assert embs.std() > 0.01, f"{lake['id']} embeddings have near-zero variance ({embs.std():.6f})"`
- Line 323: `def test_checkpoint_loadable():`
- Line 327: `    assert os.path.exists(ckpt_path), "Checkpoint not found"`
- Line 329: `    assert 'model_state_dict' in ckpt`
- Line 330: `    assert ckpt['n_channels'] == 13`

## source/tests/test_chunk08_evaluation.py

SHA-256: `1f0e4145576e9117b1c18afab5510be05e720c05baa1a92c4c62a841ca3b4138`

58 lines. Definitions: test_minmax_normalize_safeness, test_evaluation_summary_artifact_seven_methods

- Line 18: `def test_minmax_normalize_safeness():`
- Line 19: `    """Assert minmax_normalize handles zero-variance and regular arrays correctly."""`
- Line 22: `    assert np.all(normed_const == 0.0)`
- Line 26: `    assert np.allclose(normed_reg, [0.0, 0.5, 1.0])`
- Line 29: `def test_evaluation_summary_artifact_seven_methods():`
- Line 30: `    """Assert evaluation_summary_real_data.json contains valid metrics for all 7 methods."""`
- Line 32: `    assert artifact_path.exists(), "evaluation_summary_real_data.json missing"`
- Line 37: `    assert summary['n_methods'] == 7`
- Line 38: `    assert summary['scorer_non_identity_verified'] is True`
- Line 47: `        assert m in methods, f"Missing method: {m}"`
- Line 49: `        assert 'auc_roc' in metrics`
- Line 50: `        assert 'auc_pr' in metrics`
- Line 51: `        assert 'lead_time_days' in metrics`
- Line 52: `        assert 'false_positive_rate' in metrics`
- Line 53: `        assert 'synthetic_detection_rate' in metrics`
- Line 55: `    # Assert Extent Threshold metrics are NOT fabricated default placeholders`
- Line 57: `    assert isinstance(extent_m['auc_roc'], float)`
- Line 58: `    assert extent_m['auc_roc'] != 0.6500 or extent_m['auc_pr'] != 0.6100, "Extent threshold metrics must be computed dynamically, not hardcoded placeholders"`

## source/tests/test_cloud_stratified_eval.py

SHA-256: `db5092f20baaf3c0e866f3a36b6ec6f912e45e537e45b804ca15958dd5e7e379`

41 lines. Definitions: test_compute_window_cloud_fractions_nan_handling, test_cloud_stratified_eval_artifact

- Line 18: `def test_compute_window_cloud_fractions_nan_handling():`
- Line 19: `    """Assert compute_window_cloud_fractions ignores NaNs when computing window mean."""`
- Line 22: `    assert len(w_clouds) > 0`
- Line 23: `    assert not np.isnan(w_clouds).any()`
- Line 26: `def test_cloud_stratified_eval_artifact():`
- Line 27: `    """Assert run_cloud_stratified_eval generates valid JSON artifact with all 5 bins."""`
- Line 29: `    assert 'cloud_bins' in res`
- Line 30: `    assert 'cloud_robustness_metrics' in res`
- Line 35: `        assert b in bins`
- Line 36: `        assert 'window_count' in bins[b]`
- Line 37: `        assert 'score_c_auc_roc' in bins[b]`
- Line 38: `        assert 'is_thin_bin' in bins[b]`
- Line 41: `    assert artifact_path.exists()`

## source/tests/test_config.py

SHA-256: `0323d11e28ea7e9b3784a42b9924cec42beb84fa068a10b8d677e48ed936b886`

72 lines. Definitions: test_default_config_loads, test_invariant_parameters_present, test_config_loader_import, test_all_paths_defined

- Line 5: `def test_default_config_loads():`
- Line 13: `    assert isinstance(config, dict)`
- Line 14: `    assert 'project' in config`
- Line 16: `def test_invariant_parameters_present():`
- Line 26: `    assert config['temporal']['start_date'] == '2016-01-01'`
- Line 27: `    assert config['temporal']['end_date'] == '2024-10-31'`
- Line 29: `    assert config['temporal']['window_size_days'] == 180`
- Line 30: `    assert config['temporal']['stride_days'] == 30`
- Line 32: `    assert config['training']['masking_ratio'] == 0.5`
- Line 34: `    assert config['anomaly']['smoothing_span'] == 5`
- Line 36: `    assert config['evaluation']['fp_rate_target'] == 0.10`
- Line 38: `    assert config['compute']['max_training_hours'] == 72`
- Line 40: `    assert config['evaluation']['event_date'] == '2023-10-04'`
- Line 42: `    assert config['training']['seeds']['torch'] == 42`
- Line 43: `    assert config['training']['seeds']['numpy'] == 42`
- Line 44: `    assert config['training']['seeds']['synthetic_injection'] == 2023`
- Line 45: `    assert config['training']['seeds']['train_val_split'] == 7`
- Line 47: `def test_config_loader_import():`
- Line 55: `    assert config['temporal']['window_size_days'] == 180`
- Line 57: `def test_all_paths_defined():`
- Line 72: `        assert key in config['paths'], f"Missing path key: {key}"`

## source/tests/test_cusum.py

SHA-256: `289d26a570ce048735cb175176237568c2baf5f7e94baa2aa98bd635de7e4af0`

68 lines. Definitions: test_cusum_nan_skipping_recursion, test_cusum_continuous_scores, test_cusum_evaluation_artifact_and_sweep

- Line 22: `def test_cusum_nan_skipping_recursion():`
- Line 23: `    """Assert NaN entries in CH-01 series are skipped without state corruption or crash."""`
- Line 32: `    assert len(scores) == 100`
- Line 33: `    assert not np.isnan(scores).any()`
- Line 34: `    assert scores[80] > scores[10]`
- Line 37: `def test_cusum_continuous_scores():`
- Line 38: `    """Assert score output is continuous float, not binary 0/1."""`
- Line 39: `    ch01 = np.random.normal(2.0, 0.2, 200)`
- Line 44: `    assert len(unique_vals) > 10, "Score array must be continuous float"`
- Line 47: `def test_cusum_evaluation_artifact_and_sweep():`
- Line 48: `    """Assert evaluate_cusum produces valid JSON with sensitivity sweep and all INV-010 metrics."""`
- Line 50: `    assert res['is_univariate'] is True`
- Line 51: `    assert res['channel_used'] == 'CH-01_lake_area_km2'`
- Line 52: `    assert 'sensitivity_sweep_k' in res`
- Line 53: `    assert 'literature_citation' in res`
- Line 65: `        assert k in metrics, f"Missing metric key: {k}"`
- Line 68: `    assert artifact_path.exists()`

## source/tests/test_data_loader.py

SHA-256: `1cfb48f6707909dd0be7bf1223111276074ce8f31b3c593049fae7d2bb0ce9e4`

166 lines. Definitions: test_registry_role_partition, test_inv002_rejects_evaluation_lakes_in_training, test_inv002_training_loader_excludes_evaluation, test_normalization_stats_from_training_only, test_train_val_split_seed_determinism, test_inference_loader_includes_all_lakes, test_feature_tensor_shape, test_inference_loader_requires_norm_stats

- Line 23: `def test_registry_role_partition():`
- Line 28: `    assert len(by_role.get('training', [])) == 15`
- Line 31: `    assert eval_count == 5`
- Line 34: `def test_inv002_rejects_evaluation_lakes_in_training():`
- Line 53: `def test_inv002_training_loader_excludes_evaluation():`
- Line 70: `            assert lid not in eval_ids, f"Evaluation lake {lid} in training batch"`
- Line 75: `            assert lid not in eval_ids, f"Evaluation lake {lid} in validation batch"`
- Line 78: `def test_normalization_stats_from_training_only():`
- Line 94: `    assert len(leak) == 0, (`
- Line 99: `def test_train_val_split_seed_determinism():`
- Line 112: `    assert stats1['contributing_lake_ids'] == stats2['contributing_lake_ids']`
- Line 117: `def test_inference_loader_includes_all_lakes():`
- Line 136: `    assert 'SGL-001' in all_ids, "Inference loader missing South Lhonak"`
- Line 137: `    assert len(all_ids) == 20, f"Expected 20 lakes, got {len(all_ids)}"`
- Line 140: `def test_feature_tensor_shape():`
- Line 150: `    assert features.ndim == 3  # (B, T, C)`
- Line 151: `    assert features.shape[2] == 15  # 15 channels`
- Line 152: `    assert features.shape[1] > 100  # ~108 windows`
- Line 155: `def test_inference_loader_requires_norm_stats():`

## source/tests/test_data_quality.py

SHA-256: `49d6fa8497aad64652da5ed269fa46f432120f6e0d615f77a0f169f694fd7f33`

45 lines. Definitions: test_data_quality_report_exists, test_report_has_required_sections, test_data_quality_script_exists, test_south_lhonak_mentioned

- Line 10: `def test_data_quality_report_exists():`
- Line 13: `    assert os.path.isfile(report_path)`
- Line 16: `    assert len(content) > 1000, "Report seems too short"`
- Line 19: `def test_report_has_required_sections():`
- Line 29: `        assert section in content, f"Missing section: {section}"`
- Line 32: `def test_data_quality_script_exists():`
- Line 35: `    assert os.path.isfile(script_path)`
- Line 40: `def test_south_lhonak_mentioned():`
- Line 45: `    assert 'South Lhonak' in content or 'SGL-001' in content`

## source/tests/test_embedding.py

SHA-256: `779a53742ac7f434b9920705773a0ee41b9d869ac90fa714f499d62c2a369a6c`

30 lines. Definitions: test_embedding_module_exists, test_extract_function_signature, test_embedding_module_compiles

- Line 11: `def test_embedding_module_exists():`
- Line 14: `    assert callable(extract_embeddings)`
- Line 17: `def test_extract_function_signature():`
- Line 23: `    assert 'checkpoint_path' in params`
- Line 26: `def test_embedding_module_compiles():`

## source/tests/test_embedding_run.py

SHA-256: `25576ce04701c25f0bbd2ed823f9abf67c59d99b0fed621e2d55b849484e37ea`

44 lines. Definitions: test_embedding_summary_exists, test_all_lakes_have_embeddings, test_south_lhonak_embedding_shape, test_embeddings_not_collapsed

- Line 13: `def test_embedding_summary_exists():`
- Line 16: `    assert os.path.isfile(path)`
- Line 19: `def test_all_lakes_have_embeddings():`
- Line 24: `    assert len(lake_dirs) == 20, f"Only {len(lake_dirs)} lakes have embeddings"`
- Line 27: `def test_south_lhonak_embedding_shape():`
- Line 30: `    assert os.path.isfile(path)`
- Line 33: `    assert emb.shape[1] == 128, f"Expected d_model=128, got {emb.shape[1]}"`
- Line 34: `    assert emb.shape[0] > 100, f"Only {emb.shape[0]} windows"`
- Line 37: `def test_embeddings_not_collapsed():`
- Line 44: `    assert variance > 1e-6, f"Embeddings appear collapsed (var={variance})"`

## source/tests/test_encoder.py

SHA-256: `82c3e17c089a9b7eb5974cd6d3b0535312cd9122fc16f581757a7a180365ad5c`

126 lines. Definitions: test_model_instantiation, test_parameter_count, test_forward_pass_shape, test_masking_ratio, test_encode_full_sequence, test_pooled_embedding, test_loss_decreases_on_overfit, test_reconstruction_loss_masked_only

- Line 14: `def test_model_instantiation():`
- Line 17: `    assert model.n_channels == 15`
- Line 18: `    assert model.n_windows == 108`
- Line 19: `    assert model.d_model == 128`
- Line 20: `    assert model.masking_ratio == 0.5`
- Line 23: `def test_parameter_count():`
- Line 27: `    assert 500_000 < n_params < 2_500_000, (`
- Line 32: `def test_forward_pass_shape():`
- Line 42: `    assert output['reconstruction'].shape == (B, T, C)`
- Line 43: `    assert output['mask'].shape == (B, T)`
- Line 44: `    assert output['mask'].dtype == torch.bool`
- Line 45: `    assert output['loss'].ndim == 0  # scalar`
- Line 48: `def test_masking_ratio():`
- Line 60: `    assert abs(actual_ratio - 0.5) < 0.05, (`
- Line 65: `def test_encode_full_sequence():`
- Line 75: `    assert emb.shape == (B, T, 128)`
- Line 78: `def test_pooled_embedding():`
- Line 88: `    assert emb.shape == (B, 128)`
- Line 91: `def test_loss_decreases_on_overfit():`
- Line 109: `    assert losses[-1] < losses[0], (`
- Line 114: `def test_reconstruction_loss_masked_only():`
- Line 124: `    assert output['loss'].item() > 0`
- Line 125: `    assert not torch.isnan(output['loss'])`
- Line 126: `    assert not torch.isinf(output['loss'])`

## source/tests/test_environment.py

SHA-256: `f410c7b68fa0f4c42e852da0e0760cdf3d0ce897d31c570650b96808f923ae7a`

54 lines. Definitions: test_python_version, test_core_imports, test_torch_cuda_determinism_available, test_directory_structure, test_requirements_file_exists

- Line 6: `def test_python_version():`
- Line 8: `    assert sys.version_info >= (3, 10), f"Python >= 3.10 required, got {sys.version}"`
- Line 10: `def test_core_imports():`
- Line 21: `        except ImportError:`
- Line 23: `    assert not missing, f"Missing imports: {missing}"`
- Line 25: `def test_torch_cuda_determinism_available():`
- Line 29: `    assert hasattr(torch.backends, 'cudnn')`
- Line 30: `    assert hasattr(torch.backends.cudnn, 'deterministic')`
- Line 31: `    assert hasattr(torch.backends.cudnn, 'benchmark')`
- Line 33: `def test_directory_structure():`
- Line 48: `    assert not missing, f"Missing directories: {missing}"`
- Line 50: `def test_requirements_file_exists():`
- Line 54: `    assert os.path.isfile(os.path.join(repo_root, 'requirements.txt'))`

## source/tests/test_evaluation_run.py

SHA-256: `497c8752af9fbb2e34836b816d339f7b99200473c378d1e779cbfcddd0ff9d5b`

131 lines. Definitions: _load_summary, test_evaluation_summary_exists, test_all_scorers_have_results, test_scorers_have_different_auc_roc, test_lead_time_is_plausible, test_baseline_metrics_not_hardcoded, test_south_lhonak_scores_saved, test_score_a_values_are_normalized_scale, test_all_inv010_metrics_present, test_rework_version_tagged

- Line 20: `def _load_summary():`
- Line 25: `def test_evaluation_summary_exists():`
- Line 27: `    assert os.path.isfile(SUMMARY_PATH), "evaluation_summary.json missing"`
- Line 30: `def test_all_scorers_have_results():`
- Line 34: `        assert os.path.isdir(scorer_dir), f"Missing results for {scorer}"`
- Line 35: `        assert os.path.isfile(os.path.join(scorer_dir, 'e1_results.json'))`
- Line 38: `def test_scorers_have_different_auc_roc():`
- Line 50: `    assert not (auc_a == auc_b == auc_c), (`
- Line 56: `def test_lead_time_is_plausible():`
- Line 66: `            assert lt < 3000, (`
- Line 72: `def test_baseline_metrics_not_hardcoded():`
- Line 73: `    """ADVERSARIAL: Baseline metrics must not be the old hardcoded values."""`
- Line 77: `    hardcoded_fingerprint = (`
- Line 83: `    assert not hardcoded_fingerprint, (`
- Line 84: `        "Baseline metrics match the old hardcoded values — "`
- Line 89: `def test_south_lhonak_scores_saved():`
- Line 92: `    assert os.path.isdir(sgl001_dir), "No per-lake results for SGL-001"`
- Line 94: `    assert os.path.isfile(csv_path)`
- Line 97: `def test_score_a_values_are_normalized_scale():`
- Line 108: `    assert max_score_a < 100, (`
- Line 114: `def test_all_inv010_metrics_present():`
- Line 121: `            assert metric in summary['scorer_comparison'][scorer], (`
- Line 126: `def test_rework_version_tagged():`
- Line 129: `    assert summary.get('rework_version') == 'C04-R1', (`

## source/tests/test_feature_schema_v2.py

SHA-256: `e409797c1e1a16bfa92e49f8b7a0021052e857914c3e11b3adf53ba81ff084fa`

137 lines. Definitions: valid_matrix, test_default_schema_construction, test_schema_hash_deterministic, test_matrix_validation_valid, test_matrix_validation_out_of_range, test_feature_channel_rejects_invalid_physical_metadata, test_matrix_validation_enforces_shapes_and_known_keys, test_matrix_validation_preserves_missingness_observability, test_matrix_validation_returns_errors_for_nonnumeric_values

- Line 28: `def valid_matrix(schema):`
- Line 52: `def test_default_schema_construction():`
- Line 54: `    assert {channel.channel_id for channel in schema.channels} == EXPECTED_CHANNEL_IDS`
- Line 55: `    assert schema.channel_count == len(schema.channels)`
- Line 56: `    assert all(channel.temporal_cadence != "static" for channel in schema.temporal_channels)`
- Line 57: `    assert all(channel.temporal_cadence == "static" for channel in schema.static_channels)`
- Line 58: `    assert all(`
- Line 65: `def test_schema_hash_deterministic():`
- Line 70: `    assert h1 == h2`
- Line 71: `    assert len(h1) == 64`
- Line 73: `    assert schema2.compute_schema_hash() != h1`
- Line 76: `def test_matrix_validation_valid():`
- Line 79: `    assert len(errors) == 0`
- Line 82: `def test_matrix_validation_out_of_range():`
- Line 89: `    assert any("out of physical range" in e for e in errors)`
- Line 90: `    assert any("Missing required channel" in e for e in errors)`
- Line 93: `def test_feature_channel_rejects_invalid_physical_metadata():`
- Line 103: `def test_matrix_validation_enforces_shapes_and_known_keys():`
- Line 110: `    assert any("does not match" in error for error in errors)`
- Line 111: `    assert any("Unexpected matrix keys" in error for error in errors)`
- Line 114: `def test_matrix_validation_preserves_missingness_observability():`
- Line 119: `    assert schema.validate_matrix(matrix) == []`
- Line 123: `    assert any("marks non-finite values as valid" in error for error in errors)`
- Line 128: `    assert any("must encode invalid entries as NaN" in error for error in errors)`
- Line 131: `def test_matrix_validation_returns_errors_for_nonnumeric_values():`
- Line 137: `    assert any("must contain numeric values" in error for error in errors)`

## source/tests/test_figures.py

SHA-256: `e3ddc55a3097a27ea4809e5b4d52e336cd298d66acc1f1402450a0ccd8136e32`

23 lines. Definitions: test_south_lhonak_timeline_exists, test_scorer_comparison_exists, test_at_least_4_figures

- Line 10: `def test_south_lhonak_timeline_exists():`
- Line 12: `    assert os.path.isfile(os.path.join(FIG_DIR, 'south_lhonak_anomaly_timeline.png'))`
- Line 15: `def test_scorer_comparison_exists():`
- Line 17: `    assert os.path.isfile(os.path.join(FIG_DIR, 'scorer_comparison_table.png'))`
- Line 20: `def test_at_least_4_figures():`
- Line 23: `    assert len(pngs) >= 4, f"Only {len(pngs)} figures generated"`

## source/tests/test_insar.py

SHA-256: `6ca0bea719ad046fae1739000fdb7c9af7c7ac3fc4ebfc76e4b858855eba29d7`

52 lines. Definitions: test_insar_module_exists, test_feasibility_report_exists, test_south_lhonak_assessed, test_feasibility_has_evidence

- Line 13: `def test_insar_module_exists():`
- Line 15: `    assert os.path.isfile(os.path.join(INSAR_DIR, 'insar_feasibility.py'))`
- Line 18: `def test_feasibility_report_exists():`
- Line 21: `    assert os.path.isfile(report_path), "feasibility_report.json not found"`
- Line 24: `    assert 'overall_verdict' in report, "Missing overall_verdict"`
- Line 25: `    assert report['overall_verdict'] in ['FEASIBLE', 'MARGINAL', 'INFEASIBLE'], (`
- Line 30: `def test_south_lhonak_assessed():`
- Line 41: `    assert sgl_001 is not None, "SGL-001 not assessed"`
- Line 44: `def test_feasibility_has_evidence():`
- Line 49: `    assert ('evidence' in report or 'methodology' in report or `

## source/tests/test_isolation_forest.py

SHA-256: `2807ec615c83b47ec0af8f5177ebf8205b3ec5da555c3ecaba66d5c42827d593`

58 lines. Definitions: test_isolation_forest_train_leakage_boundary, test_ema_smooth_span5, test_isolation_forest_evaluation_artifact

- Line 22: `def test_isolation_forest_train_leakage_boundary():`
- Line 23: `    """Assert IsolationForest fits strictly on training-role lakes (INV-002)."""`
- Line 25: `    assert len(training_lakes) == 15`
- Line 26: `    assert 'SGL-001' not in training_lakes`
- Line 27: `    assert 'SGL-002' not in training_lakes`
- Line 28: `    assert clf.n_features_in_ == 4680  # 180 windows * 26 columns`
- Line 31: `def test_ema_smooth_span5():`
- Line 32: `    """Assert EMA smoothing function behaves deterministically with span=5."""`
- Line 35: `    assert len(smoothed) == 5`
- Line 36: `    assert smoothed[0] == 1.0`
- Line 37: `    assert smoothed[1] > 1.0 and smoothed[1] < 2.0`
- Line 40: `def test_isolation_forest_evaluation_artifact():`
- Line 41: `    """Assert evaluate_isolation_forest generates valid JSON artifact with all INV-010 metrics."""`
- Line 43: `    assert 'metrics' in res`
- Line 55: `        assert k in metrics, f"Missing metric key: {k}"`
- Line 58: `    assert artifact_path.exists()`

## source/tests/test_missing_data_policy.py

SHA-256: `c433ac613b3db32db8b73233d416c4746af7cee7088bc86d54dc4371e3bfb3cc`

81 lines. Definitions: test_medians_computed_from_training_lakes_only, test_transform_window_shape_and_columns, test_exclusion_threshold_50_percent, test_run_missing_data_policy_artifact

- Line 19: `def test_medians_computed_from_training_lakes_only():`
- Line 20: `    """Assert imputation medians use strictly the 15 training-lake IDs from normalization_stats.json."""`
- Line 22: `    assert norm_path.exists(), "normalization_stats.json missing"`
- Line 28: `    assert len(training_lakes) == 15`
- Line 29: `    assert 'SGL-001' not in training_lakes`
- Line 30: `    assert 'SGL-002' not in training_lakes`
- Line 31: `    assert 'SGL-003' not in training_lakes`
- Line 32: `    assert 'SGL-004' not in training_lakes`
- Line 33: `    assert 'SGL-005' not in training_lakes`
- Line 36: `    assert len(medians) == 13`
- Line 37: `    assert not np.isnan(medians).any()`
- Line 40: `def test_transform_window_shape_and_columns():`
- Line 41: `    """Assert transform_window converts (180, 13) to (180, 26) with binary indicators."""`
- Line 48: `    assert transformed.shape == (180, 26)`
- Line 52: `    assert np.all(transformed[:10, 13] == 1.0)  # Imputed`
- Line 53: `    assert np.all(transformed[10:, 13] == 0.0)  # Observed`
- Line 54: `    assert np.all(transformed[:10, 0] == 0.0)   # Imputed value from medians`
- Line 57: `def test_exclusion_threshold_50_percent():`
- Line 58: `    """Assert window with >50% NaN is excluded, and <=50% NaN is retained."""`
- Line 68: `    assert stats['excluded_windows'] >= 1`
- Line 69: `    assert windows_26d.shape[2] == 26`
- Line 72: `def test_run_missing_data_policy_artifact():`
- Line 73: `    """Assert run_missing_data_policy creates baseline_imputation_stats.json."""`
- Line 75: `    assert 'imputed_fraction' in summary`
- Line 76: `    assert 'imputed_percentage' in summary`
- Line 77: `    assert summary['n_training_lakes'] == 15`
- Line 78: `    assert summary['imputed_fraction'] >= 0.0 and summary['imputed_fraction'] <= 1.0`
- Line 81: `    assert artifact_path.exists()`

## source/tests/test_ocsvm.py

SHA-256: `5aca2e6ca40fc975b8442d23263832028e0495ca568caf2f32dcc3b69b7a68d0`

51 lines. Definitions: test_ocsvm_train_leakage_boundary, test_ocsvm_evaluation_artifact_and_parameters

- Line 2: `Unit test suite for One-Class SVM Baseline (Contract C08-03).`
- Line 22: `def test_ocsvm_train_leakage_boundary():`
- Line 23: `    """Assert OneClassSVM fits strictly on training-role lakes (INV-002)."""`
- Line 25: `    assert len(training_lakes) == 15`
- Line 26: `    assert 'SGL-001' not in training_lakes`
- Line 27: `    assert 'SGL-002' not in training_lakes`
- Line 30: `def test_ocsvm_evaluation_artifact_and_parameters():`
- Line 31: `    """Assert evaluate_ocsvm generates valid JSON artifact with all INV-010 metrics and parameter documentation."""`
- Line 33: `    assert 'metrics' in res`
- Line 34: `    assert 'parameters' in res`
- Line 35: `    assert res['parameters']['nu'] == 0.1`
- Line 36: `    assert 'nu_justification' in res['parameters']`
- Line 48: `        assert k in metrics, f"Missing metric key: {k}"`
- Line 51: `    assert artifact_path.exists()`

## source/tests/test_prediction_ledger.py

SHA-256: `451e237ac49612de3009611a5fe1d4fe11d62aa70c0822a3ea2b99979e283722`

127 lines. Definitions: create_sample_row, test_ledger_creation_and_hashing, test_validator_detects_duplicates, test_validator_rejects_unanchored_statistical_inputs, test_prediction_row_rejects_invalid_lineage_and_decision_fields, test_statistical_inputs_are_extracted_from_matching_persisted_rows, test_statistical_input_hash_and_estimability_are_fail_closed

- Line 16: `def create_sample_row(instance_id="inst_001", method="TS-MAE", score=0.75, label=1):`
- Line 41: `def test_ledger_creation_and_hashing():`
- Line 49: `    assert len(h1) == 64`
- Line 50: `    assert len(ledger.get_unique_lakes()) == 1`
- Line 51: `    assert ledger.finalize() == h1`
- Line 57: `    assert restored.to_records() == ledger.to_records()`
- Line 60: `def test_validator_detects_duplicates():`
- Line 66: `    assert any("Duplicate prediction entry" in e for e in errors)`
- Line 69: `def test_validator_rejects_unanchored_statistical_inputs():`
- Line 73: `    assert "Manufacturing pseudo-observation arrays is strictly prohibited" in str(excinfo.value)`
- Line 80: `def test_prediction_row_rejects_invalid_lineage_and_decision_fields():`
- Line 87: `def test_statistical_inputs_are_extracted_from_matching_persisted_rows():`
- Line 101: `    assert scores.tolist() == [0.2, 0.8]`
- Line 102: `    assert labels.tolist() == [0, 1]`
- Line 103: `    assert lakes == ("LK_CONTROL_01", "LK_CONTROL_02")`
- Line 104: `    assert set(StatisticalInputValidator.clustered_units(ledger)) == {`
- Line 109: `def test_statistical_input_hash_and_estimability_are_fail_closed():`

## source/tests/test_preprocessing.py

SHA-256: `c55625041a36244684710add462dc638ee75c823747381d3a740130af6ab63f8`

78 lines. Definitions: test_all_preprocessing_modules_exist, test_preprocess_interface, test_common_utilities_exist, test_time_windows_respect_invariants, test_no_cross_lake_operations

- Line 13: `def test_all_preprocessing_modules_exist():`
- Line 23: `        assert os.path.isfile(path), f"Missing: {mod_file}"`
- Line 26: `def test_preprocess_interface():`
- Line 38: `            assert hasattr(mod, 'preprocess'), f"{mod_name} missing preprocess()"`
- Line 41: `            assert 'lake_id' in params`
- Line 42: `            assert 'raw_dir' in params`
- Line 43: `            assert 'output_dir' in params`
- Line 44: `        except ImportError:`
- Line 51: `def test_common_utilities_exist():`
- Line 57: `    assert len(windows) > 0, "build_time_windows returned empty list"`
- Line 58: `    assert len(windows[0]) == 2, "Each window should be (start, end) tuple"`
- Line 61: `def test_time_windows_respect_invariants():`
- Line 65: `    assert len(windows) >= 5, f"Expected ≥5 windows, got {len(windows)}"`
- Line 66: `    assert len(windows) <= 15, f"Too many windows: {len(windows)}"`
- Line 69: `def test_no_cross_lake_operations():`
- Line 76: `            assert 'for lake_id in' not in content or 'for lake_id in [lake_id]' in content, (`

## source/tests/test_preprocessing_run.py

SHA-256: `bf2b49d9b52136de5ffba315b710dd891f33c6cfa47501c0899da5cab6e300fd`

57 lines. Definitions: test_preprocessing_summary_exists, test_south_lhonak_preprocessed, test_preprocessing_script_exists, test_at_least_80_percent_lakes

- Line 11: `def test_preprocessing_summary_exists():`
- Line 14: `    assert os.path.isfile(summary_path)`
- Line 17: `    assert 'overall' in summary or 'per_source' in summary`
- Line 20: `def test_south_lhonak_preprocessed():`
- Line 32: `    assert has_data, "No preprocessed data for South Lhonak"`
- Line 35: `def test_preprocessing_script_exists():`
- Line 38: `    assert os.path.isfile(script_path)`
- Line 43: `def test_at_least_80_percent_lakes():`
- Line 55: `    assert len(lake_dirs_found) >= 16, (`

## source/tests/test_protocol_e1.py

SHA-256: `7ec79cda4d98b335bd7777cb03783b410f7aedad35fcfbbdc794e89bfd740d77`

32 lines. Definitions: test_protocol_e1_f3_resolution

- Line 18: `def test_protocol_e1_f3_resolution():`
- Line 19: `    """Assert resolve_protocol_e1_f3 generates protocol_e1_real_data.json and appends Decision 006."""`
- Line 21: `    assert res['event_lake_id'] == 'SGL-001'`
- Line 22: `    assert 'f3_falsification_verdict' in res`
- Line 23: `    assert res['f3_falsification_verdict'] in ['SUCCESS', 'FAILURE', 'AMBIGUOUS_FAILURE']`
- Line 24: `    assert 'pre_event_flagged_percentage' in res`
- Line 27: `    assert artifact_path.exists()`
- Line 30: `    assert decision_log_path.exists()`
- Line 32: `    assert 'Decision 006 — Protocol E1 Falsification Resolution' in content`

## source/tests/test_protocols.py

SHA-256: `12ac162176c54ffbc99b0c8a119eb0ecf627324a65895c7e5271fef3eb58ebb9`

135 lines. Definitions: test_metrics_module_exists, test_lead_time_computation, test_lead_time_returns_none_when_no_detection, test_fp_rate_computation, test_event_date_is_inv009, test_ema_span_is_inv006, test_e1_module_exists, test_e2_module_exists, test_e3_module_exists, test_e4_module_exists, test_runner_exists, test_auc_computation, test_auc_handles_single_class, test_synthetic_detection_rate, test_peak_magnitude_computation

- Line 11: `def test_metrics_module_exists():`
- Line 14: `    assert callable(compute_full_metrics)`
- Line 17: `def test_lead_time_computation():`
- Line 26: `    assert lead_time is not None`
- Line 27: `    assert lead_time > 0`
- Line 30: `def test_lead_time_returns_none_when_no_detection():`
- Line 36: `    assert lead_time is None`
- Line 39: `def test_fp_rate_computation():`
- Line 48: `    assert abs(fp_rate - 0.10) < 0.01`
- Line 51: `def test_event_date_is_inv009():`
- Line 55: `    assert EVENT_DATE == datetime(2023, 10, 4)`
- Line 58: `def test_ema_span_is_inv006():`
- Line 61: `    assert EMA_SPAN == 5`
- Line 64: `def test_e1_module_exists():`
- Line 67: `    assert callable(run_e1_retrospective)`
- Line 70: `def test_e2_module_exists():`
- Line 73: `    assert callable(run_e2_negative_controls)`
- Line 76: `def test_e3_module_exists():`
- Line 79: `    assert callable(run_e3_synthetic)`
- Line 82: `def test_e4_module_exists():`
- Line 85: `    assert callable(run_e4_baseline)`
- Line 88: `def test_runner_exists():`
- Line 91: `    assert callable(run_full_evaluation)`
- Line 94: `def test_auc_computation():`
- Line 102: `    assert 0 <= auc['auc_roc'] <= 1`
- Line 103: `    assert 0 <= auc['auc_pr'] <= 1`
- Line 106: `def test_auc_handles_single_class():`
- Line 111: `    scores = np.random.rand(10)`
- Line 114: `    assert auc['auc_roc'] == 0.5  # Random baseline`
- Line 117: `def test_synthetic_detection_rate():`
- Line 123: `    assert abs(rate - 0.6) < 0.01`
- Line 126: `def test_peak_magnitude_computation():`
- Line 135: `    assert peak == 5.0`

## source/tests/test_provenance_manifest_schemas.py

SHA-256: `8e44628eee4df2342a339067eee9afd0782fd2f970b3c717b7b924981c942042`

212 lines. Definitions: digest, valid_provenance_dict, valid_manifest_dict, test_complete_provenance_record_validates, test_provenance_round_trip_is_lossless, test_zero_observation_blocked_query_is_legitimate_and_explicit, test_missing_required_nested_checksum_is_rejected, test_invalid_sha256_is_rejected, test_non_utc_timestamp_and_invalid_http_code_are_rejected, test_query_parameters_require_spatial_temporal_and_sensor_fields, test_scene_count_dates_and_api_calls_are_consistent, test_downloaded_filenames_must_be_unique, test_complete_dataset_manifest_validates_and_round_trips, test_manifest_rejects_unknown_evidence_class, test_manifest_requires_one_checksum_per_file, test_manifest_rejects_malformed_missingness_and_gap_statistics, test_validation_detects_post_construction_mutation

- Line 16: `def digest(payload: bytes = b"") -> str:`
- Line 20: `def valid_provenance_dict():`
- Line 52: `def valid_manifest_dict():`
- Line 86: `def test_complete_provenance_record_validates():`
- Line 88: `    assert record.validate_record() is record`
- Line 89: `    assert record.downloaded_file_manifest[0].size_bytes == 128`
- Line 92: `def test_provenance_round_trip_is_lossless():`
- Line 95: `    assert restored.to_dict() == record.to_dict()`
- Line 98: `def test_zero_observation_blocked_query_is_legitimate_and_explicit():`
- Line 112: `    assert record.total_scenes_returned == 0`
- Line 113: `    assert record.http_status_codes == [401]`
- Line 116: `def test_missing_required_nested_checksum_is_rejected():`
- Line 127: `def test_invalid_sha256_is_rejected(bad_hash):`
- Line 134: `def test_non_utc_timestamp_and_invalid_http_code_are_rejected():`
- Line 146: `def test_query_parameters_require_spatial_temporal_and_sensor_fields():`
- Line 153: `def test_scene_count_dates_and_api_calls_are_consistent():`
- Line 165: `def test_downloaded_filenames_must_be_unique():`
- Line 174: `def test_complete_dataset_manifest_validates_and_round_trips():`
- Line 176: `    assert manifest.validate_record() is manifest`
- Line 177: `    assert manifest.evidence_class is EvidenceClass.AUTHENTICATED`
- Line 178: `    assert DatasetManifest.from_json(manifest.to_json()).to_dict() == manifest.to_dict()`
- Line 181: `def test_manifest_rejects_unknown_evidence_class():`
- Line 188: `def test_manifest_requires_one_checksum_per_file():`
- Line 195: `def test_manifest_rejects_malformed_missingness_and_gap_statistics():`
- Line 207: `def test_validation_detects_post_construction_mutation():`

## source/tests/test_quarantine_enforcement.py

SHA-256: `b59814667c14387dc71368fea1e36e6759edef758222acce832e4153b0e766f6`

109 lines. Definitions: test_quarantine_registry_loads, test_quarantine_detects_tainted_paths, test_quarantine_passes_clean_paths, test_path_matching_is_boundary_aware_and_case_insensitive, test_known_hashes_are_checked_by_convenience_helper, test_registry_rejects_duplicate_ids_and_malformed_hashes

- Line 16: `def test_quarantine_registry_loads():`
- Line 18: `    assert len(reg.entries) >= 10`
- Line 19: `    assert reg.version == "2.0"`
- Line 22: `def test_quarantine_detects_tainted_paths():`
- Line 34: `        assert is_q is True, f"Failed to detect tainted path: {path}"`
- Line 39: `def test_quarantine_passes_clean_paths():`
- Line 48: `        assert is_q is False`
- Line 52: `def test_path_matching_is_boundary_aware_and_case_insensitive():`
- Line 54: `    assert reg.is_quarantined("DATA\\RAW\\SENTINEL1_GRD\\lake.csv")[0] is True`
- Line 55: `    assert reg.is_quarantined("./data/raw/sentinel1_grd/lake.csv")[0] is True`
- Line 56: `    assert reg.is_quarantined("data/raw/sentinel1_grd_backup/lake.csv")[0] is False`
- Line 59: `def test_known_hashes_are_checked_by_convenience_helper(tmp_path):`
- Line 85: `def test_registry_rejects_duplicate_ids_and_malformed_hashes(tmp_path):`

## source/tests/test_registry.py

SHA-256: `6a1468ba32a1fab65372de4cafb34d73cc3612b254c7f876bbcb32eee2f61e0d`

101 lines. Definitions: test_schema_is_valid_json, test_validate_correct_registry, test_validate_catches_duplicate_ids, test_validate_catches_invalid_bbox, test_validate_requires_evaluation_event

- Line 13: `def test_schema_is_valid_json():`
- Line 18: `    assert '$schema' in schema`
- Line 19: `    assert 'lakes' in schema['properties']`
- Line 22: `def test_validate_correct_registry():`
- Line 28: `    assert valid, f"Validation failed: {errors}"`
- Line 31: `def test_validate_catches_duplicate_ids():`
- Line 49: `        assert not valid, "Should have caught duplicate IDs"`
- Line 50: `        assert any('duplicate' in e.lower() for e in errors)`
- Line 55: `def test_validate_catches_invalid_bbox():`
- Line 75: `        assert not valid, "Should have caught invalid bounding box"`
- Line 80: `def test_validate_requires_evaluation_event():`
- Line 99: `        assert not valid, "Should require at least one evaluation_event lake"`

## source/tests/test_registry_population.py

SHA-256: `75f6c7b06e7a3a39d085c64157412e99a97c8689a462cbc480c7d332dc7f078d`

90 lines. Definitions: _load_registry, test_registry_validates_against_schema, test_minimum_lake_count, test_south_lhonak_is_evaluation_event, test_role_distribution, test_unique_ids, test_no_evaluation_lake_in_training_role, test_coordinates_in_hkh_range, test_bounding_boxes_consistent

- Line 14: `def _load_registry():`
- Line 19: `def test_registry_validates_against_schema():`
- Line 24: `    assert valid, f"Validation errors: {errors}"`
- Line 27: `def test_minimum_lake_count():`
- Line 30: `    assert len(reg['lakes']) >= 15, f"Only {len(reg['lakes'])} lakes, need >= 15"`
- Line 33: `def test_south_lhonak_is_evaluation_event():`
- Line 38: `    assert len(south_lhonak) >= 1, "South Lhonak Lake not found"`
- Line 39: `    assert south_lhonak[0]['role'] == 'evaluation_event'`
- Line 42: `def test_role_distribution():`
- Line 49: `    assert roles.get('evaluation_event', 0) >= 1, "Need >= 1 evaluation_event"`
- Line 50: `    assert roles.get('evaluation_control', 0) >= 3, "Need >= 3 evaluation_control"`
- Line 51: `    assert roles.get('training', 0) >= 10, "Need >= 10 training"`
- Line 54: `def test_unique_ids():`
- Line 58: `    assert len(ids) == len(set(ids)), f"Duplicate IDs: {set(x for x in ids if ids.count(x) > 1)}"`
- Line 61: `def test_no_evaluation_lake_in_training_role():`
- Line 66: `            assert lake['role'] != 'training', (`
- Line 72: `def test_coordinates_in_hkh_range():`
- Line 79: `        assert 25 <= lat <= 40, f"{lake['id']}: lat {lat} outside HKH range"`
- Line 80: `        assert 70 <= lon <= 100, f"{lake['id']}: lon {lon} outside HKH range"`
- Line 81: `        assert 3000 <= elev <= 7000, f"{lake['id']}: elevation {elev}m unusual for glacial lake"`
- Line 84: `def test_bounding_boxes_consistent():`
- Line 89: `        assert bb['north'] > bb['south'], f"{lake['id']}: north <= south"`
- Line 90: `        assert bb['east'] > bb['west'], f"{lake['id']}: east <= west"`

## source/tests/test_rq_answers.py

SHA-256: `6b2665628b146367f172b3da5608386157476dee441a3058e052317ced90d59c`

57 lines. Definitions: test_rq1_answer_exists, test_rq3_answer_exists, test_evidence_summary_matches_evaluation_summary, test_evidence_has_source_file, test_rq1_has_verdict

- Line 12: `def test_rq1_answer_exists():`
- Line 14: `    assert os.path.isfile(os.path.join(RQ_DIR, 'rq1_answer.md'))`
- Line 17: `def test_rq3_answer_exists():`
- Line 19: `    assert os.path.isfile(os.path.join(RQ_DIR, 'rq3_answer.md'))`
- Line 22: `def test_evidence_summary_matches_evaluation_summary():`
- Line 32: `        assert evidence_lt == actual_lt, (`
- Line 33: `            f"FABRICATION: evidence says {scorer} lead_time={evidence_lt}, "`
- Line 39: `        assert abs(evidence_auc - actual_auc) < 0.001, (`
- Line 40: `            f"FABRICATION: evidence says {scorer} auc_roc={evidence_auc}, "`
- Line 45: `def test_evidence_has_source_file():`
- Line 49: `    assert 'source_file' in evidence['rq1'], "Missing source_file traceability"`
- Line 52: `def test_rq1_has_verdict():`
- Line 56: `    assert 'Verdict' in content or 'verdict' in content`
- Line 57: `    assert len(content) > 500, "RQ1 answer too short"`

## source/tests/test_sanity_check.py

SHA-256: `3673dff770559d439f752eeec2e82ad49b6cf282c6230c67132a1ca4b607c288`

38 lines. Definitions: test_sanity_check_module_exists, test_sanity_check_report_exists, test_sanity_check_has_verdict, test_sanity_check_plots_exist

- Line 11: `def test_sanity_check_module_exists():`
- Line 14: `    assert os.path.isfile(path)`
- Line 17: `def test_sanity_check_report_exists():`
- Line 20: `    assert os.path.isfile(report_path)`
- Line 23: `    assert len(content) > 500, "Report too short"`
- Line 26: `def test_sanity_check_has_verdict():`
- Line 31: `    assert 'PASS' in content or 'FAIL' in content, "No verdict found"`
- Line 34: `def test_sanity_check_plots_exist():`
- Line 38: `    assert len(plot_files) >= 1, "No visualization plots found"`

## source/tests/test_synthetic.py

SHA-256: `9dcc4c82a2ecc27dcb085f07fae77d0bf4f46a70628a50c4a7444ebb3182bc35`

49 lines. Definitions: test_injector_module_exists, test_injection_preserves_shape, test_injection_actually_modifies, test_injection_seed_determinism, test_anomaly_config_exists

- Line 11: `def test_injector_module_exists():`
- Line 14: `    assert SyntheticInjector is not None`
- Line 17: `def test_injection_preserves_shape():`
- Line 21: `    features = np.random.rand(108, 15).astype(np.float32)`
- Line 23: `    assert modified.shape == features.shape`
- Line 26: `def test_injection_actually_modifies():`
- Line 32: `    assert not np.array_equal(modified, features)`
- Line 35: `def test_injection_seed_determinism():`
- Line 38: `    features = np.random.rand(108, 15).astype(np.float32)`
- Line 46: `def test_anomaly_config_exists():`
- Line 49: `    assert os.path.isfile(config_path)`

## source/tests/test_task_protocols_spec.py

SHA-256: `b95982a4c5b95d30780a86b3c31b5d40193138a782f449833bf2eb87111f641f`

178 lines. Definitions: test_task_b_date_chronology, test_calibration_policy_enforces_fpr_constraint, test_calibration_policy_no_feasible_threshold, test_score_c_normalization_isolated, instance, budget, test_common_instances_and_information_budget_are_enforced, test_task_specific_instance_semantics_are_enforced, test_score_c_isolation_and_boundaries_are_enforced, test_threshold_calibration_rejects_instance_overlap

- Line 22: `def test_task_b_date_chronology():`
- Line 25: `    assert res_pre["is_pre_event"] is True`
- Line 26: `    assert res_pre["is_post_event"] is False`
- Line 27: `    assert res_pre["lead_time_days"] == 19`
- Line 31: `    assert res_post["is_pre_event"] is False`
- Line 32: `    assert res_post["is_post_event"] is True`
- Line 33: `    assert res_post["lead_time_days"] < 0`
- Line 36: `    assert res_event["classification"] == "event_overlap"`
- Line 37: `    assert res_event["is_pre_event"] is False`
- Line 38: `    assert res_event["is_post_event"] is False`
- Line 39: `    assert res_event["lead_time_days"] == 0`
- Line 42: `    assert res_overlap["classification"] == "event_overlap"`
- Line 43: `    assert res_overlap["is_event_overlap"] is True`
- Line 46: `def test_calibration_policy_enforces_fpr_constraint():`
- Line 48: `    np.random.seed(42)`
- Line 49: `    clean_scores = np.random.normal(loc=0.2, scale=0.05, size=100)`
- Line 50: `    anomaly_scores = np.random.normal(loc=0.8, scale=0.1, size=20)`
- Line 61: `    assert thresh is not None`
- Line 62: `    assert meta["status"] == "FEASIBLE_THRESHOLD_FOUND"`
- Line 63: `    assert meta["observed_fpr"] <= 0.10`
- Line 64: `    assert meta["inv_007_compliant"] is True`
- Line 67: `def test_calibration_policy_no_feasible_threshold():`
- Line 79: `    assert thresh is None`
- Line 80: `    assert meta["status"] == "NO_FEASIBLE_THRESHOLD"`
- Line 81: `    assert meta["inv_007_compliant"] is False`
- Line 84: `def test_score_c_normalization_isolated():`
- Line 100: `    assert np.isclose(score_c[0], 0.5)`
- Line 102: `    assert np.isclose(score_c[1], 1.0)`
- Line 104: `    assert np.isclose(score_c[2], 0.0)`
- Line 107: `def instance(instance_id="i1", label=0):`
- Line 124: `def budget(feature_hash="sha256:feature-v2"):`
- Line 135: `def test_common_instances_and_information_budget_are_enforced():`
- Line 146: `def test_task_specific_instance_semantics_are_enforced():`
- Line 162: `def test_score_c_isolation_and_boundaries_are_enforced():`
- Line 170: `def test_threshold_calibration_rejects_instance_overlap():`

## source/tests/test_training.py

SHA-256: `96bee7369ac46f32be2ea119ea63fe52229290eb11bcb461cb32950731a8e5dc`

52 lines. Definitions: test_trainer_module_exists, test_trainer_has_required_methods, test_device_selection, test_checkpoint_format

- Line 11: `def test_trainer_module_exists():`
- Line 14: `    assert Trainer is not None`
- Line 17: `def test_trainer_has_required_methods():`
- Line 20: `    assert hasattr(Trainer, 'train_epoch')`
- Line 21: `    assert hasattr(Trainer, 'validate')`
- Line 22: `    assert hasattr(Trainer, 'fit')`
- Line 23: `    assert hasattr(Trainer, 'save_checkpoint')`
- Line 24: `    assert hasattr(Trainer, 'load_checkpoint')`
- Line 27: `def test_device_selection():`
- Line 31: `    assert isinstance(device, torch.device)`
- Line 34: `def test_checkpoint_format():`
- Line 51: `        assert loaded['epoch'] == 0`

## source/tests/test_training_run.py

SHA-256: `4233d92a0faacf4adaee931b155e0a95d3738df703bf0c0cbce45a06f9da8727`

67 lines. Definitions: test_best_checkpoint_exists, test_training_log_exists, test_loss_decreased, test_no_nan_in_training, test_training_summary_exists

- Line 14: `def test_best_checkpoint_exists():`
- Line 17: `    assert os.path.isfile(path), "Best checkpoint not found"`
- Line 19: `    assert 'model_state_dict' in checkpoint`
- Line 20: `    assert 'epoch' in checkpoint`
- Line 23: `def test_training_log_exists():`
- Line 26: `    assert os.path.isfile(log_path)`
- Line 29: `    assert len(lines) >= 10, f"Only {len(lines)} log entries"`
- Line 30: `    assert 'train_loss' in lines[0]`
- Line 33: `def test_loss_decreased():`
- Line 42: `    assert final_loss < initial_loss, (`
- Line 47: `    assert reduction > 0.3, f"Only {reduction*100:.1f}% loss reduction"`
- Line 50: `def test_no_nan_in_training():`
- Line 57: `        assert entry['train_loss'] == entry['train_loss'], f"NaN at epoch {i}"`
- Line 60: `def test_training_summary_exists():`
- Line 63: `    assert os.path.isfile(summary_path)`
- Line 66: `    assert 'convergence' in summary`
- Line 67: `    assert 'wall_time_seconds' in summary`

## source/tests/test_utils.py

SHA-256: `5412e3627e9cb0bab4cae3c4e0a727660a1bea5afecb8deb0d6351faf63fb78f`

114 lines. Definitions: test_set_seed_determinism, test_hash_file, test_hash_directory, test_verify_hash, test_setup_logger, test_log_to_jsonl

- Line 14: `def test_set_seed_determinism():`
- Line 15: `    """set_seed produces identical random sequences across calls."""`
- Line 21: `    a_np = np.random.rand(10)`
- Line 25: `    b_np = np.random.rand(10)`
- Line 28: `    assert (a_np == b_np).all(), "NumPy not deterministic after set_seed"`
- Line 29: `    assert torch.equal(a_torch, b_torch), "PyTorch not deterministic after set_seed"`
- Line 32: `def test_hash_file():`
- Line 43: `        assert result == expected, f"Hash mismatch: {result} != {expected}"`
- Line 48: `def test_hash_directory():`
- Line 59: `        assert list(result.keys()) == ['a.txt', 'b.txt'], "Not sorted"`
- Line 60: `        assert result['a.txt'] == hashlib.sha256(b'aaa').hexdigest()`
- Line 63: `def test_verify_hash():`
- Line 73: `        assert verify_hash(tmp_path, correct_hash) is True`
- Line 74: `        assert verify_hash(tmp_path, "wrong_hash") is False`
- Line 79: `def test_setup_logger():`
- Line 88: `        assert os.path.exists(log_file), "Log file not created"`
- Line 91: `        assert "Test message" in content`
- Line 94: `def test_log_to_jsonl():`
- Line 108: `        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"`
- Line 110: `        assert entry1['event'] == 'test'`
- Line 111: `        assert entry1['value'] == 42`
- Line 112: `        assert 'timestamp' in entry1, "Missing automatic timestamp"`

## source/utils/__init__.py

SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

0 lines. Definitions: 



## source/utils/config_loader.py

SHA-256: `4f6b3eea6ea6ea4fe6ba2ef90a22ca4d6408aa0775bc1f0297ecd1f49055ad8d`

90 lines. Definitions: get_default_config_path, _deep_merge, validate_config, load_config

- Line 10: `def get_default_config_path() -> str:`
- Line 16: `def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:`
- Line 27: `def validate_config(config: Dict[str, Any]) -> bool:`
- Line 53: `def load_config(experiment_name: Optional[str] = None, config_path: Optional[str] = None) -> Dict[str, Any]:`

## source/utils/hashing.py

SHA-256: `38656bb393d8a159ee93a4db701ab82b6867cc006cd5acb2ae057c79b789efd4`

60 lines. Definitions: hash_file, hash_directory, verify_hash

- Line 10: `def hash_file(path: str, algorithm: str = 'sha256') -> str:`
- Line 27: `def hash_directory(directory_path: str, algorithm: str = 'sha256') -> Dict[str, str]:`
- Line 46: `def verify_hash(path: str, expected_hash: str, algorithm: str = 'sha256') -> bool:`

## source/utils/logging_utils.py

SHA-256: `63dfd73eb18848bec3920ddf4b028890c2711862cddb6586ad5e79b6916d08d0`

67 lines. Definitions: setup_logger, log_to_jsonl

- Line 12: `def setup_logger(name: str, log_dir: Optional[str] = None, level: str = 'INFO') -> logging.Logger:`
- Line 51: `def log_to_jsonl(filepath: str, entry: Dict[str, Any]) -> None:`

## source/utils/quarantine.py

SHA-256: `4c1d1cab1311f5dd81ee13b25e9681d21cdadb319a8417d61eaa2ce0a3ca7131`

126 lines. Definitions: QuarantinedArtifactError, QuarantineEntry, QuarantineRegistry, assert_not_quarantined, __post_init__, __post_init__, load, is_quarantined, _normalise_path, assert_clean

- Line 23: `class QuarantinedArtifactError(Exception):`
- Line 29: `class QuarantineEntry:`
- Line 37: `    def __post_init__(self) -> None:`
- Line 54: `class QuarantineRegistry:`
- Line 60: `    def __post_init__(self):`
- Line 69: `    def load(cls, registry_path: Path = DEFAULT_REGISTRY_PATH) -> "QuarantineRegistry":`
- Line 79: `    def is_quarantined(self, target_path: str, target_hash: Optional[str] = None) -> Tuple[bool, Optional[str]]:`
- Line 96: `    def _normalise_path(target_path: str) -> str:`
- Line 110: `    def assert_clean(self, target_path: str, target_hash: Optional[str] = None) -> None:`
- Line 111: `        """Assert that an artifact is not quarantined. Raises QuarantinedArtifactError if tainted."""`
- Line 120: `def assert_not_quarantined(path_or_hash: str, registry_path: Path = DEFAULT_REGISTRY_PATH) -> None:`
- Line 121: `    """Convenience helper to assert that an artifact is admissible."""`

## source/utils/reproducibility.py

SHA-256: `707eb16f2ba908f39a588bf742ca2d0f60bbe825cb1cb45911d9ff7507e72b4b`

43 lines. Definitions: set_seed, get_seed_state

- Line 1: `"""Deterministic random seed setting and state retrieval for reproducibility.`
- Line 6: `import random`
- Line 12: `def set_seed(seed: int) -> None:`
- Line 13: `    """Set random seed across Python random, NumPy, PyTorch, and CUDA.`
- Line 19: `    random.seed(seed)`
- Line 20: `    np.random.seed(seed)`
- Line 32: `def get_seed_state() -> Dict[str, Any]:`
