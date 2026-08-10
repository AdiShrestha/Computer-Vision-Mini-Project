# Encoder Architecture Specification (Path B — Custom Time-Series MAE)

## Overview
This document specifies the custom **Time-Series Masked Autoencoder (TS-MAE)** architecture selected in `project/evolution/decision_log.md` (Decision 002). This specification serves as the blueprint for Chunk 03 implementation.

---

## 1. Input & Output Tensor Specifications

- **Input Tensor Shape**: `(batch_size, n_windows, n_channels)`
  - `n_windows`: 107 (rolling 180-day time windows with 30-day stride over 2016–2024 extent per INV-004)
  - `n_channels`: 15 (CH-01 area, CH-02 spectral x4, CH-03 velocity x2, CH-04 temp, CH-05 SAR x3, CH-07 coherence, CH-08 meteorology x3)
- **Output Embedding Shape**: `(batch_size, n_windows, d_model)`
  - `d_model`: 128 (latent embedding dimension per window)
- **Bottleneck Global Vector**: `(batch_size, d_model)` (mean-pooled temporal representation for downstream anomaly scoring)

---

## 2. Model Architecture Details

```text
[Input Matrix (B, 107, 15)] 
        │
        ▼
[Linear Patch Projection: 15 -> d_model (128)]
        │
        ├── Add Learned 1D Positional Embeddings (107, 128)
        │
        ▼
[Random 50% Patch Masking (INV-005)] ──> Keep unmasked patches (B, 54, 128)
        │
        ▼
[Transformer Encoder (4 layers, 8 heads, d_ff=512, dropout=0.1)]
        │
        ▼
[Latent Representations (B, 54, 128)]
        │
        ├── Re-insert Mask Tokens for 53 masked patches
        │
        ▼
[Transformer Decoder (2 layers, 4 heads, d_ff=256)]
        │
        ▼
[Linear Reconstruction Head: d_model (128) -> 15] ──> Reconstructed Matrix (B, 107, 15)
```

---

## 3. Training & Pretraining Parameters

- **Pretraining Objective**: Masked Feature Reconstruction Loss (MSE over masked tokens only)
- **Masking Strategy**: 50% random patch masking along the temporal axis (INV-005)
- **Optimizer**: AdamW (`lr=1e-3`, `weight_decay=0.05`, cosine decay scheduler)
- **Batch Size**: 16 lakes per batch
- **Deterministic Random Seed**: `set_seed(42)` (INV-012)
- **Compute Resource Profile**:
  - Model Parameters: ~1.25 Million parameters
  - VRAM Footprint: ~1.8 GB VRAM (well within 12GB INV-008 limit)
  - Estimated Pretraining Time: < 30 minutes on Apple M3 / Google Colab T4

---

## 4. References & Decision Trace
- Decision Rationale: `project/evolution/decision_log.md` (Decision 002)
- Data Quality Basis: `results/data_quality/data_quality_report.md`
- Invariant Compliance: INV-004, INV-005, INV-008, INV-012
