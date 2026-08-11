# Contract Report — C07-00

## Objective
"Correct the stale text in `invariants.md` INV-011 that still references CH-06 (InSAR deformation) for synthetic anomaly type 3. The actual implementation (Decision 003, Chunk 04) already substituted CH-05 SAR backscatter (+3 dB step), but the invariant document was never updated to match."

## Contract Information
- **Contract ID**: C07-00
- **Chunk ID**: chunk07
- **Objective (quoted verbatim)**: "Correct the stale text in `invariants.md` INV-011 that still references CH-06 (InSAR deformation) for synthetic anomaly type 3. The actual implementation (Decision 003, Chunk 04) already substituted CH-05 SAR backscatter (+3 dB step), but the invariant document was never updated to match."
- **Risk Tier**: Low
- **Implementation Owner**: Architect
- **Model Identifier**: claude-3-5-sonnet

## Scope / Inputs / Outputs
- **Inputs**:
  - `source/evaluation/synthetic/injector.py` (line 42)
  - `project/evolution/decision_log.md` (Decision 003)
- **Outputs**:
  - `project/invariants.md` (updated INV-011 text with dated revision note)
  - `project/evolution/decision_log.md` (appended Decision 004)
  - `source/tests/test_chunk07.py` (3 verification unit tests)

## Files Modified
| File | Purpose | Reason Modified | Major Changes |
|---|---|---|---|
| `project/invariants.md` | Core invariants specification | Updated | Corrected INV-011 type 3 text to reference CH-05 SAR backscatter (+3 dB step) and added dated revision note C07-00 |
| `project/evolution/decision_log.md` | Architecture decision log | Appended | Appended Decision 004 documenting INV-011 text correction |
| `source/tests/test_chunk07.py` | Verification test suite | New file | Implemented 3 verification tests for C07-00 |

## Verification
- **Command**: `pytest source/tests/test_chunk07.py`
- **Output**: 3 / 3 passed in 0.01s.
- **Full Suite Command**: `pytest`
- **Result**: PASS (201 / 201 passed).

## Human Action Status
```text
Human Action Required: false
Status: INV-011 text corrected to match actual implementation and Decision 003.
Blocks: NONE. Ready for C07-01.
```

## Evidence
- Updated INV-011 type 3 definition in `invariants.md`.
- Appended Decision 004 to `decision_log.md`.
- Verified `pytest source/tests/test_chunk07.py`: All 3 DoD unit tests PASSED.

## Definition of Done Verification
1. INV-011 type 3 references CH-05 with +3 dB magnitude and 6 windows duration — **Satisfied**.
2. INV-011 type 3 does NOT reference CH-06 — **Satisfied**.
3. Dated revision note exists in `invariants.md` referencing C07-00 and Decision 003 — **Satisfied**.
4. Decision 004 logged in `decision_log.md` — **Satisfied**.
5. All three verification tests pass — **Satisfied** (3/3 PASS).

## Invariant Status
- **INV-011 (Synthetic Anomaly Types)**: Corrected & Aligned.
- **C30 (Version History)**: Preserved.

## Final Status
`COMPLETE`

## Plain-Language Summary
Corrected INV-011 invariant text in `project/invariants.md` to reference CH-05 (+3 dB SAR backscatter step) matching Decision 003 and `injector.py`. Logged Decision 004 in `project/evolution/decision_log.md`. Verified all DoD requirements with a 3/3 unit test pass.
