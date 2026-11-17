# Changes made after the amendment PLOS-001 freeze

The freeze record (`SR1_AMENDMENT_PLOS_001.freeze.json`) binds the code that trained and scored the models; the training and scoring stages verified those hashes before running. Changes made afterwards are listed here. None alters a checkpoint, a prediction row, a score, a status or a pre-declared threshold.

| Date (UTC) | File | Change | Seen results first? | Effect |
|---|---|---|---|---|
| 2026-11-17 | `tools/build_sr1_plos_packet.py` | Added a separate VAL-only column to the scene-level score summaries and a count of autumn comparisons in which the EVAL median exceeds the FIT-autumn median. | Yes. FIT scenes are in-sample for the model, so the held-out pre-event VAL scenes are the more informative reference for Table 4; this became clear when writing the results. | Reporting only; the pre-declared FIT/VAL, FIT-autumn and EVAL columns are unchanged. |
| 2026-11-17 | `tools/build_sr1_plos_packet.py` | The final status line prints the packet path as given instead of a repository-relative path, so the verifier can recompute the packet into a temporary directory. | Yes | None on any value. |
| 2026-11-17 | `tools/build_sr1_plos_packet.py` | Added the largest scene-level (seed-median) Score A per site and split, so a sentence in the Discussion comparing pre- and post-event maxima is checked against data. | Yes | Reporting only. |
