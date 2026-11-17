# Clean-environment reproduction (2026-11-17)

* Code: `git archive` of commit 6dc4ab9f21be10e300201144702b1733c3311307 (`release/sentinel_gl_code_v1.zip`), unpacked into an empty directory.
* Data: `release/sentinel_gl_data_v1.zip` (SHA-256 6f3349e0ac049c5244509445d5c1ed669280ae843d04e15728a0f74529720388) unpacked at the root; all files of `MANIFEST.sha256` verified (0 failures).
* Hi-MAG archive placed as documented (doi:10.5281/zenodo.4275164; SHA-256 checked by the code).
* Environment: new virtual environment, Python 3.12, `pip install --require-hashes -r requirements.lock` (via uv).
* `make test`: 26 passed.
* `make all-from-deposit`: independent replay of all 10 primary checkpoints with maximum absolute prediction difference 0.0 and Score A difference 0.0; the recomputed result packet equals `docs/results/sr1_plos_packet.json` apart from its generation timestamp; Figs 1–5 are byte-identical; the manuscript verifier returned PASS.

Limitation: this ran on the same Apple M3 machine (macOS arm64) as the original analysis, in a fresh directory and environment. It was not repeated on a second platform, and re-training (Level 2 in REPRODUCE.md) was not repeated.
