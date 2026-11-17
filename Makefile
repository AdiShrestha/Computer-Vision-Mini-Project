# Sentinel-GL: one documented command sequence for the PLOS ONE analysis.
# Stages write into immutable directories and refuse to overwrite existing outputs.
PY ?= python3
export PYTHONPATH := source

.PHONY: help install test acquire zones train score replay packet figures references manuscript verify deposit all-from-deposit

help:
	@echo "install            create the hash-locked environment (python -m pip install --require-hashes -r requirements.lock)"
	@echo "test               unit and regression tests on constructed inputs (no data needed)"
	@echo "acquire            re-read the South Lhonak windows from Earth Search (network; needs the attempt_006 provenance files)"
	@echo "zones|train|score  analysis stages under amendment PLOS-001 (need the corpus from the data deposit)"
	@echo "replay             independent replay of predictions and Score A from saved checkpoints"
	@echo "packet figures manuscript verify   result packet -> figures -> DOCX -> data-driven verification"
	@echo "all-from-deposit   replay packet figures manuscript verify (uses the deposited checkpoints and scores)"

install:
	$(PY) -m pip install --require-hashes -r requirements.lock

test:
	$(PY) -B -m pytest -q -p no:cacheprovider source/tests

acquire:
	$(PY) -B source/runners/run_sr1_recentre_south_lhonak.py --attempt attempt_007

zones:
	$(PY) -B source/runners/run_sr1_plos_analysis.py zones

train:
	$(PY) -B source/runners/run_sr1_plos_analysis.py train

score:
	$(PY) -B source/runners/run_sr1_plos_analysis.py score

replay:
	$(PY) -B tools/replay_sr1_plos.py

packet:
	$(PY) -B tools/build_sr1_plos_packet.py

figures:
	$(PY) -B tools/build_sr1_plos_figures.py

references:
	$(PY) -B tools/verify_references.py

manuscript:
	$(PY) -B tools/build_plos_manuscript.py

verify:
	$(PY) -B tools/verify_plos_package.py

deposit:
	$(PY) -B tools/build_deposit.py

all-from-deposit: replay packet figures manuscript verify
