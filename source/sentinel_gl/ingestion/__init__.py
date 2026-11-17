"""Acquisition helpers used to build the Sentinel-2 corpus.

``expansion2023`` holds the HTTP range-request, transfer-ledger and COG-window
reader used by the SR-1 acquisition runners; the Sentinel-1 and CDSE modules
it imports were used for the earlier 2023 acquisition epoch whose optical
outputs are part of the corpus. No module here synthesizes observations.
"""
