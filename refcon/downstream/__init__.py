# REFCON downstream callers (run on REFCON's reference-free CN predictions):
#   markers.py  - SCEVAN-port confident-diploid detection (marker enrichment on expression)
#   classify.py - call_sample: confident-diploid -> GMM cluster -> Pearson label (T=0.75),
#                 with a pure-sample dispersion fallback (T_sigma=0.135)
# Import as a package, e.g. `from refcon.downstream.classify import call_sample`
# (see scripts/classify.py).
