"""Early exit (CALM-style) for the T5 student.

Modules:
    - :mod:`src.early_exit.heads` -- read word predictions out of any decoder layer
    - :mod:`src.early_exit.confidence` -- per-layer confidence measures
    - :mod:`src.early_exit.analysis` -- Checkpoint 1: per-layer accuracy/agreement report
"""
