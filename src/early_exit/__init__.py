"""Early exit (CALM-style) for the T5 student.

Modules:
    - :mod:`src.early_exit.heads` -- read word predictions out of any decoder layer
    - :mod:`src.early_exit.confidence` -- stopping rules (softmax gap, static, oracle)
    - :mod:`src.early_exit.thresholds` -- exit threshold (constant)
    - :mod:`src.early_exit.analysis` -- Checkpoint 1: per-layer accuracy/agreement report
    - :mod:`src.early_exit.decode` -- Part 2: early-exit greedy translator
    - :mod:`src.early_exit.run_eval` -- Part 2: speed/quality sweep, CSV + chart
"""
