# SHS Matched 50-Step SFT Runtime Matrix

Status: **passed**

| Cell | Median step (s) | p10 | p90 | Assistant tok/s | Peak GB | Projected 3916 steps (h) |
|---|---:|---:|---:|---:|---:|---:|
| reference_eager | 1.4526 | 1.4447 | 1.4571 | 7730.0 | 18.44 | 1.58 |
| reference_compile | 1.4698 | 1.4686 | 1.4722 | 7638.3 | 11.38 | 1.60 |
| triton_recompute_eager | 3.5948 | 3.5897 | 3.6094 | 3122.5 | 14.85 | 3.91 |
| triton_recompute_compile | 3.3034 | 3.2983 | 3.3081 | 3399.1 | 9.78 | 3.59 |

Triton cells use reference-recompute backward and do not claim a custom backward kernel.
