# SHS Matched Long-Decode Matrix

Status: **passed**

| Cell | P1 tok/s | P8 tok/s | P16 tok/s | P32 tok/s | P64 tok/s |
|---|---:|---:|---:|---:|---:|
| naive_qwen | 143.5 | 952.1 | 1878.8 | 3373.1 | 5329.7 |
| shs_reference | 40.2 | 312.1 | 615.6 | 1181.0 | 2262.8 |
| shs_triton_fast | 48.5 | 350.8 | 686.5 | 1181.7 | 1725.1 |

Strict Triton is unavailable because the fast kernel failed the unchanged A1 parity gate.
