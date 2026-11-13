## Claude Code A/B - 8 bugs, same model, with and without zt-memory

Each cell: mean / median per bug. Ratio: median over bugs of (zt arm / plain).

| measure | plain | zt-loc | zt-focus | zt-loc-span | zt-loc / plain | zt-focus / plain | zt-loc-span / plain |
|---|---|---|---|---|---|---|---|
| bugs fixed (tests pass) | 7/8 | 7/8 | 6/8 | 5/8 | | | |
| cost per bug (USD) | 0.0424 / 0.0342 | 0.0486 / 0.0425 | 0.0452 / 0.0414 | 0.0421 / 0.043 | 1.12 | 1.15 | 1.20 |
| tokens read by the model | 5.18e+04 / 3.94e+04 | 5.55e+04 / 4.35e+04 | 5.28e+04 / 4.42e+04 | 5.16e+04 / 4.65e+04 | 1.10 | 1.04 | 1.12 |
| output tokens | 1.56e+03 / 1.36e+03 | 1.07e+03 / 986 | 1.04e+03 / 878 | 909 / 928 | 0.67 | 0.69 | 0.63 |
| largest single context | 1.13e+04 / 1.01e+04 | 1.55e+04 / 1.51e+04 | 1.51e+04 / 1.44e+04 | 1.46e+04 / 1.45e+04 | 1.41 | 1.38 | 1.43 |
| API turns | 6 / 5 | 4.12 / 3.5 | 4.38 / 4 | 4.25 / 3.5 | 0.69 | 0.78 | 0.75 |
| files opened with Read | 1 / 1 | 0.375 / 0 | 0.375 / 0 | 0.25 / 0 | 0.00 | 0.00 | 0.00 |
| wall time (s) | 32.9 / 27.8 | 26 / 25.9 | 31.9 / 22 | 25.3 / 25.9 | 0.81 | 0.84 | 0.87 |

- zt-loc: cheaper than plain on 2/8 bugs (sign test p = 0.289); fixed only with zt-loc: 0, only without: 0; zt_query_context calls per bug 1.0 (never called on 0)
- zt-focus: cheaper than plain on 2/8 bugs (sign test p = 0.289); fixed only with zt-focus: 0, only without: 1; zt_query_context calls per bug 1.0 (never called on 0)
- zt-loc-span: cheaper than plain on 3/8 bugs (sign test p = 0.727); fixed only with zt-loc-span: 0, only without: 2; zt_query_context calls per bug 1.0 (never called on 0); zt_read_span calls per bug 0.2 (used on 2)
- runs that touched the benchmark's own clone (contamination): 0
