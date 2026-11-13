## Claude Code A/B - 9 bugs, same model, with and without zt-memory

Each cell: mean / median per bug. Ratio: median over bugs of (zt arm / plain).

| measure | plain | zt-4k | zt-loc | zt-4k / plain | zt-loc / plain |
|---|---|---|---|---|---|
| bugs fixed (tests pass) | 8/9 | 7/9 | 9/9 | | |
| cost per bug (USD) | 0.114 / 0.0851 | 0.11 / 0.0699 | 0.0847 / 0.0699 | 1.16 | 0.96 |
| tokens read by the model | 1.65e+05 / 4.88e+04 | 1.61e+05 / 7.93e+04 | 1.38e+05 / 1.08e+05 | 1.18 | 1.14 |
| output tokens | 3.88e+03 / 1.23e+03 | 3.24e+03 / 1.46e+03 | 2.88e+03 / 2.33e+03 | 0.86 | 0.95 |
| largest single context | 1.95e+04 / 1.65e+04 | 2.25e+04 / 1.89e+04 | 1.77e+04 / 1.61e+04 | 1.20 | 1.07 |
| API turns | 10 / 6 | 8.44 / 5 | 10.1 / 8 | 0.80 | 1.00 |
| files opened with Read | 0.444 / 0 | 0.222 / 0 | 1.44 / 1 | 0.00 | 1.00 |
| wall time (s) | 172 / 41.5 | 64.5 / 24.8 | 75.6 / 41.7 | 0.82 | 0.96 |

- zt-4k: cheaper than plain on 4/9 bugs (sign test p = 1); fixed only with zt-4k: 0, only without: 1; zt_query_context calls per bug 1.0 (never called on 0)
- zt-loc: cheaper than plain on 5/9 bugs (sign test p = 1); fixed only with zt-loc: 1, only without: 0; zt_query_context calls per bug 0.9 (never called on 1)
- runs that touched the benchmark's own clone (contamination): 0
