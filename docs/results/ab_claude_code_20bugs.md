## Claude Code A/B - 20 bugs, same model, with and without zt-memory

Each cell: mean / median per bug. Ratio: median over bugs of (zt arm / plain).

| measure | plain | zt | zt-4k | zt / plain | zt-4k / plain |
|---|---|---|---|---|---|
| bugs fixed (tests pass) | 20/20 | 20/20 | 19/20 | | |
| cost per bug (USD) | 0.0689 / 0.0577 | 0.0851 / 0.0916 | 0.0658 / 0.0581 | 1.37 | 1.07 |
| tokens read by the model | 1.05e+05 / 7.42e+04 | 1.22e+05 / 1.11e+05 | 1.03e+05 / 8.5e+04 | 1.44 | 1.12 |
| output tokens | 2.53e+03 / 2.05e+03 | 1.62e+03 / 1.49e+03 | 1.82e+03 / 1.45e+03 | 0.73 | 0.75 |
| largest single context | 1.44e+04 / 1.37e+04 | 2.19e+04 / 2.17e+04 | 1.78e+04 / 1.73e+04 | 1.57 | 1.36 |
| API turns | 9.6 / 8 | 7 / 6.5 | 7.5 / 7 | 0.75 | 0.82 |
| files opened with Read | 0.9 / 1 | 0.2 / 0 | 0.35 / 0 | 0.00 | 0.00 |
| wall time (s) | 40.8 / 32 | 36.5 / 33.6 | 36.9 / 31.2 | 1.02 | 0.98 |

- zt: cheaper than plain on 3/20 bugs (sign test p = 0.00258); fixed only with zt: 0, only without: 0; zt_query_context calls per bug 0.9 (never called on 2)
- zt-4k: cheaper than plain on 8/20 bugs (sign test p = 0.503); fixed only with zt-4k: 0, only without: 1; zt_query_context calls per bug 0.8 (never called on 3)
- runs that touched the benchmark's own clone (contamination): 0
