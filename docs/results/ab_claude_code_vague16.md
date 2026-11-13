## Claude Code A/B - 16 bugs, same model, with and without zt-memory

Each cell: mean / median per bug. Ratio: median over bugs of (zt arm / plain).

| measure | plain | zt-4k | zt-loc | zt-4k / plain | zt-loc / plain |
|---|---|---|---|---|---|
| bugs fixed (tests pass) | 13/16 | 15/16 | 15/16 | | |
| cost per bug (USD) | 0.0483 / 0.041 | 0.0574 / 0.0579 | 0.0616 / 0.0441 | 1.24 | 1.02 |
| tokens read by the model | 6.32e+04 / 5.55e+04 | 7.09e+04 / 6.83e+04 | 9.4e+04 / 5.94e+04 | 1.09 | 1.16 |
| output tokens | 1.77e+03 / 1.57e+03 | 1.25e+03 / 992 | 2.2e+03 / 1.24e+03 | 0.68 | 0.87 |
| largest single context | 1.21e+04 / 1.11e+04 | 1.6e+04 / 1.54e+04 | 1.42e+04 / 1.26e+04 | 1.34 | 1.09 |
| API turns | 7.19 / 7 | 5.5 / 5 | 8.12 / 6.5 | 0.78 | 1.00 |
| files opened with Read | 1.06 / 1 | 0.25 / 0 | 1.25 / 1 | 0.00 | 1.00 |
| wall time (s) | 26.4 / 22.1 | 23.4 / 20.1 | 29.8 / 19.1 | 0.90 | 0.94 |

- zt-4k: cheaper than plain on 4/16 bugs (sign test p = 0.0768); fixed only with zt-4k: 2, only without: 0; zt_query_context calls per bug 1.0 (never called on 0)
- zt-loc: cheaper than plain on 8/16 bugs (sign test p = 1); fixed only with zt-loc: 2, only without: 0; zt_query_context calls per bug 1.0 (never called on 0)
- runs that touched the benchmark's own clone (contamination): 0
