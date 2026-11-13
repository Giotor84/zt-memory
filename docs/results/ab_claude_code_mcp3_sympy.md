## Claude Code A/B - 2 bugs, same model, with and without zt-memory

Each cell: mean / median per bug. Ratio: median over bugs of (zt arm / plain).

| measure | plain | zt-loc | zt-focus | zt-loc-span | zt-loc / plain | zt-focus / plain | zt-loc-span / plain |
|---|---|---|---|---|---|---|---|
| bugs fixed (tests pass) | 2/2 | 2/2 | 2/2 | 2/2 | | | |
| cost per bug (USD) | 0.0514 / 0.0514 | 0.0842 / 0.0842 | 0.0853 / 0.0853 | 0.0744 / 0.0744 | 1.47 | 1.53 | 1.38 |
| tokens read by the model | 7.38e+04 / 7.38e+04 | 1.28e+05 / 1.28e+05 | 1.38e+05 / 1.38e+05 | 1.3e+05 / 1.3e+05 | 1.51 | 1.76 | 1.72 |
| output tokens | 1.57e+03 / 1.57e+03 | 2.43e+03 / 2.43e+03 | 2.23e+03 / 2.23e+03 | 1.83e+03 / 1.83e+03 | 1.43 | 1.37 | 1.13 |
| largest single context | 1.31e+04 / 1.31e+04 | 1.91e+04 / 1.91e+04 | 1.96e+04 / 1.96e+04 | 1.84e+04 / 1.84e+04 | 1.40 | 1.44 | 1.38 |
| API turns | 7.5 / 7.5 | 9.5 / 9.5 | 8 / 8 | 8 / 8 | 1.18 | 1.12 | 1.12 |
| files opened with Read | 0 / 0 | 0.5 / 0.5 | 0 / 0 | 0 / 0 | - | - | - |
| wall time (s) | 30.4 / 30.4 | 41.4 / 41.4 | 37.4 / 37.4 | 49.4 / 49.4 | 1.29 | 1.16 | 1.56 |

- zt-loc: cheaper than plain on 0/2 bugs (sign test p = 0.5); fixed only with zt-loc: 0, only without: 0; zt_query_context calls per bug 0.5 (never called on 1)
- zt-focus: cheaper than plain on 0/2 bugs (sign test p = 0.5); fixed only with zt-focus: 0, only without: 0; zt_query_context calls per bug 0.5 (never called on 1)
- zt-loc-span: cheaper than plain on 0/2 bugs (sign test p = 0.5); fixed only with zt-loc-span: 0, only without: 0; zt_query_context calls per bug 1.0 (never called on 0); zt_read_span calls per bug 0.5 (used on 1)
- runs that touched the benchmark's own clone (contamination): 0
