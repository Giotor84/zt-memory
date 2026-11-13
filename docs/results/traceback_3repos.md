## Traceback anchors - 42 bugs (3 repos)

- logs naming at least one indexed file:line: 42/42
- logs naming a line in a file the fix changes: 8/42

| method | fix spot in context | mean tokens |
|---|---|---|
| zt | 25/42 (60%) | 8477 |
| zt-no-anchors | 22/42 (52%) | 8355 |
| bm25-chunks | 25/42 (60%) | 8372 |
| bm25-chunks-no-anchors | 21/42 (50%) | 8118 |
| bm25-files | 11/42 (26%) | 11988 |

- zt vs zt-no-anchors: 3 cases won, 0 lost (McNemar p = 0.25)
- bm25-chunks vs bm25-chunks-no-anchors: 4 cases won, 0 lost (McNemar p = 0.125)
- zt vs bm25-files: 14 cases won, 0 lost (McNemar p = 0.000122)
