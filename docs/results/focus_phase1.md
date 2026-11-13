# Contesto a fuoco — fase 1

### commit — 150 casi

| variante | token medi | vs A | punto nel corpo | punto almeno nella mappa | blocchi interi / firme (medi) |
|---|---|---|---|---|---|
| A oggi | 8771 | +0% | 92/150 (61%) | 92/150 (61%) | 35.2 / 0.0 |
| B anti-cloni | 8538 | -3% | 92/150 (61%) | 92/150 (61%) | 33.7 / 0.1 |
| C fuoco F=3 | 2760 | -69% | 68/150 (45%) | 92/150 (61%) | 7.6 / 25.2 |
| D fuoco F=5 | 3072 | -65% | 73/150 (49%) | 92/150 (61%) | 8.7 / 24.1 |
| E = D + reinvestimento | 3679 | -58% | 73/150 (49%) | 97/150 (65%) | 8.7 / 44.1 |

| confronto (punto nel corpo) | vinti | persi | McNemar p |
|---|---|---|---|
| B contro A (effetto anti-cloni) | 0 | 0 | 1 |
| C contro C0 (anti-cloni a parità di F=3) | 0 | 0 | 1 |
| D contro D0 (anti-cloni a parità di F=5) | 3 | 1 | 0.625 |
| C contro A | 0 | 24 | 1.19e-07 |
| D contro A | 0 | 19 | 3.81e-06 |
| C contro B (costo del fuoco F=3) | 0 | 24 | 1.19e-07 |
| D contro B (costo del fuoco F=5) | 0 | 19 | 3.81e-06 |
| E contro A | 0 | 19 | 3.81e-06 |
| mappa: E contro D (resa del reinvestimento) | 5 | 0 | 0.0625 |

| curva del fuoco (oltre la griglia) | token medi | vs A | punto nel corpo | persi contro A (McNemar p) |
|---|---|---|---|---|
| F=3 senza anti-cloni | 2810 | -68% | 68/150 (45%) | 24 (1.19e-07) |
| F=3 | 2760 | -69% | 68/150 (45%) | 24 (1.19e-07) |
| F=5 senza anti-cloni | 3101 | -65% | 71/150 (47%) | 21 (9.54e-07) |
| F=5 | 3072 | -65% | 73/150 (49%) | 19 (3.81e-06) |
| F=10 | 4062 | -54% | 83/150 (55%) | 9 (0.00391) |
| F=15 | 5132 | -41% | 87/150 (58%) | 5 (0.0625) |

### italiano — 150 casi

| variante | token medi | vs A | punto nel corpo | punto almeno nella mappa | blocchi interi / firme (medi) |
|---|---|---|---|---|---|
| A oggi | 8317 | +0% | 42/150 (28%) | 42/150 (28%) | 41.9 / 0.0 |
| B anti-cloni | 7964 | -4% | 42/150 (28%) | 42/150 (28%) | 38.3 / 0.1 |
| C fuoco F=3 | 2516 | -70% | 19/150 (13%) | 42/150 (28%) | 7.9 / 28.7 |
| D fuoco F=5 | 2742 | -67% | 21/150 (14%) | 42/150 (28%) | 8.9 / 27.7 |
| E = D + reinvestimento | 3326 | -60% | 21/150 (14%) | 49/150 (33%) | 8.9 / 46.5 |

| confronto (punto nel corpo) | vinti | persi | McNemar p |
|---|---|---|---|
| B contro A (effetto anti-cloni) | 0 | 0 | 1 |
| C contro C0 (anti-cloni a parità di F=3) | 0 | 0 | 1 |
| D contro D0 (anti-cloni a parità di F=5) | 2 | 2 | 1 |
| C contro A | 0 | 23 | 2.38e-07 |
| D contro A | 0 | 21 | 9.54e-07 |
| C contro B (costo del fuoco F=3) | 0 | 23 | 2.38e-07 |
| D contro B (costo del fuoco F=5) | 0 | 21 | 9.54e-07 |
| E contro A | 0 | 21 | 9.54e-07 |
| mappa: E contro D (resa del reinvestimento) | 7 | 0 | 0.0156 |

| curva del fuoco (oltre la griglia) | token medi | vs A | punto nel corpo | persi contro A (McNemar p) |
|---|---|---|---|---|
| F=3 senza anti-cloni | 2610 | -69% | 19/150 (13%) | 23 (2.38e-07) |
| F=3 | 2516 | -70% | 19/150 (13%) | 23 (2.38e-07) |
| F=5 senza anti-cloni | 2799 | -66% | 21/150 (14%) | 21 (9.54e-07) |
| F=5 | 2742 | -67% | 21/150 (14%) | 21 (9.54e-07) |
| F=10 | 3470 | -58% | 23/150 (15%) | 19 (3.81e-06) |
| F=15 | 4266 | -49% | 35/150 (23%) | 7 (0.0156) |

### traceback — 42 casi

| variante | token medi | vs A | punto nel corpo | punto almeno nella mappa | blocchi interi / firme (medi) |
|---|---|---|---|---|---|
| A oggi | 8477 | +0% | 25/42 (60%) | 25/42 (60%) | 33.4 / 0.0 |
| B anti-cloni | 8193 | -3% | 25/42 (60%) | 25/42 (60%) | 31.1 / 0.1 |
| C fuoco F=3 | 3842 | -55% | 23/42 (55%) | 25/42 (60%) | 6.2 / 24.3 |
| D fuoco F=5 | 4064 | -52% | 24/42 (57%) | 25/42 (60%) | 6.9 / 23.6 |
| E = D + reinvestimento | 4811 | -43% | 24/42 (57%) | 27/42 (64%) | 6.9 / 43.6 |

| confronto (punto nel corpo) | vinti | persi | McNemar p |
|---|---|---|---|
| B contro A (effetto anti-cloni) | 0 | 0 | 1 |
| C contro C0 (anti-cloni a parità di F=3) | 0 | 0 | 1 |
| D contro D0 (anti-cloni a parità di F=5) | 1 | 0 | 1 |
| C contro A | 0 | 2 | 0.5 |
| D contro A | 0 | 1 | 1 |
| C contro B (costo del fuoco F=3) | 0 | 2 | 0.5 |
| D contro B (costo del fuoco F=5) | 0 | 1 | 1 |
| E contro A | 0 | 1 | 1 |
| mappa: E contro D (resa del reinvestimento) | 2 | 0 | 0.5 |

| curva del fuoco (oltre la griglia) | token medi | vs A | punto nel corpo | persi contro A (McNemar p) |
|---|---|---|---|---|
| F=3 senza anti-cloni | 3944 | -53% | 23/42 (55%) | 2 (0.5) |
| F=3 | 3842 | -55% | 23/42 (55%) | 2 (0.5) |
| F=5 senza anti-cloni | 4146 | -51% | 23/42 (55%) | 2 (0.5) |
| F=5 | 4064 | -52% | 24/42 (57%) | 1 (1) |
| F=10 | 4255 | -50% | 25/42 (60%) | 0 (1) |
| F=15 | 4317 | -49% | 25/42 (60%) | 0 (1) |

