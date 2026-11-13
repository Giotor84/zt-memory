# Contesto a fuoco — fase 1b

### commit — 150 casi; A trova il punto nel corpo in 92

- ground truth in un file di test: 0 casi
- blocchi (chunk) scelti da A: media 33.1, di produzione 21.1
- chunk senza simbolo (intestazioni, blocchi di modulo) tra i primi 10 di produzione: media 3.5

| posizione del blocco giusto (A, quando lo trova) | casi (contando solo produzione) | casi (contando tutti i chunk) |
|---|---|---|
| 1 | 40 | 31 |
| 2 | 14 | 15 |
| 3 | 7 | 5 |
| 4-5 | 8 | 8 |
| 6-10 | 14 | 18 |
| 11-15 | 4 | 3 |
| 16-20 | 3 | 3 |
| 21-30 | 2 | 8 |
| 31+ | 0 | 1 |

- posizione mediana 2 (produzione), 2 (tutti); entro la 5ª: 69/92; oltre la 10ª: 9/92
- punteggio del blocco giusto rispetto al migliore: mediana 0.95; sotto 0,6: 25/92
- calo di punteggio dal 1° al 5° candidato: mediana 0.37 (quote: <0,2 12, <0,35 62 su 147)

| variante | token medi | vs A | punto nel corpo | persi / vinti contro A | McNemar p | entro il criterio (≥35% token in meno, ≤2 punti persi) |
|---|---|---|---|---|---|---|
| A | 8771 | -0% | 92/150 (61%) | 0 / 0 | 1 | - |
| B | 8538 | -3% | 92/150 (61%) | 0 / 0 | 1 | - |
| D | 3072 | -65% | 73/150 (49%) | 19 / 0 | 3.81e-06 | no |
| F15 | 5132 | -41% | 87/150 (58%) | 5 / 0 | 0.0625 | no |
| M50 | 3481 | -60% | 79/150 (53%) | 13 / 0 | 0.000244 | no |
| M70 | 4645 | -47% | 83/150 (55%) | 9 / 0 | 0.00391 | no |
| M85 | 5608 | -36% | 89/150 (59%) | 3 / 0 | 0.25 | sì |
| G20 | 3170 | -64% | 74/150 (49%) | 18 / 0 | 7.63e-06 | no |
| G35 | 3828 | -56% | 78/150 (52%) | 14 / 0 | 0.000122 | no |
| Dx | 3108 | -65% | 70/150 (47%) | 22 / 0 | 4.77e-07 | no |
| F15x | 5162 | -41% | 81/150 (54%) | 11 / 0 | 0.000977 | no |
| M70x | 4598 | -48% | 79/150 (53%) | 13 / 0 | 0.000244 | no |
| M85x | 5524 | -37% | 83/150 (55%) | 9 / 0 | 0.00391 | no |

### italiano — 150 casi; A trova il punto nel corpo in 42

- ground truth in un file di test: 0 casi
- blocchi (chunk) scelti da A: media 38.9, di produzione 21.6
- chunk senza simbolo (intestazioni, blocchi di modulo) tra i primi 10 di produzione: media 2.8

| posizione del blocco giusto (A, quando lo trova) | casi (contando solo produzione) | casi (contando tutti i chunk) |
|---|---|---|
| 1 | 5 | 3 |
| 2 | 4 | 1 |
| 3 | 6 | 2 |
| 4-5 | 5 | 4 |
| 6-10 | 4 | 10 |
| 11-15 | 10 | 5 |
| 16-20 | 5 | 7 |
| 21-30 | 2 | 3 |
| 31+ | 1 | 7 |

- posizione mediana 6 (produzione), 11 (tutti); entro la 5ª: 20/42; oltre la 10ª: 18/42
- punteggio del blocco giusto rispetto al migliore: mediana 0.53; sotto 0,6: 23/42
- calo di punteggio dal 1° al 5° candidato: mediana 0.35 (quote: <0,2 16, <0,35 72 su 145)

| variante | token medi | vs A | punto nel corpo | persi / vinti contro A | McNemar p | entro il criterio (≥35% token in meno, ≤2 punti persi) |
|---|---|---|---|---|---|---|
| A | 8317 | -0% | 42/150 (28%) | 0 / 0 | 1 | - |
| B | 7964 | -4% | 42/150 (28%) | 0 / 0 | 1 | - |
| D | 2742 | -67% | 21/150 (14%) | 21 / 0 | 9.54e-07 | no |
| F15 | 4266 | -49% | 35/150 (23%) | 7 / 0 | 0.0156 | no |
| M50 | 3129 | -62% | 24/150 (16%) | 18 / 0 | 7.63e-06 | no |
| M70 | 3990 | -52% | 29/150 (19%) | 13 / 0 | 0.000244 | no |
| M85 | 4863 | -42% | 39/150 (26%) | 3 / 0 | 0.25 | sì |
| G20 | 2831 | -66% | 23/150 (15%) | 19 / 0 | 3.81e-06 | no |
| G35 | 3331 | -60% | 23/150 (15%) | 19 / 0 | 3.81e-06 | no |
| Dx | 2729 | -67% | 21/150 (14%) | 21 / 0 | 9.54e-07 | no |
| F15x | 4278 | -49% | 34/150 (23%) | 8 / 0 | 0.00781 | no |
| M70x | 3881 | -53% | 27/150 (18%) | 15 / 0 | 6.1e-05 | no |
| M85x | 4742 | -43% | 36/150 (24%) | 6 / 0 | 0.0312 | no |

### traceback — 42 casi; A trova il punto nel corpo in 25

- ground truth in un file di test: 0 casi
- blocchi (chunk) scelti da A: media 31.7, di produzione 4.1
- chunk senza simbolo (intestazioni, blocchi di modulo) tra i primi 10 di produzione: media 0.9

| posizione del blocco giusto (A, quando lo trova) | casi (contando solo produzione) | casi (contando tutti i chunk) |
|---|---|---|
| 1 | 11 | 6 |
| 2 | 3 | 0 |
| 3 | 1 | 0 |
| 4-5 | 0 | 4 |
| 6-10 | 2 | 4 |
| 11-15 | 0 | 4 |
| 16-20 | 0 | 1 |
| 21-30 | 0 | 4 |
| 31+ | 0 | 0 |

- posizione mediana 1 (produzione), 7 (tutti); entro la 5ª: 15/17; oltre la 10ª: 0/17
- punteggio del blocco giusto rispetto al migliore: mediana 1.00; sotto 0,6: 2/23
- calo di punteggio dal 1° al 5° candidato: mediana 0.34 (quote: <0,2 0, <0,35 7 su 13)

| variante | token medi | vs A | punto nel corpo | persi / vinti contro A | McNemar p | entro il criterio (≥35% token in meno, ≤2 punti persi) |
|---|---|---|---|---|---|---|
| A | 8477 | -0% | 25/42 (60%) | 0 / 0 | 1 | - |
| B | 8193 | -3% | 25/42 (60%) | 0 / 0 | 1 | - |
| D | 4064 | -52% | 24/42 (57%) | 1 / 0 | 1 | no |
| F15 | 4317 | -49% | 25/42 (60%) | 0 / 0 | 1 | sì |
| M50 | 3946 | -53% | 23/42 (55%) | 2 / 0 | 0.5 | no |
| M70 | 4104 | -52% | 24/42 (57%) | 1 / 0 | 1 | no |
| M85 | 4277 | -50% | 24/42 (57%) | 1 / 0 | 1 | no |
| G20 | 4064 | -52% | 24/42 (57%) | 1 / 0 | 1 | no |
| G35 | 4147 | -51% | 24/42 (57%) | 1 / 0 | 1 | no |
| Dx | 4070 | -52% | 24/42 (57%) | 1 / 0 | 1 | no |
| F15x | 4259 | -50% | 24/42 (57%) | 1 / 0 | 1 | no |
| M70x | 4073 | -52% | 23/42 (55%) | 2 / 0 | 0.5 | no |
| M85x | 4221 | -50% | 24/42 (57%) | 1 / 0 | 1 | no |

