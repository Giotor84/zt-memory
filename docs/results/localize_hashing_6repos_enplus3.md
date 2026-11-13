# Localization benchmark

### all repositories: 150 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 59% | 76% | 0.82 | 0.677 | 84% | 69% | 69% | 8620 |
| bm25-chunks | 59% | 75% | 0.80 | 0.662 | 85% | 69% | 69% | 8167 |
| bm25-chunks-matched | 59% | 75% | 0.80 | 0.662 | 85% | 69% | 71% | 8525 |
| bm25-files | 54% | 78% | 0.80 | 0.638 | 60% | 49% | 49% | 12026 |
| grep | 36% | 73% | 0.82 | 0.519 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.015 [-0.011, +0.043] | 1 / 1 | 1.0000 | +0.00 [-0.02, +0.02] | 1.06 |
| bm25-chunks-matched | +0.015 [-0.011, +0.043] | 0 / 2 | 0.5000 | +0.00 [-0.02, +0.02] | 1.01 |
| bm25-files | +0.039 [-0.007, +0.084] | 34 / 3 | 0.0000 | +0.20 [+0.13, +0.27] | 0.72 |
| grep | +0.159 [+0.098, +0.221] | - | - | - | - |

### attrs: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 40% | 52% | 0.70 | 0.480 | 70% | 61% | 60% | 7057 |
| bm25-chunks | 44% | 52% | 0.62 | 0.494 | 70% | 57% | 56% | 6840 |
| bm25-chunks-matched | 44% | 52% | 0.62 | 0.494 | 70% | 60% | 60% | 6943 |
| bm25-files | 36% | 60% | 0.62 | 0.482 | 58% | 45% | 48% | 12086 |
| grep | 40% | 68% | 0.70 | 0.520 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | -0.014 [-0.082, +0.052] | 1 / 0 | 1.0000 | +0.04 [-0.01, +0.12] | 1.03 |
| bm25-chunks-matched | -0.014 [-0.082, +0.052] | 0 / 0 | 1.0000 | +0.01 [-0.01, +0.04] | 1.02 |
| bm25-files | -0.002 [-0.099, +0.095] | 4 / 1 | 0.3750 | +0.16 [-0.02, +0.33] | 0.58 |
| grep | -0.040 [-0.155, +0.076] | - | - | - | - |

### click: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 76% | 84% | 0.88 | 0.805 | 88% | 81% | 84% | 8358 |
| bm25-chunks | 76% | 84% | 0.88 | 0.805 | 88% | 81% | 84% | 8436 |
| bm25-chunks-matched | 76% | 84% | 0.88 | 0.805 | 88% | 81% | 84% | 8335 |
| bm25-files | 72% | 80% | 0.84 | 0.765 | 71% | 55% | 56% | 11979 |
| grep | 52% | 80% | 0.83 | 0.661 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.000 [+0.000, +0.000] | 0 / 0 | 1.0000 | +0.00 [+0.00, +0.00] | 0.99 |
| bm25-chunks-matched | +0.000 [+0.000, +0.000] | 0 / 0 | 1.0000 | +0.00 [+0.00, +0.00] | 1.00 |
| bm25-files | +0.040 [-0.037, +0.125] | 8 / 1 | 0.0391 | +0.26 [+0.06, +0.46] | 0.70 |
| grep | +0.144 [+0.011, +0.287] | - | - | - | - |

### cobra: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 72% | 88% | 0.93 | 0.802 | 93% | 79% | 84% | 10127 |
| bm25-chunks | 68% | 88% | 0.93 | 0.765 | 93% | 79% | 84% | 9629 |
| bm25-chunks-matched | 68% | 88% | 0.93 | 0.765 | 93% | 79% | 84% | 10044 |
| bm25-files | 64% | 96% | 0.93 | 0.738 | 70% | 50% | 56% | 11991 |
| grep | 60% | 92% | 0.93 | 0.714 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.038 [-0.037, +0.113] | 0 / 0 | 1.0000 | +0.00 [+0.00, +0.01] | 1.05 |
| bm25-chunks-matched | +0.038 [-0.037, +0.113] | 0 / 0 | 1.0000 | +0.00 [+0.00, +0.01] | 1.01 |
| bm25-files | +0.064 [-0.052, +0.184] | 7 / 0 | 0.0156 | +0.29 [+0.11, +0.48] | 0.84 |
| grep | +0.088 [-0.046, +0.219] | - | - | - | - |

### flask: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 40% | 60% | 0.68 | 0.506 | 80% | 58% | 56% | 9860 |
| bm25-chunks | 44% | 56% | 0.64 | 0.509 | 88% | 62% | 60% | 8668 |
| bm25-chunks-matched | 44% | 56% | 0.64 | 0.509 | 88% | 58% | 60% | 9593 |
| bm25-files | 40% | 56% | 0.64 | 0.485 | 44% | 44% | 44% | 11911 |
| grep | 24% | 60% | 0.78 | 0.399 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | -0.003 [-0.060, +0.052] | 0 / 1 | 1.0000 | -0.04 [-0.12, +0.00] | 1.14 |
| bm25-chunks-matched | -0.003 [-0.060, +0.052] | 0 / 1 | 1.0000 | +0.00 [-0.11, +0.09] | 1.03 |
| bm25-files | +0.021 [-0.073, +0.112] | 3 / 0 | 0.2500 | +0.14 [+0.02, +0.28] | 0.83 |
| grep | +0.106 [-0.022, +0.246] | - | - | - | - |

### ky: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 84% | 100% | 0.98 | 0.913 | 98% | 73% | 72% | 8808 |
| bm25-chunks | 80% | 100% | 0.96 | 0.880 | 94% | 70% | 72% | 8374 |
| bm25-chunks-matched | 80% | 100% | 0.96 | 0.880 | 96% | 73% | 76% | 8777 |
| bm25-files | 60% | 96% | 0.94 | 0.728 | 55% | 45% | 36% | 12170 |
| grep | 8% | 72% | 0.93 | 0.327 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.033 [-0.040, +0.113] | 0 / 0 | 1.0000 | +0.02 [+0.00, +0.07] | 1.05 |
| bm25-chunks-matched | +0.033 [-0.040, +0.113] | 0 / 1 | 1.0000 | -0.00 [-0.10, +0.08] | 1.00 |
| bm25-files | +0.185 [+0.062, +0.319] | 9 / 0 | 0.0039 | +0.27 [+0.13, +0.43] | 0.72 |
| grep | +0.586 [+0.448, +0.701] | - | - | - | - |

### requests: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 44% | 72% | 0.76 | 0.557 | 75% | 62% | 60% | 7509 |
| bm25-chunks | 40% | 68% | 0.76 | 0.520 | 75% | 62% | 60% | 7051 |
| bm25-chunks-matched | 40% | 68% | 0.76 | 0.520 | 75% | 62% | 60% | 7456 |
| bm25-files | 52% | 80% | 0.84 | 0.633 | 61% | 56% | 52% | 12017 |
| grep | 32% | 68% | 0.79 | 0.489 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.037 [-0.047, +0.127] | 0 / 0 | 1.0000 | +0.00 [+0.00, +0.00] | 1.06 |
| bm25-chunks-matched | +0.037 [-0.047, +0.127] | 0 / 0 | 1.0000 | +0.00 [+0.00, +0.00] | 1.01 |
| bm25-files | -0.076 [-0.197, +0.038] | 3 / 1 | 0.6250 | +0.07 [-0.08, +0.24] | 0.62 |
| grep | +0.068 [-0.058, +0.200] | - | - | - | - |
