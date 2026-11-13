# Localization benchmark

### all repositories: 150 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 15% | 43% | 0.57 | 0.291 | 72% | 27% | 28% | 8317 |
| bm25-chunks | 12% | 38% | 0.52 | 0.247 | 62% | 23% | 24% | 6214 |
| bm25-files | 6% | 27% | 0.40 | 0.162 | 13% | 11% | 10% | 9248 |
| grep | 5% | 32% | 0.45 | 0.169 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.044 [+0.017, +0.074] | 7 / 1 | 0.0703 | +0.04 [+0.01, +0.07] | 1.34 |
| bm25-files | +0.130 [+0.081, +0.180] | 28 / 1 | 0.0000 | +0.16 [+0.10, +0.22] | 0.90 |
| grep | +0.122 [+0.069, +0.178] | - | - | - | - |

### attrs: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 4% | 16% | 0.36 | 0.122 | 48% | 6% | 8% | 7063 |
| bm25-chunks | 0% | 4% | 0.28 | 0.058 | 36% | 6% | 8% | 5203 |
| bm25-files | 4% | 12% | 0.24 | 0.081 | 4% | 4% | 4% | 7307 |
| grep | 4% | 28% | 0.28 | 0.112 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.064 [-0.006, +0.163] | 0 / 0 | 1.0000 | +0.00 [+0.00, +0.00] | 1.36 |
| bm25-files | +0.041 [-0.080, +0.158] | 2 / 1 | 1.0000 | +0.02 [-0.08, +0.14] | 0.97 |
| grep | +0.010 [-0.113, +0.131] | - | - | - | - |

### click: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 20% | 36% | 0.60 | 0.284 | 67% | 32% | 32% | 9165 |
| bm25-chunks | 12% | 32% | 0.48 | 0.215 | 65% | 32% | 32% | 6986 |
| bm25-files | 4% | 16% | 0.29 | 0.143 | 16% | 16% | 16% | 9567 |
| grep | 16% | 36% | 0.45 | 0.241 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.070 [+0.023, +0.132] | 0 / 0 | 1.0000 | +0.00 [+0.00, +0.00] | 1.31 |
| bm25-files | +0.141 [+0.061, +0.244] | 4 / 0 | 0.1250 | +0.16 [+0.04, +0.32] | 0.96 |
| grep | +0.043 [-0.097, +0.189] | - | - | - | - |

### cobra: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 24% | 84% | 0.85 | 0.512 | 90% | 38% | 48% | 9646 |
| bm25-chunks | 32% | 84% | 0.82 | 0.533 | 89% | 33% | 40% | 7250 |
| bm25-files | 16% | 48% | 0.56 | 0.263 | 28% | 20% | 20% | 8290 |
| grep | 4% | 48% | 0.55 | 0.214 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | -0.021 [-0.085, +0.029] | 2 / 0 | 0.5000 | +0.05 [+0.00, +0.15] | 1.33 |
| bm25-files | +0.249 [+0.108, +0.399] | 7 / 0 | 0.0156 | +0.18 [+0.05, +0.34] | 1.16 |
| grep | +0.299 [+0.166, +0.444] | - | - | - | - |

### flask: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 8% | 12% | 0.30 | 0.135 | 58% | 27% | 28% | 8062 |
| bm25-chunks | 4% | 12% | 0.20 | 0.084 | 32% | 19% | 20% | 5729 |
| bm25-files | 4% | 8% | 0.16 | 0.078 | 4% | 4% | 4% | 9983 |
| grep | 0% | 16% | 0.24 | 0.085 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.051 [+0.013, +0.112] | 2 / 0 | 0.5000 | +0.08 [+0.00, +0.20] | 1.41 |
| bm25-files | +0.057 [-0.065, +0.177] | 6 / 0 | 0.0312 | +0.23 [+0.08, +0.40] | 0.81 |
| grep | +0.050 [-0.039, +0.157] | - | - | - | - |

### ky: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 24% | 56% | 0.65 | 0.384 | 83% | 28% | 24% | 7924 |
| bm25-chunks | 20% | 56% | 0.59 | 0.359 | 73% | 18% | 16% | 6234 |
| bm25-files | 4% | 48% | 0.57 | 0.217 | 11% | 7% | 4% | 12406 |
| grep | 0% | 36% | 0.63 | 0.176 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.025 [-0.013, +0.075] | 3 / 1 | 0.6250 | +0.10 [-0.00, +0.23] | 1.27 |
| bm25-files | +0.166 [+0.046, +0.310] | 5 / 0 | 0.0625 | +0.21 [+0.07, +0.37] | 0.64 |
| grep | +0.208 [+0.066, +0.365] | - | - | - | - |

### requests: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 12% | 52% | 0.68 | 0.311 | 84% | 28% | 28% | 8039 |
| bm25-chunks | 4% | 40% | 0.76 | 0.234 | 74% | 28% | 28% | 5884 |
| bm25-files | 4% | 32% | 0.56 | 0.187 | 16% | 12% | 12% | 7933 |
| grep | 4% | 28% | 0.54 | 0.187 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.077 [-0.001, +0.176] | 0 / 0 | 1.0000 | +0.00 [+0.00, +0.00] | 1.37 |
| bm25-files | +0.125 [+0.017, +0.242] | 4 / 0 | 0.1250 | +0.16 [+0.04, +0.32] | 1.01 |
| grep | +0.124 [+0.020, +0.241] | - | - | - | - |
