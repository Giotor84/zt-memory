# Localization benchmark

### all repositories: 150 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 40% | 68% | 0.75 | 0.535 | 79% | 61% | 61% | 8771 |
| bm25-chunks | 39% | 67% | 0.76 | 0.516 | 77% | 58% | 58% | 7714 |
| bm25-files | 19% | 60% | 0.70 | 0.383 | 28% | 21% | 19% | 12320 |
| grep | 12% | 61% | 0.76 | 0.326 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.020 [-0.010, +0.050] | 5 / 0 | 0.0625 | +0.03 [+0.01, +0.05] | 1.14 |
| bm25-files | +0.152 [+0.086, +0.215] | 65 / 2 | 0.0000 | +0.40 [+0.32, +0.48] | 0.71 |
| grep | +0.209 [+0.141, +0.271] | - | - | - | - |

### attrs: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 32% | 40% | 0.46 | 0.371 | 58% | 51% | 52% | 6181 |
| bm25-chunks | 28% | 40% | 0.58 | 0.356 | 54% | 43% | 44% | 5250 |
| bm25-files | 12% | 36% | 0.42 | 0.216 | 24% | 20% | 20% | 12102 |
| grep | 16% | 40% | 0.54 | 0.302 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.015 [-0.011, +0.059] | 2 / 0 | 0.5000 | +0.08 [-0.00, +0.20] | 1.18 |
| bm25-files | +0.155 [-0.013, +0.331] | 8 / 0 | 0.0078 | +0.31 [+0.15, +0.49] | 0.51 |
| grep | +0.069 [-0.112, +0.262] | - | - | - | - |

### click: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 40% | 64% | 0.80 | 0.528 | 88% | 69% | 72% | 9449 |
| bm25-chunks | 48% | 64% | 0.80 | 0.569 | 84% | 69% | 72% | 8609 |
| bm25-files | 12% | 64% | 0.72 | 0.373 | 16% | 12% | 12% | 12704 |
| grep | 12% | 68% | 0.84 | 0.330 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | -0.041 [-0.121, +0.038] | 0 / 0 | 1.0000 | +0.00 [+0.00, +0.00] | 1.10 |
| bm25-files | +0.156 [+0.030, +0.292] | 15 / 0 | 0.0001 | +0.57 [+0.39, +0.76] | 0.74 |
| grep | +0.198 [+0.037, +0.352] | - | - | - | - |

### cobra: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 48% | 84% | 0.98 | 0.665 | 98% | 68% | 76% | 10715 |
| bm25-chunks | 32% | 80% | 0.98 | 0.518 | 98% | 66% | 72% | 10261 |
| bm25-files | 36% | 80% | 0.93 | 0.546 | 60% | 32% | 32% | 12132 |
| grep | 16% | 92% | 0.97 | 0.442 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.147 [+0.057, +0.250] | 1 / 0 | 1.0000 | +0.02 [+0.00, +0.05] | 1.04 |
| bm25-files | +0.118 [-0.083, +0.311] | 12 / 1 | 0.0034 | +0.36 [+0.14, +0.57] | 0.88 |
| grep | +0.223 [+0.057, +0.379] | - | - | - | - |

### flask: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 32% | 48% | 0.56 | 0.405 | 60% | 43% | 44% | 8488 |
| bm25-chunks | 28% | 48% | 0.52 | 0.378 | 60% | 43% | 44% | 6827 |
| bm25-files | 8% | 36% | 0.48 | 0.228 | 20% | 20% | 20% | 11864 |
| grep | 0% | 36% | 0.44 | 0.160 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.028 [-0.002, +0.075] | 0 / 0 | 1.0000 | +0.00 [+0.00, +0.00] | 1.24 |
| bm25-files | +0.177 [+0.063, +0.304] | 7 / 1 | 0.0703 | +0.23 [+0.04, +0.43] | 0.72 |
| grep | +0.245 [+0.119, +0.380] | - | - | - | - |

### ky: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 60% | 96% | 0.96 | 0.759 | 96% | 70% | 64% | 9800 |
| bm25-chunks | 64% | 96% | 0.94 | 0.765 | 94% | 68% | 60% | 8997 |
| bm25-files | 36% | 88% | 0.89 | 0.578 | 32% | 24% | 16% | 12531 |
| grep | 16% | 76% | 0.89 | 0.401 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | -0.006 [-0.078, +0.064] | 1 / 0 | 1.0000 | +0.02 [+0.00, +0.05] | 1.09 |
| bm25-files | +0.181 [+0.010, +0.349] | 12 / 0 | 0.0005 | +0.47 [+0.30, +0.64] | 0.78 |
| grep | +0.358 [+0.198, +0.509] | - | - | - | - |

### requests: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 28% | 76% | 0.76 | 0.483 | 76% | 64% | 60% | 7993 |
| bm25-chunks | 36% | 72% | 0.72 | 0.508 | 72% | 60% | 56% | 6337 |
| bm25-files | 12% | 56% | 0.74 | 0.359 | 16% | 16% | 16% | 12589 |
| grep | 12% | 52% | 0.86 | 0.322 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | -0.025 [-0.111, +0.052] | 1 / 0 | 1.0000 | +0.04 [+0.00, +0.12] | 1.26 |
| bm25-files | +0.124 [-0.018, +0.269] | 11 / 0 | 0.0010 | +0.48 [+0.30, +0.66] | 0.63 |
| grep | +0.161 [+0.013, +0.312] | - | - | - | - |
