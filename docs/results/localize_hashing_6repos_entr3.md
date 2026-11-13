# Localization benchmark

### all repositories: 150 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 29% | 67% | 0.73 | 0.457 | 80% | 54% | 56% | 10004 |
| bm25-chunks | 29% | 63% | 0.72 | 0.434 | 77% | 49% | 50% | 8460 |
| bm25-chunks-matched | 29% | 63% | 0.72 | 0.434 | 77% | 52% | 55% | 9754 |
| bm25-files | 23% | 56% | 0.65 | 0.380 | 30% | 21% | 22% | 12283 |
| grep | 14% | 62% | 0.71 | 0.329 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.023 [-0.008, +0.054] | 9 / 0 | 0.0039 | +0.05 [+0.02, +0.08] | 1.18 |
| bm25-chunks-matched | +0.023 [-0.008, +0.054] | 3 / 1 | 0.6250 | +0.02 [-0.00, +0.05] | 1.03 |
| bm25-files | +0.077 [+0.016, +0.141] | 55 / 4 | 0.0000 | +0.33 [+0.26, +0.41] | 0.81 |
| grep | +0.128 [+0.066, +0.191] | - | - | - | - |

### attrs: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 20% | 36% | 0.38 | 0.283 | 56% | 33% | 40% | 7967 |
| bm25-chunks | 16% | 32% | 0.38 | 0.252 | 54% | 29% | 32% | 6861 |
| bm25-chunks-matched | 16% | 32% | 0.38 | 0.252 | 56% | 36% | 44% | 7891 |
| bm25-files | 12% | 28% | 0.42 | 0.208 | 16% | 16% | 16% | 12197 |
| grep | 12% | 52% | 0.58 | 0.290 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.032 [+0.004, +0.078] | 2 / 0 | 0.5000 | +0.04 [+0.00, +0.12] | 1.16 |
| bm25-chunks-matched | +0.032 [+0.004, +0.078] | 0 / 1 | 1.0000 | -0.03 [-0.12, +0.04] | 1.01 |
| bm25-files | +0.075 [-0.084, +0.239] | 8 / 2 | 0.1094 | +0.17 [-0.03, +0.38] | 0.65 |
| grep | -0.006 [-0.136, +0.127] | - | - | - | - |

### click: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 32% | 76% | 0.79 | 0.501 | 83% | 60% | 64% | 10926 |
| bm25-chunks | 32% | 76% | 0.83 | 0.492 | 83% | 60% | 64% | 9887 |
| bm25-chunks-matched | 32% | 76% | 0.83 | 0.492 | 83% | 60% | 64% | 10897 |
| bm25-files | 32% | 68% | 0.71 | 0.486 | 41% | 24% | 28% | 12480 |
| grep | 20% | 64% | 0.75 | 0.389 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.009 [-0.105, +0.117] | 0 / 0 | 1.0000 | +0.00 [+0.00, +0.00] | 1.11 |
| bm25-chunks-matched | +0.009 [-0.105, +0.117] | 0 / 0 | 1.0000 | +0.01 [+0.00, +0.03] | 1.00 |
| bm25-files | +0.015 [-0.146, +0.171] | 9 / 0 | 0.0039 | +0.36 [+0.18, +0.56] | 0.88 |
| grep | +0.112 [-0.027, +0.255] | - | - | - | - |

### cobra: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 32% | 88% | 0.93 | 0.579 | 93% | 65% | 68% | 11106 |
| bm25-chunks | 32% | 76% | 0.89 | 0.498 | 93% | 60% | 64% | 9909 |
| bm25-chunks-matched | 32% | 76% | 0.89 | 0.498 | 93% | 60% | 64% | 10959 |
| bm25-files | 36% | 80% | 0.89 | 0.555 | 58% | 36% | 36% | 12071 |
| grep | 36% | 84% | 0.87 | 0.557 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.081 [-0.026, +0.196] | 1 / 0 | 1.0000 | +0.05 [+0.00, +0.13] | 1.12 |
| bm25-chunks-matched | +0.081 [-0.026, +0.196] | 1 / 0 | 1.0000 | +0.05 [+0.00, +0.13] | 1.01 |
| bm25-files | +0.024 [-0.140, +0.180] | 9 / 1 | 0.0215 | +0.29 [+0.09, +0.48] | 0.92 |
| grep | +0.022 [-0.155, +0.194] | - | - | - | - |

### flask: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 12% | 48% | 0.54 | 0.273 | 72% | 41% | 44% | 10201 |
| bm25-chunks | 12% | 36% | 0.58 | 0.250 | 64% | 32% | 36% | 7816 |
| bm25-chunks-matched | 12% | 36% | 0.58 | 0.250 | 64% | 40% | 44% | 9407 |
| bm25-files | 8% | 20% | 0.28 | 0.172 | 8% | 8% | 8% | 11870 |
| grep | 4% | 40% | 0.46 | 0.176 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.023 [+0.005, +0.043] | 2 / 0 | 0.5000 | +0.09 [+0.00, +0.21] | 1.31 |
| bm25-chunks-matched | +0.023 [+0.005, +0.043] | 0 / 0 | 1.0000 | +0.01 [+0.00, +0.03] | 1.08 |
| bm25-files | +0.101 [-0.004, +0.224] | 9 / 0 | 0.0039 | +0.33 [+0.16, +0.51] | 0.86 |
| grep | +0.097 [-0.017, +0.229] | - | - | - | - |

### ky: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 56% | 92% | 0.93 | 0.700 | 95% | 68% | 68% | 10618 |
| bm25-chunks | 52% | 100% | 0.91 | 0.696 | 93% | 62% | 56% | 8994 |
| bm25-chunks-matched | 52% | 100% | 0.91 | 0.696 | 93% | 62% | 64% | 10596 |
| bm25-files | 28% | 88% | 0.87 | 0.481 | 28% | 19% | 24% | 12771 |
| grep | 4% | 68% | 0.85 | 0.285 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | +0.004 [-0.040, +0.061] | 3 / 0 | 0.2500 | +0.05 [+0.00, +0.14] | 1.18 |
| bm25-chunks-matched | +0.004 [-0.040, +0.061] | 1 / 0 | 1.0000 | +0.05 [+0.00, +0.14] | 1.00 |
| bm25-files | +0.219 [+0.053, +0.386] | 12 / 1 | 0.0034 | +0.49 [+0.31, +0.67] | 0.83 |
| grep | +0.416 [+0.290, +0.543] | - | - | - | - |

### requests: 25 cases

| method | right file 1st | right file in top 5 | recall@10 | MRR | right file in context | changed lines in context | fix spot in context | mean tokens |
|---|---|---|---|---|---|---|---|---|
| zt | 20% | 64% | 0.84 | 0.405 | 84% | 56% | 52% | 9209 |
| bm25-chunks | 28% | 60% | 0.76 | 0.416 | 76% | 50% | 48% | 7294 |
| bm25-chunks-matched | 28% | 60% | 0.76 | 0.416 | 76% | 52% | 48% | 8773 |
| bm25-files | 20% | 52% | 0.74 | 0.377 | 26% | 21% | 20% | 12307 |
| grep | 8% | 64% | 0.77 | 0.276 | - | - | - | - |

| zt vs | MRR diff [95% CI] | fix spot: only zt / only other | McNemar p | changed lines diff [95% CI] | tokens zt / other |
|---|---|---|---|---|---|
| bm25-chunks | -0.011 [-0.091, +0.063] | 1 / 0 | 1.0000 | +0.06 [+0.00, +0.16] | 1.26 |
| bm25-chunks-matched | -0.011 [-0.091, +0.063] | 1 / 0 | 1.0000 | +0.04 [+0.00, +0.12] | 1.05 |
| bm25-files | +0.028 [-0.099, +0.161] | 8 / 0 | 0.0078 | +0.35 [+0.18, +0.53] | 0.75 |
| grep | +0.129 [+0.002, +0.262] | - | - | - | - |
