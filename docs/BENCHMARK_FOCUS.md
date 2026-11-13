# Contesto a fuoco — fase 1 (offline, 20/10/2026)

Idea: mandare al modello il **corpo** solo dei blocchi che contano e, per gli altri, solo **firma + prima frase della
docstring + righe**; prima togliere i cloni (stub `.pyi`/`.d.ts` quando c'è l'implementazione, quasi-duplicati,
ordine MMR solo tra simboli). Nessuna chiamata a LLM: si misura se il punto che la vera correzione ha cambiato
finisce **nel corpo** (il modello può correggerlo) o **almeno nella mappa** (vede dove guardare).

`examples/benchmark_focus.py`, encoder leggero, budget 12000, stesso indice per tutte le varianti di un caso
(un solo giro: l'energia dei simboli dipende dall'ordine dei sync, varianti di giri diversi non sono confrontabili).

Varianti: **A** oggi · **B** anti-cloni (MMR λ=0,3 + stub + duplicati > 0,95), tutto intero · **C** B + fuoco F=3 ·
**D** B + fuoco F=5 · **E** D + fino a 20 firme in più con i token risparmiati · fuori griglia: **C0/D0** fuoco senza
anti-cloni, **F=10, F=15**. Fuoco: corpo per i blocchi fissati dal traceback, i primi F scelti e quelli con punteggio
≥ 0,6 × il migliore; stub e test mai a fuoco; un blocco resta intero se la firma costerebbe quanto il corpo.

## Risultati

| | domanda = messaggio di commit (150) | | domanda in italiano (150) | | domanda = log di pytest (42) | |
|---|---|---|---|---|---|---|
| variante | token | punto nel corpo | token | punto nel corpo | token | punto nel corpo |
| A oggi | 8771 | 92 (61%) | 8317 | 42 (28%) | 8477 | 25 (60%) |
| B anti-cloni | **-3%** | 92 (61%) | **-4%** | 42 (28%) | **-3%** | 25 (60%) |
| C fuoco F=3 | -69% | 68 (45%) | -70% | 19 (13%) | -55% | 23 (55%) |
| D fuoco F=5 | -65% | 73 (49%) | -67% | 21 (14%) | -52% | 24 (57%) |
| F=10 | -54% | 83 (55%) | -58% | 23 (15%) | **-50%** | **25 (60%)** |
| F=15 | -41% | 87 (58%) | -49% | 35 (23%) | -49% | 25 (60%) |
| E = D + firme | -58% | 73 (49%), mappa 97 (65%) | -60% | 21 (14%), mappa 49 (33%) | -43% | 24 (57%), mappa 27 (64%) |

Test di McNemar (punto nel corpo, casi vinti / persi):

| confronto | commit | italiano | traceback |
|---|---|---|---|
| B contro A (anti-cloni) | 0 / 0 | 0 / 0 | 0 / 0 |
| C contro C0 (anti-cloni a parità di F=3) | 0 / 0 | 0 / 0 | 0 / 0 |
| D contro D0 (anti-cloni a parità di F=5) | 3 / 1 (p 0,63) | 2 / 2 | 1 / 0 |
| C contro A | 0 / 24 (p < 0,001) | 0 / 23 (p < 0,001) | 0 / 2 (p 0,5) |
| D contro A | 0 / 19 (p < 0,001) | 0 / 21 (p < 0,001) | 0 / 1 |
| F=10 contro A | 0 / 9 (p 0,004) | 0 / 19 (p < 0,001) | **0 / 0** |
| F=15 contro A | 0 / 5 (p 0,06) | 0 / 7 (p 0,02) | 0 / 0 |
| mappa: E contro D (resa delle firme in più) | 5 / 0 (p 0,06) | 7 / 0 (p 0,02) | 2 / 0 |

## Cosa vuol dire

1. **Anti-cloni: -3/-4% di token, nessun caso perso** su 342 domande. Poco ma gratis. A parità di F non porta il punto
   giusto più in alto (0 a 0 con F=3, 6 a 3 in totale con F=5: non significativo).
2. **Fuoco con F=3 o 5: taglia due terzi dei token ma perde 1 punto giusto su 5** nelle domande descrittive. Il punto
   da correggere, quando c'è, spesso non è tra i primi blocchi: la classifica non è abbastanza netta.
   **Criterio della fase 1 (≥ 40% di token in meno, ≤ 2 punti persi) non raggiunto** per commit e italiano
   (F=15: -41% ma -3,3 punti; -49% ma -4,7 punti).
3. **Con un traceback il fuoco funziona**: F=10 dimezza i token (-50%) **senza perdere nessun caso** (25/42 come oggi):
   il punto giusto è già fissato in testa.
4. **Le firme in più (E)** allargano la mappa (+5, +7, +2 casi) per circa 600 token: utile solo se il modello può poi
   chiedere il corpo di ciò che vede in mappa.

## Prossimi passi possibili

- Anti-cloni attivi di default (-3/4%, nessuna perdita misurata).
- Fuoco automatico **solo quando la domanda contiene un traceback** (F=10: -50% a parità di risultato su 42 casi).
- Per le domande descrittive, fase 2 con il modello: fuoco F=5 + mappa, e il modello può chiedere il corpo di un
  blocco in mappa (`NEED file:Lx-y`, un giro in più). La mappa contiene il punto giusto quanto oggi (61%).

Dati: `results/focus_phase1.{md,jsonl}`.

## Fase 1b — dove sta il blocco giusto e fuoco adattivo (`examples/benchmark_focus_adaptive.py`)

**Contatore F.** Scorre la classifica e salta test e stub: F vuol dire F blocchi di produzione. Ground truth in file di
test: **0 casi** in tutti e tre i set (i casi escludono i test dai file corretti), quindi la regola non falsa la metrica.
Le intestazioni e il codice di modulo tra i simboli (`<header>`, `<gap:...>`) invece contano: in media 3,5 dei primi 10
blocchi di produzione (commit). Toglierli dal conteggio peggiora (D: 73 → 70; M85: 89 → 83): la correzione sta spesso
proprio lì. Regola scartata (`focus_skip_glue`, spenta).

**Posizione del blocco giusto in A, quando A lo trova:**

| | commit (92 casi) | italiano (42 casi) | log pytest |
|---|---|---|---|
| chunk scelti da A (di produzione) | 33,1 (21,1) | 38,9 (21,6) | 31,7 (4,1) |
| posizione mediana (produzione) | **2** | **6** | 1 |
| entro la 5ª / oltre la 10ª | 69 / 9 | 20 / **18** | 15 / 0 (+6 casi già fissati dal traceback) |
| posizioni 6–10 / 11–15 | 14 / 4 | 4 / **10** | 2 / 0 |
| punteggio del blocco giusto / migliore (mediana) | 0,95 | **0,53** | 1,00 |

Con le domande in italiano il blocco giusto ha un punteggio mediocre e sta spesso fra l'11ª e la 15ª posizione: per
questo F=10 non aggiunge quasi nulla e F=15 salta al 23%. Il problema lì è la classifica, non il taglio.

**Fuoco adattivo** (sempre con anti-cloni, F minimo 3, rapporto 0,6):

| variante | commit: token / punto nel corpo / persi | italiano | log pytest |
|---|---|---|---|
| A oggi | 8771 / 92 / – | 8317 / 42 / – | 8477 / 25 / – |
| F=15 | -41% / 87 / 5 (p 0,06) | -49% / 35 / 7 (p 0,02) | -49% / 25 / **0** |
| massa 70% (M70) | -47% / 83 / 9 | -52% / 29 / 13 | -52% / 24 / 1 |
| **massa 85% (M85)** | **-36% / 89 / 3 (p 0,25)** | **-42% / 39 / 3 (p 0,25)** | **-50% / 24 / 1** |
| calo piatto → F=15 (G20, G35) | -64%, -56% / 74, 78 / 18, 14 | -66%, -60% / 23, 23 / 19, 19 | -52% / 24 / 1 |

- **M85 rispetta il criterio** (≥ 35% di token in meno, ≤ 2 punti persi) su commit e italiano; sui log di pytest perde 1
  caso su 42 (2,4 punti), mentre F=15 non ne perde nessuno.
- La regola sul "calo piatto" non riconosce i casi difficili: bocciata.
- M85: i blocchi di produzione vanno interi finché coprono l'85% del punteggio totale dei blocchi scelti (minimo 3);
  gli altri diventano firme.

Dati: `results/focus_phase1b.{md,jsonl}`.
