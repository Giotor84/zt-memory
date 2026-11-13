# Claude Code con e senza zt-memory (A/B su bug reali)

`examples/ab_claude_code.py`: la stessa sessione di Claude Code senza interfaccia (`claude -p`, stesso modello
Sonnet 5.5, stessi strumenti Bash/Read/Edit/Write/Glob/Grep) corregge bug veri, con e senza il server MCP di zt.

- **Compiti**: 20 bug di attrs (9), click (9) e flask (2) presi da `benchmark_bugfix.py mine`: commit in cui i test
  falliscono prima della correzione e passano dopo.
- **Richiesta** (uguale per tutti, in italiano): l'output di pytest dei test che falliscono, poi "trova la causa e
  correggi il codice sorgente, non modificare i test".
- **Repository**: copia al commit precedente la correzione, **senza storia git** (la correzione non si può cercare nel
  log). Le sessioni non hanno mai letto il clone originale (controllato nei transcript).
- **Varianti**: `plain` senza zt; `zt` con il server MCP e una riga di system prompt "usa prima zt_query_context"
  (come una nota in memoria); `zt-4k` uguale ma con risposte di zt al massimo di 4000 token invece di 12000;
  `zt-loc` con `mode="locations"` (solo file, righe, firme, ≤ 500 token), provata sui test con descrizioni a parole.
- **Voto**: si rimettono i test originali della correzione e si eseguono.

## Richiesta = log di pytest (18/10/2026, 60 sessioni, 4,40 $)

| misura (media / mediana per bug) | senza zt | zt 12k | zt 4k |
|---|---|---|---|
| bug corretti (test verdi) | 20/20 | 20/20 | 19/20 |
| costo per bug | 0,069 / 0,058 $ | 0,085 / 0,092 $ | 0,066 / 0,058 $ |
| token letti dal modello | 105k / 74k | 122k / 111k | 103k / 85k |
| contesto più grande in una chiamata | 14,4k | 21,9k | 17,8k |
| turni | 9,6 / 8 | 7 / 6,5 | 7,5 / 7 |
| file aperti con Read | 0,9 | 0,2 | 0,35 |
| tempo | 41 / 32 s | 37 / 34 s | 37 / 31 s |

- **Qualità uguale**: senza zt Claude corregge già 20 bug su 20. L'unico mancato con zt 4k è un caso in cui l'agente
  ha corretto un altro file e ha controllato solo una parte dei test.
- **Con 12000 token zt costa di più**: +37% per bug (mediana), più economico solo in 3 bug su 20 (p = 0.003). Fa
  risparmiare turni (−25%) e letture di file, ma la sua risposta (~7700 token) resta nel contesto e viene riletta a
  ogni turno successivo: più di quanto risparmia.
- **Con 4000 token il costo è pari** (+7% mediana, più economico in 8 bug su 20, non significativo).
- Per questo `zt mcp` ora risponde **al massimo con 4000 token** se il client non chiede altro (`token_budget`), o
  con il valore di `zt mcp --budget N`.

## Bug descritti a parole, in italiano (16 bug di attrs, click, flask; 48 sessioni)

Stessi progetti, ma la richiesta è una descrizione del sintomo come la scriverebbe un utente, senza log e senza nomi di
file o funzioni interne (`examples/data/ab_vague_it.json`); i test della correzione **restano nascosti** e servono solo
per il voto.

| misura (media / mediana per bug) | senza zt | zt 4k | zt-loc |
|---|---|---|---|
| bug corretti | 13/16 | 15/16 | 15/16 |
| costo per bug | 0,048 / 0,041 $ | 0,057 / 0,058 $ | 0,062 / 0,044 $ |
| mediana dei rapporti di costo con "senza zt" | | +24% | **+2%** |
| più economico di "senza zt" | | 4/16 | 8/16 |
| turni | 7,2 / 7 | 5,5 / 5 | 8,1 / 6,5 |
| file aperti con Read | 1,1 | 0,25 | 1,25 |

- 2 bug corretti solo con zt (entrambe le varianti), 0 solo senza: non significativo. Anche senza zt Claude aveva
  trovato e modificato il file giusto; la differenza è nel dettaglio della correzione, non nella ricerca.
- La media di `zt-loc` è alzata da un solo caso (attrs 97f8d17565: 27 turni, 0,28 $); la mediana è pari a "senza zt".

## Progetto grande: sympy (1609 file Python, 9 bug descritti a parole; 27 sessioni)

| misura (media / mediana per bug) | senza zt | zt 4k | zt-loc |
|---|---|---|---|
| bug corretti | 8/9 | 7/9 | 9/9 |
| costo per bug | 0,114 / 0,085 $ | 0,110 / 0,070 $ | 0,085 / 0,070 $ |
| mediana dei rapporti di costo con "senza zt" | | +16% | **-4%** |
| turni | 10 / 6 | 8,4 / 5 | 10,1 / 8 |

- In tutti i casi l'agente ha modificato il file giusto, con e senza zt: anche in un progetto di 1600 file il nome
  pubblico di una funzione (sinc, acot, refine, fcode) basta a Grep.
- zt costa meno nei due bug lunghi (21–27 turni senza zt: -20% e -11%) e quasi sempre di più in quelli brevi (4–6
  turni: da +17% a +69%): la sua risposta è un costo fisso che si ripaga solo quando evita molti giri di ricerca.

## `mode="locations"` sui 25 bug descritti a parole (16 + 9 di sympy)

Ipotesi: il codice restituito da zt resta nella conversazione e viene ripagato a ogni turno; a un client che sa
leggere i file basta sapere dove guardare. Verificata con la stessa configurazione dei giri precedenti (encoder
leggero), così cambia solo il contenuto della risposta: 500 token di posizioni invece di 4000 di codice. Qwen3 non
avrebbe aiutato: Claude ha scritto tutte le 25 query a zt in inglese, con nomi di funzioni (scenario "domanda
tradotta", dove Qwen3 e l'encoder leggero si equivalgono: 57% contro 56%).

| totale sui 25 bug | senza zt | zt 4k | zt-loc |
|---|---|---|---|
| bug corretti | 21/25 | 22/25 | **24/25** |
| costo totale | 1,80 $ | 1,91 $ (+6%) | **1,75 $ (-3%)** |
| mediana dei rapporti di costo con "senza zt" | | 1,16 | **0,98** |
| più economico di "senza zt" | | 8/25 | 13/25 |
| turni (mediana) | 7 | 5 | 7 |
| chiamate medie Grep / Read / Bash | 1,40 / 1,44 / 4,16 | 0,52 / 0,28 / 3,12 | 0,96 / 1,88 / 3,28 |
| contesto più grande (mediana) | 12,7k | 16,7k | 13,3k |

- **Il sovraccosto sparisce**: da +16/+24% (mediana) con 4000 token di codice a -4/+2% con le sole posizioni.
- **I turni non scendono**: Claude sostituisce un Grep con una chiamata a zt e poi apre il file con Read (1,9 letture
  a bug contro 1,4). Il guadagno è che la risposta piccola non pesa sui turni successivi.
- 3 bug corretti in più rispetto a "senza zt" (3 a 0): non significativo (p = 0.25) e non dovuto alla ricerca — in
  tutti i casi anche senza zt Claude aveva modificato il file giusto.
- Per questo **`zt mcp` usa ora `mode="locations"` di default** (`zt mcp --mode code` per tornare al codice).
- Cautela: la variante `zt-loc` è stata eseguita qualche ora dopo le altre (stesso modello); un solo giro per variante.

## Cosa vuol dire, e cosa non dice

- Con un modello forte che sa usare Grep, su progetti piccoli, medi e anche grandi con nomi pubblici chiari, zt **non
  migliora la qualità in modo dimostrato**.
- Restituendo codice (4000–12000 token) costa dal +7% al +37% in più; restituendo solo posizioni costa **come
  Claude Code da solo**. Non fa risparmiare: con questi progetti Grep trova già il file al primo colpo.
- Non misurato: codice con nomi in italiano, modelli più piccoli (Haiku), sessioni lunghe con più richieste.

Dati grezzi: `results/ab_claude_code_{20bugs,vague16,sympy9}.{md,jsonl}`.

## MCP: locations, focus, locations + zt_read_span (ottobre 2026, misura interrotta)

Quattro bracci sugli stessi bug: Claude Code da solo, `--mode locations`, `--mode focus`,
`--mode locations --read-span`. La misura è stata fermata a 10 bug completi su 25 (8 attrs/click/flask con
descrizioni vaghe, 2 sympy), 41 sessioni, 2,06 $. Rapporti = mediana per bug (braccio zt / da solo).

| | da solo | locations | focus | locations + read_span |
|---|---|---|---|---|
| bug risolti (8 vaghi) | 7/8 | 7/8 | 6/8 | 5/8 |
| bug risolti (2 sympy) | 2/2 | 2/2 | 2/2 | 2/2 |
| costo per bug, vaghi | 1 | 1,12 | 1,15 | 1,20 |
| costo per bug, sympy | 1 | 1,47 | 1,53 | 1,38 |
| contesto più grande, vaghi | 1 | 1,41 | 1,38 | 1,43 |

- Nessun braccio con zt costa meno di Claude Code da solo; nessuna differenza è significativa (sign test p ≥ 0,29).
- `focus` non migliora `locations`: più codice in partenza, stesso costo o poco di più, un bug risolto in meno.
- `zt_read_span` è stato usato poco (3 chiamate in 10 bug): in Claude Code il modello preferisce `Read`.
  Due bug persi in più rispetto a "da solo": campione troppo piccolo per attribuirli al tool.
- Decisione: il default resta `--mode locations`, `--read-span` resta spento. `--read-span` serve dove il
  client non ha un suo strumento di lettura (Claude Desktop): lì non è misurato.

Dati grezzi: `results/ab_claude_code_mcp3_{vague,sympy}.{md,jsonl}`.
