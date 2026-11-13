# Relazione su zt-memory: cosa è stato fatto e cosa dicono i test (18/10/2026, v0.4.0)

## 1. Cosa è zt-memory

Un motore di contesto locale per i modelli che scrivono codice. Indicizza un progetto funzione per funzione e, a ogni
domanda, restituisce solo i pezzi di codice utili, con file e righe esatte, entro un limite di token. Cerca per parole
esatte (BM25) e per significato (embedding) insieme. Gira tutto sul PC, senza servizi esterni.

## 2. Cosa è stato costruito

| componente | stato |
|---|---|
| Indice per simbolo (tree-sitter, 8 linguaggi), sync incrementale, lock tra processi | fatto |
| Ciclo locale `zt code`: contesto → patch → test → annulla (llama.cpp + Qwen2.5-Coder-14B) | fatto, verificato su Windows |
| Server MCP `zt mcp` con 3 tool; parte anche nei progetti senza zt; encoder caricato solo quando serve | fatto, collegato a Claude Code |
| `exclude_globs` per tenere fuori dall'indice file vietati (progetto privato) | fatto |
| Riconoscimento dei traceback nella domanda (`file:riga` → simboli fissati per primi) | fatto |
| Risposte MCP in modalità `code` al massimo di 4000 token (prima 12000) | fatto, deciso dai dati del test E |
| `zt_query_context(mode="locations")`: solo file, righe, firma e docstring, ≤ 500 token | fatto, misurato (test H), **default di `zt mcp`** |
| Pulizia: espansione geodetica rimossa (non serviva), licenza Apache-2.0, README | fatto |
| Backend Amazon Bedrock per `zt code`, `zt engine` e il benchmark (v0.4.0: MiniMax M2.5, Qwen3-235B, limite di chiamate, registro dei costi) | fatto e testato con un client finto; **verifica dal vivo in un ambiente SageMaker air-gapped da fare** ([BEDROCK.md](BEDROCK.md)) |
| Knowledge base con concetti, decisioni, glossario, domande frequenti e problemi | [KNOWLEDGE_BASE.md](KNOWLEDGE_BASE.md) |
| Test automatici | 176 superati (senza rete e senza boto3) |

## 3. Test e risultati

### A. Ciclo locale sul PC (RTX 4060 Ti 16 GB)

Bug di prova corretto al primo tentativo, test verdi, ~14 s in tutto; ~10 GB di VRAM, 24–27 token/s con contesto 16k.

### B. Trovare il punto da correggere: 150 bug reali in 6 progetti (Python, Go, TypeScript)

| domanda | zt | BM25 sulle funzioni | BM25 su file interi |
|---|---|---|---|
| messaggio del commit (inglese) | **61%** | 58% | 19% (65 casi a 2) |
| italiano senza nomi di codice, encoder Qwen3 | **41%** | 27% a parità di token (21 casi a 0) | — |
| domanda tradotta in inglese | 57% (Qwen3), 56% (encoder leggero) | 51% | — |

- Il guadagno vero è tagliare il codice per funzione: contro i file interi zt vince nettamente con ~30% di token in meno.
- Qwen3 conta solo quando la domanda è in un'altra lingua; tradotta in inglese basta l'encoder leggero.

### C. Bug corretti dal 14B locale (14 bug di click)

| | zt | BM25 su file interi |
|---|---|---|
| bug risolti | 1/14 | 0/14 |
| file giusto nel contesto | 93–100% | 21–29% |
| patch inutilizzabili (giro senza / con log dei test) | 2 / 3 | 6 / 10 |
| token di contesto | 8404 | 13840 |

Il contesto giusto arriva al modello, ma il 14B quasi mai scrive la correzione corretta: qui il limite è il modello.

### D. Traceback nella domanda (42 bug, domanda = output di pytest)

| | trova il punto giusto |
|---|---|
| zt con traceback | **60%** |
| zt senza traceback | 52% (3 casi a 0) |
| BM25 su file interi | 26% |

Quando il log indica il codice sorgente (12 casi): 10 contro 7.

### E. Claude Code con e senza zt — richiesta = log di pytest (20 bug, 60 sessioni)

| | senza zt | zt 12000 token | zt 4000 token |
|---|---|---|---|
| bug corretti | 20/20 | 20/20 | 19/20 |
| costo per bug (mediana) | 0,058 $ | 0,092 $ (+37%) | 0,058 $ (+7%) |
| turni (mediana) | 8 | 6,5 | 7 |

### F. Claude Code — bug descritti a parole in italiano, test nascosti (16 bug, 48 sessioni)

| | senza zt | zt 4000 token | zt-loc (solo posizioni) |
|---|---|---|---|
| bug corretti | 13/16 | 15/16 | 15/16 |
| costo per bug (mediana) | 0,041 $ | 0,058 $ | 0,044 $ |
| mediana dei rapporti di costo con "senza zt" | | +24% | **+2%** |
| turni (mediana) | 7 | 5 | 6,5 |

Nei 2 casi vinti, anche senza zt Claude aveva trovato il file giusto: la differenza è nel dettaglio della correzione.

### G. Claude Code — progetto grande: sympy, 1609 file (9 bug descritti a parole, 27 sessioni)

| | senza zt | zt 4000 token | zt-loc (solo posizioni) |
|---|---|---|---|
| bug corretti | 8/9 | 7/9 | 9/9 |
| costo per bug (media / mediana) | 0,114 / 0,085 $ | 0,110 / 0,070 $ | 0,085 / 0,070 $ |
| mediana dei rapporti di costo con "senza zt" | | +16% | **-4%** |
| turni (media / mediana) | 10 / 6 | 8,4 / 5 | 10,1 / 8 |

- In tutte le 27 sessioni l'agente ha modificato il file giusto, con e senza zt: anche in 1600 file il nome pubblico
  di una funzione (sinc, acot, refine, fcode) basta a Grep.
- zt costa meno nei due bug lunghi (21–27 turni: -20% e -11%) e quasi sempre di più in quelli brevi (+17%..+69%).
  La sua risposta è un costo fisso che si ripaga solo quando evita molti giri di ricerca.

## 4. Perché zt costava di più a Claude Code, e la correzione (test H)

In E, F e G zt riduce i turni (-20/-25%) e i file aperti, ma il costo non scende: i ~4000 token di codice restituiti
da `zt_query_context` restano nella conversazione e vengono ripagati a ogni turno successivo. A un client che sa
leggere i file da solo serve sapere **dove** guardare.

`zt_query_context` ha quindi `mode="locations"`: stessa selezione, ma restituisce solo file, righe, firma e prima
frase della docstring di ogni risultato, dal migliore in giù, al massimo 500 token (circa 9 volte meno del codice).

### H. `zt-loc` sui 25 bug descritti a parole (F + G), stessa configurazione dei giri precedenti

Encoder leggero come negli altri giri, così cambia solo la risposta (500 token di posizioni contro 4000 di codice).
Qwen3 non avrebbe cambiato il quadro: Claude ha scritto tutte le 25 query a zt in inglese e con nomi di funzioni, il
caso in cui Qwen3 e l'encoder leggero si equivalgono (57% contro 56% nel test B). In più huggingface.co non è
raggiungibile da questo ambiente cloud.

| totale sui 25 bug | senza zt | zt 4000 token | zt-loc |
|---|---|---|---|
| bug corretti | 21/25 | 22/25 | **24/25** |
| costo totale | 1,80 $ | 1,91 $ (+6%) | **1,75 $ (-3%)** |
| mediana dei rapporti di costo con "senza zt" | | 1,16 | **0,98** |
| più economico di "senza zt" | | 8/25 | 13/25 |
| turni (mediana) | 7 | 5 | 7 |
| chiamate medie Grep / Read | 1,40 / 1,44 | 0,52 / 0,28 | 0,96 / 1,88 |

- **Il sovraccosto sparisce**: da +16/+24% a -4/+2% (mediana). Per questo `locations` è ora il default di `zt mcp`.
- **I turni non scendono**: come previsto, Claude sostituisce un Grep con zt e poi apre il file con Read. Il guadagno è
  che la risposta piccola non pesa sui turni successivi.
- I 3 bug corretti in più (3 a 0) non sono significativi (p = 0.25) e non vengono dalla ricerca: anche senza zt Claude
  aveva modificato il file giusto.

## 5. Conclusioni

**Dove zt serve (misurato):**
1. **Modelli locali su GPU consumer**: il 14B vede solo quello che gli passi; con zt il file giusto è nel contesto nel
   93–100% dei casi contro 21–29%, con meno token.
2. **Domande in italiano, con Qwen3**: 41% contro 27% a parità di token.
3. **Errori incollati che indicano il codice sorgente**: 10 su 12 contro 7.

**Dove non fa risparmiare (misurato):** Claude Code su progetti piccoli, medi e grandi con nomi pubblici chiari, sia con
log sia con descrizioni vaghe: Grep trova già il file giusto. Restituendo codice zt costava dal 7% al 37% in più; con
`mode="locations"` (ora il default) costa quanto Claude da solo, con qualità pari o leggermente migliore (24/25
contro 21/25, non significativo).

**Posizionamento:** zt è un motore di contesto per **LLM locali su GPU consumer**; con Claude Code è un indice
gratuito (costo pari, `mode="locations"`) che conviene tenere acceso dove Grep fatica: codice con nomi in italiano,
progetti con nomi poco espressivi, sessioni lunghe — tutti casi ancora da misurare. Il risultato più originale per un paper resta la ricerca nel
codice con domande non in inglese, con controllo a parità di token.

## 6. Costi dei test con Claude Code

| test | sessioni | costo |
|---|---|---|
| prova iniziale | 2 | 0,16 $ |
| E — log di pytest | 60 | 4,40 $ |
| F — descrizioni in italiano | 32 | 1,69 $ |
| G — sympy | 18 | 2,01 $ |
| H — zt-loc (16 + 9 bug) | 25 | 1,75 $ |
| **totale** | **137** | **10,01 $** |

## 7. Limiti dei test

- Un solo giro per variante: le sessioni di un agente variano, differenze di 1–2 bug non sono affidabili.
- Le descrizioni in italiano sono scritte da Claude guardando la correzione, con regole fisse (solo il sintomo, nessun
  nome interno, nessun testo d'errore), non da utenti reali.
- Nei test con Claude Code l'encoder è quello leggero, non Qwen3 (huggingface.co non è raggiungibile da qui).
- La variante `zt-loc` è stata eseguita qualche ora dopo le altre, con lo stesso modello.
- Non misurati: codice con nomi in italiano (progetto privato), modelli cloud più piccoli (Haiku), sessioni lunghe con più
  richieste, memoria delle decisioni tra sessioni.

## 8. Dove sono i dati

| test | documento | dati grezzi |
|---|---|---|
| B | `docs/BENCHMARK_LOCALIZE.md` | `docs/results/localize_*.md` |
| C | `docs/BENCHMARK_BUGFIX.md` | — |
| D | `docs/BENCHMARK_TRACEBACK.md` | `docs/results/traceback_3repos.{md,jsonl}` |
| E, F, G, H | `docs/BENCHMARK_CLAUDE_CODE.md` | `docs/results/ab_claude_code_{20bugs,vague16,sympy9}.{md,jsonl}` |
