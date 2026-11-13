# Traceback nella domanda: i punti indicati dall'errore vanno per primi

Quando la domanda contiene l'output di un errore (traceback Python, `file:riga` di pytest, mypy, ruff, tsc, eslint,
go, rustc, node, Java), `zt` riconosce i punti `file:riga`, li collega ai file indicizzati (anche se il percorso è
assoluto, Windows o di una copia installata in `site-packages`) e mette **per primi** i simboli che contengono quelle
righe: prima il codice sorgente, poi i test, dal punto più interno. Una funzione lunga viene tagliata a una finestra
di 80 righe attorno alla riga indicata. Si spegne con `trace_anchors = 0` in `[router]`.

Nell'intestazione della risposta MCP compare `pinned from the error output in the query: file:riga, ...`.

## Misura (18/10/2026)

`examples/benchmark_traceback.py`: 42 bug reali (attrs 20, click 20, flask 2) presi da
`benchmark_bugfix.py mine`, cioè commit in cui i test falliscono prima della correzione e passano dopo. La domanda è
**l'output di pytest dei test che falliscono** (le ultime 150 righe, al massimo 8000 caratteri): quello che uno
sviluppatore incolla in chat. Metrica: almeno una riga cambiata dalla correzione è nel contesto, per ogni file
corretto. Encoder di test (`HashingEncoder`), budget 12000.

| metodo | punto da correggere nel contesto | token medi |
|---|---|---|
| zt | **25/42 (60%)** | 8477 |
| zt senza traceback | 22/42 (52%) | 8355 |
| BM25 sui simboli | 25/42 (60%) | 8372 |
| BM25 sui simboli senza traceback | 21/42 (50%) | 8118 |
| BM25 su file interi | 11/42 (26%) | 11988 |

- Con il traceback zt vince 3 casi e non ne perde nessuno (BM25 sui simboli: 4 a 0). Sui 42 casi non è significativo
  (p = 0.25), perché **quasi tutti i log sono asserzioni fallite**: indicano solo la riga del test, non il codice
  sbagliato.
- Il guadagno è tutto nei log che indicano righe del codice sorgente: **12 casi**, dove il punto giusto entra nel
  contesto **10 volte su 12 contro 7**; negli **8** in cui il log indica proprio il file da correggere, **7 su 8 contro 4**.
- Costo: circa 120 token in più a domanda (il simbolo del test fissato in testa).
- zt contro i file interi: 14 casi a 0 (p = 0.0001), con il 30% di token in meno.

Tabella grezza: `results/traceback_3repos.md`.
