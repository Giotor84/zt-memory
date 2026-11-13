# Benchmark end-to-end di correzione bug

Domanda: **con lo stesso modello locale, il contesto di zt-memory fa risolvere più bug (o gli stessi con meno token)
di un contesto "ingenuo" a file interi?** Si misura il risultato finale (test verdi), non la recall: un contesto con
più file giusti può risolvere meno bug ("Recall Trap", arXiv 2608.14838).

## Metodo

1. **Casi reali dalla storia git** (`mine`): commit di correzione che toccano codice Python (≤ 2 file, ≤ 40 righe)
   *e* test. Un caso è valido solo se, nel tuo ambiente, i test del commit **falliscono** sul codice prima della
   correzione e **passano** dopo. I duplicati (stessa correzione su due rami) vengono scartati.
2. **Il modello vede**: il codice prima della correzione e il messaggio del commit come segnalazione del bug.
   **Non vede** i test nuovi: vengono applicati solo per dare il voto, sopra eventuali modifiche del modello ai test.
3. **Tre condizioni**, stesso prompt di sistema, stesso protocollo SEARCH/REPLACE, un nuovo tentativo con il testo
   reale del file se la patch non combacia, temperatura 0:

   | condizione | contesto |
   |---|---|
   | `bm25-files` | BM25 su file interi, impacchettati in ordine finché il budget è pieno (l'ultimo tagliato) |
   | `zt` | router di zt-memory: lessicale + embedding + energia, span per simbolo, taglio dinamico |
   | `bm25-matched` | controllo: come `bm25-files` ma fermo ai token usati da `zt` nello stesso caso |

   Il controllo serve a separare "meno token" da "selezione migliore": se `zt` batte `bm25-files` ma non
   `bm25-matched`, il guadagno viene solo dal contesto più corto.
4. **Metriche** per caso e condizione: risolto sì/no, token del prompt (contati dal tokenizer del server), secondi di
   LLM, tentativi, file corretti presenti nel contesto, patch inutilizzabile. Confronto appaiato con McNemar esatto.

## Comandi (Windows)

```powershell
git clone https://github.com/pallets/click C:\src\click
cd C:\path\to\zt-memory

# 1. estrai i casi (qualsiasi stato del server)
python examples\benchmark_bugfix.py mine --repo C:\src\click --workdir C:\zt_bench\click --limit 15

# 2. facoltativo, llama-server SPENTO: primo indice su GPU (serve torch con CUDA)
python examples\benchmark_bugfix.py index --workdir C:\zt_bench\click --device cuda

# 3. llama-server ACCESO: tutti i casi, tutte le condizioni (riprende da dove si era fermato)
python examples\benchmark_bugfix.py run --workdir C:\zt_bench\click

# 4. tabelle
python examples\benchmark_bugfix.py report --workdir C:\zt_bench\click
```

`cases.json` è modificabile a mano: togli i casi che non sono bug veri (es. "skip flaky test") o riscrivi una
segnalazione troppo vaga. Le dipendenze di test della repo devono essere installate nello stesso Python.

## Cosa è verificato

- Test automatici su una repo git sintetica: estrazione (scarta il commit i cui test passano già), le tre condizioni,
  il voto, il report e la ripresa senza rifare richieste.
- Su `click` vero (ultimi 800 commit, qui su Linux): 42 candidati → 15 casi validi in 5.5 min. Le tre condizioni
  sono state eseguite su 2 casi con un modello minuscolo a pesi casuali, solo per verificare il funzionamento
  (nessun bug risolto, come previsto). Token usati: `bm25-files` circa 15 600, `zt` 10 300–11 600.
  **Con questo tipo di domande il risparmio di `zt` è stato del 25–35%, non del 40%.** Il taglio dinamico
  interviene poco quando molti span hanno punteggi simili.

## Cosa non è ancora misurato

Il risultato con Qwen2.5-Coder-14B: va eseguito sul PC con la GPU. Stima dei tempi, non misurata: circa 1–2 minuti
per caso e condizione, quindi circa 1–1.5 ore per 15 casi × 3 condizioni, più il primo indice (circa 20 minuti su CPU).

## Limiti

- Con 15 casi la differenza diventa statisticamente significativa solo se è grande: per esempio, 5 casi risolti
  solo da una condizione e 0 dall'altra danno p = 0.06. Il report mostra anche i casi in cui una sola condizione
  ha risolto, e la tabella caso per caso.
- Il messaggio di commit è una segnalazione povera rispetto a una vera issue: alcuni casi possono essere
  irrisolvibili per tutte le condizioni. `--with-failure-log` aggiunge l'output del test fallito, come in una
  segnalazione con traceback, ma rivela il nome del test.
- La memoria segue la storia (i casi sono eseguiti dal più vecchio): l'energia riflette le modifiche passate,
  come nell'uso reale. Per questo l'ordine dei casi conta.
