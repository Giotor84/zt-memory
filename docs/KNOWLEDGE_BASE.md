# zt-memory — knowledge base (v0.4.0)

Cosa è zt-memory, come funziona dentro, perché è fatto così, come si usa e cosa fare quando qualcosa non va.
Ogni decisione riporta il dato che la giustifica; i risultati completi sono in [RELAZIONE.md](RELAZIONE.md).

Indice: [1. In breve](#1-in-breve) · [2. Concetti](#2-concetti-spiegati-semplice) ·
[3. Architettura](#3-architettura) · [4. Modi d'uso](#4-quale-modo-duso-scegliere) ·
[5. Configurazione](#5-configurazione) · [6. Decisioni](#6-decisioni-prese-e-perché) ·
[7. Glossario](#7-glossario) · [8. Domande frequenti](#8-domande-frequenti) ·
[9. Problemi](#9-risoluzione-dei-problemi) · [10. Limiti](#10-limiti-noti) · [11. Mappa](#11-dove-trovare-cosa)

---

## 1. In breve

zt-memory è un **motore di contesto locale** per i modelli che scrivono codice. Indicizza un progetto funzione per
funzione e, a ogni domanda, restituisce solo i pezzi di codice utili (con file e righe esatte) entro un limite di
token. Non genera mai testo da solo: sceglie cosa far leggere al modello.

Il problema che risolve: un modello vede solo quello che gli passi. Passargli file interi costa token e lo distrae;
passargli i pezzi sbagliati lo fa sbagliare. Su 150 bug reali, zt mette il punto da correggere nel contesto nel 61%
dei casi, contro il 19% di un sistema che passa i file interi più simili alla domanda, con circa il 30% di token in meno.

---

## 2. Concetti, spiegati semplice

**Simbolo (chunk).** L'unità dell'indice: una funzione, una classe, un metodo. tree-sitter divide i file per
simbolo in 8 linguaggi (Python, JavaScript, TypeScript/TSX, Go, Rust, Java, C, C++); per gli altri file zt usa
blocchi ancorati al contenuto. Ogni simbolo ha un'identità stabile, `file::Classe.metodo`: se il codice si sposta di
qualche riga, l'identità resta. Le funzioni con lo stesso nome (overload) diventano `nome~hash(firma)`.

**Ricerca ibrida.** Due ricerche in parallelo, poi unite:
- *per parole* (BM25 su SQLite FTS5, con token adatti al codice: `apply_discount` diventa anche `apply` e `discount`):
  trova nomi esatti, codici d'errore, stringhe;
- *per significato* (embedding): trova il codice giusto anche quando la domanda usa parole diverse dal codice.

**Fusione RRF.** Le due classifiche si uniscono con il *Reciprocal Rank Fusion*: ogni candidato prende
`peso / (k + 1 + posizione)` da ciascuna lista in cui compare (k = 10). Se la domanda contiene nomi di codice, la
lista per parole pesa il doppio.

**Energia.** Ogni simbolo modificato o recuperato di recente si "scalda" e poi si raffredda nel tempo, come un
oggetto con inerzia e attrito: `E_t = γ·E_(t-1) + ‖v_t‖`. Nel punteggio finale l'energia dà un piccolo bonus
(`1 + 1,5 · energia normalizzata`). È una scorciatoia per "ciò su cui stai lavorando adesso".

**Taglio dinamico.** Dopo i primi 3 risultati, zt smette di aggiungere codice quando il punteggio scende sotto il 25%
del migliore, invece di riempire il budget con codice poco pertinente.

**Budget di token.** Il tetto alla dimensione del contesto restituito. Con il modello locale vale la finestra del
modello meno lo spazio per la risposta; con Claude Code via MCP è al massimo 4000 token di codice, o 500 di sole
posizioni (vedi sotto).

**Modalità `locations`.** Per i client che sanno aprire i file da soli (Claude Code, Cursor): invece del codice zt
restituisce solo file, righe, firma e prima frase della docstring di ogni risultato. Il client poi legge solo le righe
che servono. È il default di `zt mcp`.

**Traceback fissati.** Se la domanda contiene un errore (`File "x.py", line 12`, `x.py:12:`, `x.ts:12:5`,
`x.go:12`), i simboli a quelle righe entrano per primi nel contesto: prima il codice sorgente, poi i test, dal punto
più interno.

**Encoder.** Trasforma il testo in vettori per la ricerca per significato.
- `hashing` (default): leggero, istantaneo, nessun modello da scaricare;
- `hf:Qwen/Qwen3-Embedding-0.6B`: un vero modello (1–2 GB di RAM); serve davvero solo quando la domanda è in una
  lingua diversa da quella del codice.

**Proiettore e rango.** I vettori dell'encoder vengono compressi con una SVD calcolata una volta sul progetto
(`rank`, default 128): occupano meno disco e la ricerca è più veloce.

**Store.** Dove vive l'indice, in `.zt/store`: vettori in file mappati in memoria (SSD) più un database SQLite (in
modalità WAL) con metadati e indice per parole. Una cache in RAM (1,5 GB di default) tiene i vettori usati più spesso.

**Sync incrementale.** `zt sync` controlla data e dimensione di ogni file. Se un file è cambiato, rilegge solo i
simboli diversi; se è cambiata solo la data ma non i byte, non fa niente. Il primo sync indicizza tutto.

**Decisioni (SPEC/DECISION).** Note di progetto salvate con `zt_record_decision` e collegate a file o simboli:
tornano fuori insieme al codice a cui si riferiscono.

**Patch SEARCH/REPLACE.** Il formato con cui il modello propone le modifiche in `zt code`: per ogni blocco, il testo da
cercare e quello nuovo. zt applica tutti i blocchi o nessuno. Prova prima la corrispondenza esatta, poi ignorando gli
spazi a fine riga, poi ignorando l'indentazione. Fa un backup prima di scrivere (`zt undo` torna indietro).

**Geometria iperbolica.** Le posizioni nell'indice stanno in una "palla di Poincaré": il raggio separa i livelli
(cartella, file, simbolo). Dentro lo stesso livello la classifica equivale alla normale similarità coseno. È rimasta
perché costa poco e cambiarla vorrebbe dire ricostruire tutti gli indici, ma non porta un guadagno misurato.

---

## 3. Architettura

```
                         ┌───────────────────────── zt sync ─────────────────────────┐
 file del progetto ──► discovery (gitignore, exclude_globs) ──► chunking (tree-sitter)
                         ──► encoder ──► proiettore SVD ──► store (memmap + SQLite/FTS5)

 domanda ──┬─► encoder ► SVD ► ricerca per significato ──────┐
           ├─► token di codice ► BM25 (FTS5) ────────────────┤
           └─► file:riga dell'errore ► simboli fissati ──────┤
                     espansione: decisioni collegate, intestazioni dei file
                     fusione RRF · peso del tipo · energia ► taglio dinamico ► budget
                     ► testo esatto dal disco (verificato con hash) ► contesto
                               │
            ┌──────────────────┼────────────────────┬─────────────────────┐
        zt mcp (locations     zt code (14B locale  zt code (Bedrock:      zt route
        o code) per Claude    via llama.cpp)       MiniMax, Qwen3-235B)   (prompt da copiare)
```

| modulo (`src/zt_memory/`) | cosa fa |
|---|---|
| `discovery.py` | quali file entrano nell'indice: rispetta `.gitignore`, `exclude_globs`, limiti di dimensione |
| `syntax.py`, `chunking.py` | divide i file in simboli (tree-sitter, con ripiego su `ast` e su blocchi) |
| `encoders.py`, `projector.py` | encoder (hashing o Qwen3) e compressione SVD congelata |
| `lexical.py` | ricerca per parole (FTS5/BM25) con token adatti al codice |
| `storage.py`, `index.py`, `working_set.py` | file mappati su SSD, SQLite WAL, ricerca a blocchi, cache in RAM |
| `hyperbolic.py`, `dynamics.py` | palla di Poincaré; energia con inerzia e attrito |
| `manager.py` | sync incrementale, decisioni, rinforzo, lock tra processi |
| `router.py`, `stacktrace.py`, `outline.py` | scelta del contesto; punti `file:riga` degli errori; firme per la modalità locations |
| `mcp_server.py` | server MCP su stdio, senza dipendenze esterne |
| `engine_client.py`, `engine_bedrock.py` | client per server compatibili OpenAI e per Amazon Bedrock |
| `editing.py`, `coding.py` | patch SEARCH/REPLACE, backup, ciclo `zt code` |
| `bench_*.py`, `replay.py`, `evaluation.py` | benchmark e misure |
| `project.py`, `cli.py`, `watch.py` | `.zt/config.toml`, comandi, osservazione dei file durante l'editing |

**Concorrenza.** Lo store ha un lock tra processi. Il server MCP lo apre solo durante una chiamata, quindi hook git,
`zt watch` e `zt code` possono lavorare in parallelo; se lo store è occupato, risponde "store busy, retry".

---

## 4. Quale modo d'uso scegliere

| situazione | modo | perché |
|---|---|---|
| Claude Code, Cursor, Cline | `zt mcp` (default `locations`) | costo pari a Claude da solo, qualità pari o leggermente migliore (24/25 bug contro 21/25, non significativo) |
| modello locale su GPU consumer | `zt code` con llama.cpp | è dove zt serve di più: il file giusto è nel contesto nel 93–100% dei casi contro 21–29% |
| ambiente senza GPU né internet, con Bedrock | `zt code` con `backend = "bedrock"` | stessa pipeline, modello in cloud; qualità non ancora misurata |
| modelli >= 120B via API, più giri ammessi | `[router] preset = "api"` + `NEED` | 5 corpi + mappa di firme (-58% di token al giro 1); il modello chiede ciò che manca ([NEED.md](NEED.md)); da misurare in fase 2 |
| stessi modelli, una chiamata sola | `[router] preset = "single"` | corpi fino all'85% del punteggio: -36/-42% di token, 3 casi persi su 150 (fase 1b) |
| un altro strumento o una chat qualunque | `zt route "..."` | stampa il prompt con il contesto, da incollare |
| domande in italiano su codice in inglese | encoder Qwen3 | 41% contro 27% a parità di token; se la domanda viene tradotta, l'encoder leggero basta |

---

## 5. Configurazione

Tutto in `.zt/config.toml`, scritto da `zt init`; solo valori semplici (numeri, testi, vero/falso).

| sezione | chiavi principali | note |
|---|---|---|
| `[store]` | `path` | default `.zt/store` |
| `[encoder]` | `spec` (`hashing` o `hf:Qwen/Qwen3-Embedding-0.6B`), `device` (`cpu`/`cuda`) | cambiare encoder richiede di ricostruire lo store |
| `[memory]` | `rank` (128), `exclude_globs`, `chunk_max_lines` (120), `respect_gitignore`, `lock_timeout_s` (30), `ram_budget_mb` (1536) | `exclude_globs` va impostato **prima** del primo sync |
| `[router]` | `token_budget` (12000), `cutoff_alpha` (0,25), `energy_weight` (1,5), `trace_anchors` (4), `test_path_weight` (1,0) | `trace_anchors = 0` spegne i traceback fissati |
| `[engine]` | `backend` (`openai`/`bedrock`), `url`, `model`, `ctx`, `max_new_tokens`, `temperature`, `retries`, `timeout_s`; per Bedrock `region`, `proxy`, `max_calls`, `aws_retries` | vedi [BEDROCK.md](BEDROCK.md) |

`exclude_globs = "segreti.py, tests/fixtures/*.json"`: ogni modello viene confrontato con la fine del percorso, senza
distinguere maiuscole e minuscole; `*` non attraversa le cartelle. È una lista di esclusioni: un file nuovo con dati
sensibili entra finché non lo aggiungi. Per vedere cosa entrerebbe: `zt status --json`, campo `pending.added`.

---

## 6. Decisioni prese e perché

| decisione | dato che la giustifica | dove |
|---|---|---|
| Indicizzare per simbolo, non per file | punto giusto nel contesto 61% contro 19% dei file interi (65 casi a 2), con il 30% di token in meno | BENCHMARK_LOCALIZE |
| Ricerca ibrida, non solo per significato né solo per parole | requests con Qwen3: recall@10 0,845 contro 0,823 della sola ricerca per parole e 0,795 di quella per significato | README, ablazioni |
| Energia con peso 1,5 | +0,002…+0,027 di recall@10 in 4 progetti su 4; mai negativa | README, ablazioni |
| Taglio dinamico a 0,25 | -40% di token per -3,5 punti di copertura dei file giusti (requests) | README, ablazioni |
| Espansione geodetica rimossa | nessun effetto in 4 progetti, peggiore in click | README |
| Encoder leggero di default, Qwen3 su richiesta | con domande tradotte 56% contro 57%: Qwen3 conta solo con domande in un'altra lingua (41% contro 27%) | BENCHMARK_LOCALIZE |
| Traceback fissati in testa | 60% contro 52% (3 casi a 0); dove il log indica il sorgente 10/12 contro 7/12 | BENCHMARK_TRACEBACK |
| MCP: al massimo 4000 token di codice (prima 12000) | con 12000 Claude Code spendeva +37% per bug; con 4000 +7% | BENCHMARK_CLAUDE_CODE |
| MCP: default `locations` | su 25 bug descritti a parole: 4000 token di codice +16% (mediana), sole posizioni -3% sul totale | BENCHMARK_CLAUDE_CODE |
| Patch SEARCH/REPLACE tolleranti ai blocchi ``` | il 14B li mette dentro le sezioni SEARCH; senza tolleranza la patch fallisce | editing.py |
| Bedrock: `retries` separato da `aws_retries` | `retries` di zt code = una chiamata a pagamento in più ogni volta; i tentativi di rete non devono moltiplicarle | BEDROCK.md |
| Bedrock: nessuna chiamata automatica, tetto `max_calls` | ogni chiamata costa; `zt engine --probe` solo su richiesta | BEDROCK.md |
| Il progetto si cerca risalendo le cartelle, ma mai fino alla radice del disco | un `.zt` in `C:\` o `/` veniva preso da ogni sottocartella: `zt sync` indicizzava il disco intero e 4 test fallivano (revisione del 01/11/2026). `ZT_DISCOVERY_CEILING` ferma la ricerca a una cartella (i test la usano su `tmp_path`) | project.py |
| Licenza Apache-2.0 | permissiva, con protezione sui brevetti | LICENSE |

---

## 7. Glossario

| termine | significato |
|---|---|
| BM25 | classifica per parole: premia i documenti che contengono le parole rare della domanda |
| chunk / simbolo | unità dell'indice: funzione, classe, metodo, o blocco di un file senza grammatica |
| contesto | il testo (codice e note) che il modello legge prima di rispondere |
| embedding | vettore numerico che rappresenta il significato di un testo |
| encoder | il modello che produce gli embedding |
| FTS5 | l'indice di ricerca testuale integrato in SQLite |
| MCP | Model Context Protocol: lo standard con cui Claude Code, Cursor e altri usano strumenti esterni |
| McNemar | test statistico per confronti appaiati (stessi bug, due metodi): dice se la differenza può essere caso |
| RRF | Reciprocal Rank Fusion: unisce più classifiche sommando l'inverso delle posizioni |
| SVD | scomposizione che riduce i vettori a meno dimensioni conservando le più informative |
| token | unità di testo dei modelli (circa 3–4 caratteri di codice) |
| WAL | modalità di SQLite che permette letture mentre un altro processo scrive |
| reasoningContent | blocchi di "ragionamento" dei modelli come MiniMax M2.5, prima della risposta |
| Converse | l'API unificata di Amazon Bedrock per i modelli di chat |

---

## 8. Domande frequenti

**zt fa risparmiare soldi con Claude Code?** No, nelle misure fatte: Claude trova già il file giusto con Grep. Con la
modalità `locations` costa quanto senza; con risposte di codice costava dal 7% al 37% in più. Dove zt fa la differenza
è con i modelli locali.

**Serve Qwen3?** Solo se fai domande in una lingua diversa da quella del codice e nessuno le traduce (ad esempio
`zt code "..."` in italiano con il modello locale). Con Claude Code le query arrivano già in inglese.

**Devo riavviare Claude Code dopo `zt init`?** No: il server MCP cerca il progetto a ogni chiamata. Va riavviato solo
quando aggiorni zt.

**Cosa succede ai file che modifico mentre lavoro?** Ogni chiamata MCP fa un sync incrementale (fino a 100 file
cambiati); oltre, risponde con l'indice attuale e lo dice nell'intestazione. Gli span restituiti vengono comunque
riletti dal disco.

**I miei file vietati possono finire all'agente?** Non se sono in `exclude_globs` prima del primo sync. Un file escluso
dopo essere stato indicizzato viene tolto al sync successivo.

**Quanto spazio occupa l'indice?** Circa 40 MiB per 15.000 simboli con rango 64 (FTS5 compreso).

**Posso usare zt senza GPU?** Sì: indice e ricerca girano su CPU. Il primo sync con Qwen3 su CPU è lento (~2,3 simboli
al secondo); con l'encoder leggero è immediato.

**Con Bedrock quanto spendo?** Ogni chiamata è registrata in `.zt/engine_usage.jsonl` (token in ingresso e in uscita);
il costo si calcola con il listino del momento. `max_calls` (default 20) ferma il processo prima di sorprese.

---

## 9. Risoluzione dei problemi

| sintomo | causa | cosa fare |
|---|---|---|
| `zt is not enabled for this project` (MCP) | nessun `.zt/` nella cartella o sopra | `zt init`, eventuali `exclude_globs`, `zt sync` |
| `zt is not enabled` in una sottocartella, con `.zt` nella radice del disco | dal 01/11/2026 un `.zt` in `C:\` o `/` vale solo se la sessione parte proprio da lì | `zt init` nella cartella del progetto, oppure `--repo` |
| `index never built` | `zt init` fatto, sync no | `zt sync` (il primo può essere lungo) |
| `store busy, retry` / codice d'uscita 3 | un altro processo sta sincronizzando | aspetta o alza `lock_timeout_s` |
| `index NOT refreshed: N files changed` | troppi file cambiati per un sync dentro la chiamata | `zt sync` a mano o `zt mcp --max-inline-sync N` |
| primo sync lentissimo | Qwen3 su CPU | `zt sync` con `--device cuda` a server LLM spento, oppure encoder `hashing` |
| `encoder forced to CPU` | `zt code` con server LLM locale sulla GPU | normale: evita di esaurire la VRAM |
| `compute engine not reachable` (codice 1) | llama-server spento | avvialo (comando in ENGINE_WINDOWS.md) e controlla con `zt engine` |
| codice d'uscita 4 | nessuna patch utilizzabile dal modello | `--show-answer` per vedere la risposta; più `--retries`; domanda più precisa |
| codice d'uscita 5 | patch applicata ma `--check` fallito | con `--revert-on-fail` viene annullata; altrimenti `zt undo` |
| codice d'uscita 6 | errore del motore (rete, Bedrock, risposta vuota, `max_calls`) | leggi il messaggio; per Bedrock la tabella errori in BEDROCK.md |
| test falliti su Windows con fine riga | file scritti con CRLF | i test scrivono con `newline="\n"`; controlla `core.autocrlf` |
| `boto3 is not installed` | manca l'extra Bedrock | `pip install "zt-memory[bedrock]"` (offline: dalla wheelhouse) |
| `empty answer ... aumenta max_new_tokens` | il ragionamento di MiniMax ha usato tutto il budget | alza `max_new_tokens` (massimo 8192) |

Codici d'uscita di `zt`: 0 ok, 1 errore, 2 uso, 3 store occupato, 4 nessuna modifica utilizzabile, 5 test falliti,
6 errore del motore LLM.

---

## 10. Limiti noti

- Con modelli forti che usano Grep (Claude Code) zt non migliora la qualità in modo dimostrato e non fa risparmiare.
- Il 14B locale trova il contesto giusto ma corregge pochi bug (1 su 14 su click).
- Non misurati: codice con nomi in italiano, modelli cloud piccoli, sessioni lunghe, memoria delle decisioni tra
  sessioni, qualità delle patch con Bedrock (MiniMax M2.5, Qwen3-235B).
- Cambiare encoder o rango richiede di ricostruire lo store; le decisioni registrate non si ricostruiscono dal codice.
- `exclude_globs` è una lista di esclusioni, non di file ammessi.
- La consistenza tra SQLite e i file mappati è garantita da una compensazione in caso di errore, non da un'unica
  transazione; lo store è comunque ricostruibile dal repository.

---

## 11. Dove trovare cosa

| documento | contenuto |
|---|---|
| [../README.md](../README.md) | panoramica, installazione, comandi |
| [RELAZIONE.md](RELAZIONE.md) | tutti i test e i risultati, con i costi |
| [MCP.md](MCP.md) | server MCP: tool, modalità, collegamento a Claude Code, Cursor, Cline |
| [BEDROCK.md](BEDROCK.md) | backend Amazon Bedrock: configurazione, profili, limite di costo, errori |
| [NEED.md](NEED.md) | modelli grandi via API: contesto a fuoco, mappa, protocollo `NEED`, giro 2 append/lean, fase 2 |
| [BENCHMARK_FOCUS.md](BENCHMARK_FOCUS.md) | contesto a fuoco: anti-cloni, fuoco, posizione del blocco giusto, fuoco adattivo |
| [ENGINE_WINDOWS.md](ENGINE_WINDOWS.md) | llama.cpp e Qwen2.5-Coder-14B su Windows con GPU da 16 GB |
| [ENCODER_GUIDE.md](ENCODER_GUIDE.md) | scelta e uso degli encoder |
| [BENCHMARK_LOCALIZE.md](BENCHMARK_LOCALIZE.md) | 150 bug: dove sta il punto da correggere |
| [BENCHMARK_BUGFIX.md](BENCHMARK_BUGFIX.md) | bug corretti dal modello locale |
| [BENCHMARK_TRACEBACK.md](BENCHMARK_TRACEBACK.md) | traceback fissati in testa |
| [BENCHMARK_CLAUDE_CODE.md](BENCHMARK_CLAUDE_CODE.md) | A/B con Claude Code: costi e qualità |
| `results/` | dati grezzi dei benchmark |
