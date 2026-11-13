# zt-memory — context engine locale per LLM di codice

Indicizza una repository **per simbolo** (funzioni, classi, metodi, con tree-sitter) e a ogni domanda restituisce
solo gli span di codice utili, con percorso e righe esatte, entro un budget di token. Il navigatore non genera mai
token: unisce ricerca lessicale (BM25) e per embedding e aggiunge un piccolo bonus ai simboli toccati di recente
(l'"energia"). Tutto in locale: indice su SSD, encoder su CPU, nessun servizio esterno.

Tre modi di usarlo:

| uso | comando | per chi |
|---|---|---|
| **server MCP** | `zt mcp` | Claude Code, Cursor, Cline e agenti locali: dove guardare (file, righe, firme) o il codice esatto → [docs/MCP.md](docs/MCP.md) |
| **ciclo di modifica locale** | `zt code "..."` | un 14B su una GPU da 16 GB: contesto → patch verificata → test → memoria aggiornata → [docs/ENGINE_WINDOWS.md](docs/ENGINE_WINDOWS.md) |
| **modelli grandi via API (>=120B)** | `[router] preset = "api"` + `zt code` | contesto a fuoco + mappa di firme; il modello chiede il codice che gli manca con `NEED` (un giro in più con un prompt corto, uguale per ogni modello) → [docs/NEED.md](docs/NEED.md) |
| **modello cloud via Amazon Bedrock** | `zt code "..."` con `[engine] backend = "bedrock"` | ambienti senza GPU né internet (es. SageMaker con proxy): MiniMax M2.5, Qwen3-235B → [docs/BEDROCK.md](docs/BEDROCK.md) |
| **CLI / libreria** | `zt route "..."` | prompt pronto per qualsiasi LLM compatibile OpenAI |

## Cosa è misurato

| misura | risultato | dove |
|---|---|---|
| ciclo completo su RTX 4060 Ti 16 GB (Qwen2.5-Coder-14B Q4_K_M, contesto 16k) | bug corretto al primo tentativo, test verdi; ~10 GB di VRAM, ~27 token/s, 14 s in totale | `examples/e2e_fix_bug.py` |
| ricerca ibrida contro solo lessicale (requests, 91 domande dalla storia git, Qwen3-Embedding) | recall@10 0.845 contro 0.823, intervallo [+0.001, +0.038] | `examples/eval_git_history.py` |
| taglio dinamico del contesto (α = 0.25) contro nessun taglio | −40% token per −3.5 punti di copertura dei file giusti (requests) | `examples/eval_git_history.py` |
| token: contesto zt contro file interi a budget pieno (click, 2 casi, solo verifica tecnica) | 26–34% in meno (10.3–11.6k contro 15.6k) | `examples/benchmark_bugfix.py` |
| punto da correggere nel contesto, 150 bug reali in 6 repo (Python, Go, TypeScript) | zt 61% (uguale con Qwen3 e con l'encoder di test), BM25 sui simboli 58%, BM25 su file interi 19% (65 casi a 2, p < 0.0001), con il 29–31% di token in meno dei file interi | [docs/BENCHMARK_LOCALIZE.md](docs/BENCHMARK_LOCALIZE.md) |
| stessi 150 bug, domande in italiano senza nomi di codice | zt con Qwen3 41%, zt con encoder di test 28%, BM25 sui simboli a parità di token 27% (21 casi a 0, p < 0.0001): qui Qwen serve | [docs/BENCHMARK_LOCALIZE.md](docs/BENCHMARK_LOCALIZE.md) |
| le stesse domande tradotte in inglese dal client | zt con Qwen3 57%, con encoder di test 56%, BM25 a parità di token 51%: tradurre vale più di Qwen | [docs/BENCHMARK_LOCALIZE.md](docs/BENCHMARK_LOCALIZE.md) |
| domanda = output di pytest dei test che falliscono, 42 bug reali (attrs, click, flask) | zt 60%, senza riconoscere il traceback 52% (3 a 0); dove il log indica il codice sorgente 10/12 contro 7/12 | [docs/BENCHMARK_TRACEBACK.md](docs/BENCHMARK_TRACEBACK.md) |
| contesto a fuoco (offline, 342 domande): corpo solo per i blocchi migliori, firme per gli altri | anti-cloni -3/4% di token senza perdite; fuoco F=5 -65% di token ma -12 punti sul punto giusto con domande descrittive; con traceback F=10 -50% senza perdite. Non attivo di default | [docs/BENCHMARK_FOCUS.md](docs/BENCHMARK_FOCUS.md) |
| bug risolti dal 14B locale, 14 bug di click (due giri) | 1/14 con zt, 0/14 con BM25: il modello è al limite, non decide. Patch applicabili 11/14 contro 4/14 | [docs/BENCHMARK_BUGFIX.md](docs/BENCHMARK_BUGFIX.md) |
| Claude Code con e senza zt via MCP, 20 bug reali con il log di pytest (60 sessioni) | stessa qualità (20/20 senza zt); risposte da 12000 token: +37% di costo; da 4000 (ora il default): costo pari. Con log precisi e repo medie zt **non fa risparmiare** | [docs/BENCHMARK_CLAUDE_CODE.md](docs/BENCHMARK_CLAUDE_CODE.md) |
| Claude Code, bug descritti a parole in italiano: 16 bug in repo medie + 9 in sympy (1609 file), 75 sessioni | con 4000 token di codice: qualità pari, costo +16/+24% (mediana). Con `mode="locations"` (solo posizioni, ora il default di `zt mcp`): 24/25 bug contro 21/25 senza zt, costo **-3%** in totale | [docs/BENCHMARK_CLAUDE_CODE.md](docs/BENCHMARK_CLAUDE_CODE.md) |

## Installazione

```bash
pip install -e ".[dev,hf]"            # + ",bedrock" per il backend Amazon Bedrock (boto3)
zt init --encoder hf:Qwen/Qwen3-Embedding-0.6B --rank 128   # scrive .zt/config.toml, esclude .zt/ da git
zt sync                                                      # il primo indicizza tutto; poi solo i simboli cambiati
zt hook install                                              # facoltativo: sync in background a ogni commit
```

## Comandi

```bash
zt sync | zt status | zt watch                  # indice incrementale, stato, polling durante l'editing
zt route "fix retry logic on 503"               # prompt con il contesto selezionato
zt mcp [--repo PATH] [--mode code]              # server MCP su stdio (3 tool, default: solo posizioni); senza --repo trova .zt/
zt engine [--probe]                             # server LLM: salute, contesto, velocità (Bedrock: profilo; --probe = 1 chiamata)
zt code "fix retry logic on 503" --check "python -m pytest -q" --revert-on-fail
zt undo                                         # annulla l'ultima modifica di zt code
python examples/benchmark_bugfix.py mine|index|run|report --workdir DIR   # benchmark end-to-end
```

Codici di uscita: 0 ok, 1 errore, 2 uso, 3 store occupato, 4 nessuna modifica utilizzabile, 5 test falliti,
6 server LLM irraggiungibile.

## Come funziona

```
domanda ─┬─► encoder (CPU) ► SVD rango r ► k-NN sullo store (SSD, memmap) ──────────┐
         └─► token code-aware ► FTS5/BM25 (SQLite) ─────────────────────────────────┤
             espansione dai seed: archi spec/decisione ↔ codice, header dei file ◄──┤
             RRF: w_g/(k+1+rank_geo) + w_l/(k+1+rank_lex) · peso del tipo · (1 + w_E·energia)
             ► taglio dinamico ► span esatti da disco (verificati con hash) ► budget token ► LLM
```

- **Traceback**: se la domanda contiene un errore (`File "...", line N`, `file.py:N:`, `file.ts:N:C`, `file.go:N`),
  i simboli a quelle righe vanno per primi nel contesto, prima il sorgente e poi i test ([docs/BENCHMARK_TRACEBACK.md](docs/BENCHMARK_TRACEBACK.md)).
- **Identità per simbolo**: `file::Classe.metodo`, stabile quando il codice si sposta; overload come `nome~hash(firma)`.
- **Esclusioni**: `[memory] exclude_globs = "segreti.py, tests/fixtures/*.json"` tiene fuori dall'indice file che
  git traccia ma che un agente non deve leggere ([docs/MCP.md](docs/MCP.md)).
- **Sync incrementale**: un `stat()` per file; mtime cambiato ma stessi byte → nessun lavoro; poi diff per simbolo.
- **Energia**: ogni modifica (sorpresa dell'embedding) e ogni recupero scaldano il simbolo; si raffredda nel tempo
  (`E_t = γE + ‖v‖`, con inerzia β). Nel ranking pesa `w_E = 1.5`: guadagno piccolo, mai negativo nelle misure.
- **Geometria**: le posizioni stanno in una palla di Poincaré; il raggio separa i livelli (cartella, file,
  simbolo), mentre **dentro lo stesso livello il ranking equivale al coseno degli embedding**. L'espansione
  geodetica "ad albero" è stata **rimossa** (nessun effetto in 4 repo, peggiore in click). Le posizioni iperboliche
  restano perché costano poco e cambiarle vorrebbe dire ricostruire tutti gli store esistenti, senza un guadagno
  misurato. I vecchi `config.toml` con `geodesic_*` si caricano ancora (con un avviso).
- **Patch**: blocchi SEARCH/REPLACE, applicati tutti o nessuno; corrispondenza esatta, poi senza spazi finali, poi
  senza indentazione; tollera i ```` ``` ```` dei modelli chat; backup e `zt undo`.

## Moduli (`src/zt_memory/`)

| modulo | ruolo |
|---|---|
| `syntax.py`, `chunking.py` | simboli con tree-sitter (Python, JS, TS/TSX, Go, Rust, Java, C, C++) + fallback `ast` |
| `encoders.py`, `projector.py` | Qwen3-Embedding (o `HashingEncoder` per i test) + SVD troncata congelata |
| `lexical.py` | FTS5/BM25 con token code-aware, nello stesso DB SQLite |
| `storage.py`, `index.py`, `working_set.py` | memmap su SSD + SQLite WAL, ricerca a blocchi, cache RAM a budget |
| `hyperbolic.py`, `dynamics.py` | palla di Poincaré (distanza, mappe), energia con inerzia e attrito |
| `manager.py` | sync incrementale, decisioni/spec, rinforzo, lock tra processi |
| `router.py`, `stacktrace.py`, `need.py` | `route_to_compute_engine(query)` → `RoutedContext.render()` / `.render_context()`; punti `file:riga` degli errori; protocollo `NEED` |
| `mcp_server.py`, `outline.py` | server MCP stdio, senza dipendenze esterne; modalità `locations` (firme e righe, senza codice) |
| `engine_client.py`, `engine_bedrock.py`, `editing.py`, `coding.py` | client llama.cpp (contesto, tokenizer) e Amazon Bedrock (Converse, profili, limite di chiamate), patch, ciclo `zt code` |
| `bench_bugfix.py`, `replay.py`, `evaluation.py` | benchmark end-to-end, ablazioni sulla storia git, calibrazione del rango |
| `project.py`, `cli.py`, `watch.py`, `discovery.py` | `.zt/config.toml`, comandi, polling, file rispettando `.gitignore` |

## Budget hardware (16 GB RAM + 16 GB VRAM)

| risorsa | componente | valore |
|---|---|---|
| VRAM | Qwen2.5-Coder-14B Q4_K_M + KV q8_0 16k + buffer | **~10 GB misurati** (RTX 4060 Ti) |
| RAM | encoder Qwen3-Embedding-0.6B su CPU | ~5.4 GB RSS misurati durante la calibrazione |
| RAM | working set dello store (`ram_budget_mb`) | 1.5 GB (default) |
| SSD | 15k chunk, rango 64 | 40 MiB con l'indice FTS5 |

## Ablazioni sulla storia git (`examples/eval_git_history.py`)

Ogni messaggio di commit diventa una domanda e i file modificati sono la risposta; tutte le configurazioni sono
misurate sullo stesso stato della memoria (confronto appaiato, bootstrap al 95%).

Qwen3-Embedding-0.6B, requests, 91 domande:

| config | hit@1 | hit@5 | hit@10 | rec@10 | MRR |
|---|---|---|---|---|---|
| solo lessicale | 0.516 | 0.758 | 0.890 | 0.823 | 0.636 |
| solo embedding | 0.407 | 0.769 | 0.890 | 0.795 | 0.552 |
| ibrida | 0.516 | 0.813 | 0.890 | 0.827 | 0.649 |
| default (ibrida + energia 1.5) | 0.505 | **0.824** | 0.890 | **0.845** | 0.647 |

Con l'encoder di test (`HashingEncoder`, 4 repo, 643 domande) il lessicale porta quasi tutto il risultato; l'energia
a peso 1.5 dà recall@10 tra +0.002 e +0.027 in 4/4 repo. Le configurazioni ancora misurabili sono L, G, A, P
(energia), R (energia come terza lista RRF) e F (default).

## Limiti noti

- Con il 14B locale i bug **risolti** restano pochi (1/14 su click): zt porta il file giusto nel contesto, ma il
  modello spesso non scrive la correzione giusta ([docs/BENCHMARK_BUGFIX.md](docs/BENCHMARK_BUGFIX.md)).
- Domande in una lingua diversa da quella del codice: con l'encoder di test la resa cala molto (28% contro 61%);
  meglio Qwen3 o una domanda tradotta dal client ([docs/BENCHMARK_LOCALIZE.md](docs/BENCHMARK_LOCALIZE.md)).
- Il primo `zt sync` con Qwen3 su CPU è lento (~2.3 pezzi/s): su repo grandi conviene farlo con `--device cuda`
  a server LLM spento.
- Il taglio dinamico interviene poco quando molti span hanno punteggi simili: il risparmio di token dipende dalla
  domanda (26–40% nelle misure finora).
- Linguaggi senza grammatica tree-sitter → blocchi ancorati al contenuto (stabili salvo tagli a `chunk_max_lines`).
- Cambiare encoder o rango richiede di ricostruire lo store; le decisioni registrate (SPEC/DECISION) non si
  ricostruiscono dal codice.
- Consistenza SQLite↔memmap: compensazione in caso di rollback, non una transazione unica; lo store è comunque
  ricostruibile dalla repository.

## Licenza

Apache-2.0 — vedi [LICENSE](LICENSE) e [NOTICE](NOTICE).

Spiegazioni, decisioni e glossario: [docs/KNOWLEDGE_BASE.md](docs/KNOWLEDGE_BASE.md).

```bash
pytest   # 193 test
```
