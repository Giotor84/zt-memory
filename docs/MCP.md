# zt-memory come server MCP

`zt mcp` espone la memoria di una repository a qualsiasi client MCP (Claude Code, Claude Desktop, Cursor, Cline, agenti locali)
sul trasporto **stdio** (JSON-RPC 2.0, un messaggio per riga). Nessuna dipendenza in più: il protocollo è
implementato nel pacchetto.

## Tool

| tool | argomenti | cosa restituisce |
|---|---|---|
| `zt_query_context` | `query` (obbligatorio), `token_budget`, `line_numbers`, `mode` | gli span di codice esatti più rilevanti (`path  Lstart-end  [simboli]`), tagliati al budget, con le decisioni collegate. Se la domanda contiene un traceback, i simboli alle righe indicate vanno per primi. Aggiorna l'indice prima (solo i file cambiati) |
| `zt_record_decision` | `summary` (obbligatorio), `paths` (file o `file::Simbolo`) | id della decisione e collegamenti creati; i percorsi sconosciuti vengono elencati |
| `zt_status` | `limit` | simboli più "caldi" (modificati o recuperati di recente), file cambiati dall'ultimo sync, dimensione dello store |
| `zt_read_span` (solo con `zt mcp --read-span`) | `spans` (da 1 a 3, `path:L<inizio>-<fine>`), `line_numbers` | le righe chieste, lette dal disco in quel momento: al massimo 300 righe per tratto e 600 per chiamata. È il giro `NEED` via MCP |

Con più progetti (`--project`, vedi Claude Desktop) ogni tool ha in più l'argomento obbligatorio `project`.

### `mode="locations"` (default di `zt mcp`): solo dove guardare

Con `mode="locations"` — il default di `zt mcp`; `zt mcp --mode code` o `mode="code"` nella chiamata per avere il
codice — il tool non restituisce il codice ma
una lista, dal risultato migliore in giù, di file, righe, firma e prima frase della docstring, entro 500 token
(`token_budget` per cambiarli):

```
zt-memory: 14 location(s) in 6 file(s), ~467 tokens, best first (signatures only: read the lines you need)
- src/click/core.py  L2434-2445  def get_error_hint(self, ctx: Context) -> str:  — Get a stringified version of the param ...
- src/click/core.py  L1579-1587  def add_command(self, cmd: Command, name: str | None = None) -> None:  — Registers another ...
```

Serve ai client che leggono i file da soli (Claude Code, Cursor): il codice restituito da zt resta nella conversazione
e viene riletto a ogni turno successivo, mentre una lista di posizioni costa circa 10 volte meno e il client apre
solo le righe che gli servono. La selezione è la stessa della modalità `code` (con il budget del progetto, 12000);
cambia solo cosa viene restituito. Su 25 bug reali con Claude Code costa quanto Claude da solo (-3% in totale),
mentre 4000 token di codice costavano il 16% in più (mediana): [BENCHMARK_CLAUDE_CODE.md](BENCHMARK_CLAUDE_CODE.md).

### `mode="focus"`: pochi corpi interi + mappa

È il preset `api` del router (vedi [NEED.md](NEED.md)): toglie i doppioni (stub, blocchi quasi identici), manda
**interi i 5 blocchi migliori** (più quelli con punteggio almeno 0,6 volte il migliore) e tutti gli altri come
**mappa**, cioè `path  Lx-y  firma — docstring`, fino a 20 firme in più. Se nella domanda c'è un traceback: 10 corpi
interi, senza mappa. La selezione usa il budget del progetto (12000), salvo `token_budget` nella chiamata.

```
zt-memory: 8 whole, 53 as signatures from 16 file(s), ~3158 tokens (budget 12000)
...
## Other relevant code (signatures only: get a body with zt_read_span (path:Lstart-end) if you need it)
- src/attr/_make.py  L2673-2701  class Converter:  — Stores a converter callable.
```

Il corpo di una firma della mappa si chiede con `zt_read_span` (se attivo) o con lo strumento di lettura del client.
Si attiva per chiamata (`mode="focus"`) o come default del server (`zt mcp --mode focus`).

### `zt_read_span`: leggere solo le righe che servono

Il client chiede fino a 3 intervalli di righe, copiati dalla mappa o dalle posizioni (`src/app/db.py:L120-180`).
Serve soprattutto ai client **senza uno strumento proprio per leggere file**, come Claude Desktop. In Claude Code
c'è già `Read`. Non usa l'encoder né lo store: legge solo i file, quindi non occupa RAM e non blocca `zt sync`.

Si legge **solo un file che zt indicizzerebbe**. Questa regola è la stessa usata dal giro `NEED` di `zt code`
(`spans.py`). Il tool rifiuta:
- i file in `exclude_globs` (in un progetto reale, i quattro file vietati). Il controllo vale sia sul percorso scritto sia sul
  percorso reale, quindi un link simbolico verso un file escluso viene rifiutato;
- i percorsi fuori dal progetto: assoluti, con `..`, oppure link che portano fuori;
- i file e le cartelle nascoste (`.git`, `.zt`, `.env`), le cartelle escluse (`node_modules`, `.venv`, ...) e i file
  che non sono codice o testo;
- i file ignorati da git e quelli più grandi di `max_file_bytes`.

Un rifiuto dice solo il motivo: niente contenuto, e per i file esclusi nemmeno se esistono.

Gli errori di argomento tornano al modello come `isError: true` (può correggersi); tool o metodi sconosciuti sono
errori di protocollo (`-32602`, `-32601`).

## Prima di collegarlo

```powershell
cd C:\percorso\della\repo
zt init --encoder hf:Qwen/Qwen3-Embedding-0.6B --rank 128
zt sync          # il primo è lento su CPU (~2.3 pezzi/s): farlo prima, non dentro il client
zt hook install  # opzionale: sync in background a ogni commit/checkout
```

**File che l'agente non deve leggere** (dati, segreti, file vietati dalle regole del progetto): da impostare in
`.zt/config.toml` **dopo `zt init` e prima del primo `zt sync`**

```toml
[memory]
exclude_globs = "debtors.py, test_pipeline_safety.py, PROJECT_STATE.json, tests/fixtures/*.json"
```

Ogni modello è confrontato con la fine del percorso relativo (`debtors.py` ovunque, `pkg/debtors.py` solo lì;
`*` non attraversa le cartelle), senza distinguere maiuscole e minuscole. Un file escluso dopo essere stato
indicizzato viene tolto dall'indice al sync successivo. È una lista di esclusioni: un file nuovo con dati sensibili
entra finché non lo aggiungi. Prima del primo sync controlla l'elenco esatto dei file che entrerebbero con `zt status --json` (campo `pending.added`).

`zt_query_context` fa comunque un sync incrementale a ogni chiamata, quindi l'indice segue anche le modifiche non
ancora committate.

## Claude Code

Una volta sola, a livello utente, **senza `--repo`**: il server parte in ogni sessione e cerca `.zt/` risalendo dalla
cartella della sessione.

```powershell
claude mcp add --scope user zt-memory -e HF_HUB_OFFLINE=1 -- C:\path\to\zt-memory\.venv\Scripts\zt.exe mcp
```

Con `--repo PERCORSO` invece il server serve sempre quella repository, da qualsiasi cartella.

Oppure, per condividerlo con chi usa la stessa repo, un file `.mcp.json` nella radice:

```json
{
  "mcpServers": {
    "zt-memory": {
      "command": "C:\\path\\to\\zt-memory\\.venv\\Scripts\\zt.exe",
      "args": ["mcp", "--repo", "C:\\percorso\\della\\repo"]
    }
  }
}
```

## Claude Desktop

Claude Desktop non ha una cartella di sessione e non legge file da solo: servono `--read-span` e il percorso dei
progetti. Più progetti stanno **in una sola voce** con `--project` (ripetibile, `NOME=PERCORSO`; senza nome si usa il
nome della cartella). File `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "zt-memory": {
      "command": "C:\\path\\to\\zt-memory\\.venv\\Scripts\\zt.exe",
      "args": ["mcp", "--read-span", "--mode", "focus",
               "--project", "C:\\path\\to\\zt-memory",
               "--project", "legale=C:\\percorso\\progetto"],
      "env": {"HF_HUB_OFFLINE": "1"}
    }
  }
}
```

- Ogni progetto ha il suo `.zt/config.toml` (con i suoi `exclude_globs`) e il suo encoder. L'encoder si carica alla
  prima chiamata su quel progetto e viene liberato dopo 15 minuti senza chiamate.
- Ogni tool chiede `project`. Se non è chiaro di quale progetto si parla, le istruzioni del server dicono a Claude di
  chiederlo.
- `--mode focus` in Desktop è una scelta ragionata ma **non misurata**: senza `Read`, una lista di posizioni
  obbliga a una chiamata `zt_read_span` in più. La misura A/B è stata fatta in Claude Code
  ([BENCHMARK_CLAUDE_CODE.md](BENCHMARK_CLAUDE_CODE.md)).
- `--repo` e `--project` non si usano insieme. Con un solo progetto basta `--repo`.

## Cursor / Cline

Stesso blocco `mcpServers` in `.cursor/mcp.json` (Cursor) o in `cline_mcp_settings.json` (Cline).

## Come si comporta

- **Progetti senza zt**: il server parte comunque. Se viene chiamato un tool, risponde (come errore leggibile dal
  modello) "zt non è attivo per questo progetto" con i passi per attivarlo; nessun modello viene caricato.
- **Attivazione durante la sessione**: il progetto viene cercato a ogni chiamata finché non lo trova, quindi dopo
  `zt init` + `zt sync` il tool funziona **senza riavviare** Claude Code. Con `zt init` fatto ma senza sync, il tool
  risponde "indice mai costruito" e chiede di lanciare `zt sync` (il primo è lungo: non viene mai fatto dentro una
  chiamata).
- **Dimensione delle risposte**: con `locations` al massimo 500 token; con `code` al massimo 4000, salvo `token_budget`
  nella chiamata o `zt mcp --budget N`. Il `[router] token_budget` del progetto (12000) vale per `zt code`, `zt route`
  e per la selezione in modalità `locations`.
- **RAM**: l'encoder (Qwen3, 1–2 GB) si carica alla **prima chiamata** (qualche secondo) e viene liberato dopo
  15 minuti senza chiamate (`--idle-unload SECONDI`, 0 = mai). `--preload` lo carica all'avvio.
- **Aggiornamento dell'indice**: a ogni chiamata, se sono cambiati al massimo 100 file (`--max-inline-sync N`) fa un
  sync incrementale; oltre, risponde con l'indice attuale e lo dice nell'intestazione ("index NOT refreshed").
  Gli span restituiti sono comunque riletti dal disco se il file è cambiato.
- **Lock**: lo store viene aperto solo durante una chiamata e chiuso subito dopo, quindi hook git, `zt watch` e
  `zt code` continuano a funzionare. Se un altro processo sta sincronizzando, il tool risponde "store busy, retry".
- **stdout** contiene solo messaggi del protocollo; log e barre di avanzamento vanno su stderr.
- **Protocollo**: handshake `initialize` con le versioni 2025-11-25, 2025-06-18, 2025-03-26, 2024-11-05. I client
  recenti (SDK `mcp` 2.x, protocollo 2026-07-28) provano prima `server/discover`, ricevono "method not found" e
  ripiegano sull'handshake: verificato con l'SDK Python ufficiale `mcp` 2.3.0 (negozia 2025-11-25, i tre tool
  funzionano).
- **GPU**: l'encoder usa il device di `.zt/config.toml` (CPU di default). Con un client cloud la GPU è libera:
  `zt mcp --device cuda` accelera il sync. Con llama-server acceso resta su CPU.

## Quanto fa risparmiare

Misurato con Claude Code su 45 bug reali ([BENCHMARK_CLAUDE_CODE.md](BENCHMARK_CLAUDE_CODE.md)): su progetti piccoli,
medi e grandi (sympy, 1600 file), con log o con descrizioni a parole in italiano, Claude con Grep trova già il file
giusto da solo. zt quindi **non fa risparmiare**: restituendo codice costa dal 7% al 37% in più, restituendo solo
posizioni (il default) costa quanto Claude da solo. Non misurato: codice con nomi in italiano, modelli più piccoli,
sessioni lunghe.
