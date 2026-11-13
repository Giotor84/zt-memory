# Motore di calcolo su Windows (llama.cpp + Qwen2.5-Coder-14B) e ciclo `zt code`

## Verificato

- Asset di release e flag letti dal sorgente di llama.cpp (build 11718) e provati con il binario reale: la riga di
  `start_llama.ps1` viene accettata; `-fa` senza valore è un errore (ora vuole `on|off|auto`).
- `llama-server` reale con un modello Qwen2 minuscolo: `/health`, `/props` (contesto per slot), `/tokenize`, chat.
  `zt code` legge il contesto dal server, conta i token esattamente e rifiuta pulito una risposta inutilizzabile.
- Misurato sul PC di sviluppo (RTX 4060 Ti 16 GB, driver 610.88, llama.cpp b11719 CUDA 12.4): ~10 GB di VRAM per il
  server (11.2 GB totali col desktop), 23.8–27 token/s; `examples/e2e_fix_bug.py` superato (patch corretta al primo
  tentativo, 1092 token di contesto, 6.5 s di LLM).
- Non ancora misurato: la percentuale di bug veri risolti → `examples/benchmark_bugfix.py` ([BENCHMARK_BUGFIX.md](BENCHMARK_BUGFIX.md)).

## 1. Installazione (una volta)

```powershell
cd C:\path\to\zt-memory
.\examples\windows\setup_llama.ps1          # llama.cpp CUDA 12.4 in C:\llama + modello in C:\models
```

Requisito: `nvidia-smi` deve mostrare `CUDA Version` ≥ 12.4 (altrimenti aggiornare il driver NVIDIA).

## 2. Avvio del server (in una finestra PowerShell dedicata)

```powershell
.\examples\windows\start_llama.ps1           # -Ctx 12288 se la VRAM libera è poca
```

| componente | VRAM stimata |
|---|---|
| pesi Q4_K_M (8.99 GB su disco) | 8.4 GiB |
| KV cache q8_0 (48 layer × 8 teste KV × 128 dim) | 0.1 MiB/token → 1.2 GiB @12288, 1.6 GiB @16384 |
| buffer di calcolo + contesto CUDA | ~0.7–1.2 GiB |
| **totale** | **~10.6 GiB @12288, ~11 GiB @16384** |

Il desktop di Windows usa anch'esso VRAM: controllare `nvidia-smi` dopo l'avvio. Il contesto deve contenere
domanda **e** risposta: con `-c 16384` e `max_new_tokens = 3072` il router riceve fino a 13 184 token (limitati a
12 000 dal router); con il taglio dinamico ne usa in media il 40% in meno.

## 3. Coesistenza VRAM/RAM

- L'encoder del navigatore (Qwen3-Embedding-0.6B) gira **sempre su CPU** durante `zt code`: se la configurazione
  dice `device = "cuda"`, `zt code` lo forza su CPU (`--allow-gpu-encoder` per disattivare la protezione).
- RAM: encoder fp32 ~2.4 GB + Python/torch ~1 GB nel processo `zt`; `llama-server` usa pochi GB di RAM perché i
  pesi stanno in VRAM.
- `zt sync --device cuda` (primo ingest veloce) va lanciato solo con `llama-server` **spento**.

## 4. Ciclo di modifica

```powershell
zt engine                                   # salute, contesto servito, velocità, budget del router
zt code "descrizione del bug o della modifica" --check "python -m pytest -q" --revert-on-fail
zt code "..." --dry-run                     # mostra solo il diff
zt undo                                     # ripristina l'ultima modifica e risincronizza
```

Flusso: sync → router (budget = contesto servito − risposta, verificato con `/tokenize`) → LLM con contratto
SEARCH/REPLACE → tutte le modifiche risolte in memoria (tutto-o-niente; se un SEARCH non combacia, il testo reale
del file viene rimandato al modello per un nuovo tentativo) → conferma → backup in `.zt/backups` → scrittura
atomica → `ingest_files` sui file toccati (sorpresa + impulso scaldano i simboli modificati) → test opzionale.

Codici di uscita: 0 applicato, 4 nessuna modifica utilizzabile, 5 test falliti (con `--revert-on-fail`: ripristinato),
6 server non raggiungibile.

## 5. Test end-to-end

```powershell
python examples\e2e_fix_bug.py                                        # navigatore con encoder di test (veloce)
python examples\e2e_fix_bug.py --encoder hf:Qwen/Qwen3-Embedding-0.6B  # navigatore reale
```

Crea un piccolo progetto con un bug vero (`apply_discount` tratta la percentuale come frazione, 3 test falliti),
lancia `zt code` dalla CLI reale e verifica: patch sul file giusto, test verdi, energia del simbolo modificato in
aumento. Esce con 0 solo se tutte le fasi passano.
