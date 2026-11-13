# Guida: encoder neurale reale e scelta del rango SVD

Obiettivo: sostituire `HashingEncoder` (baseline lessicale, solo per test) con un embedding model su CPU e
scegliere il rango SVD **misurando** sul proprio codice, non a priori.

## 1. Modello consigliato: `Qwen/Qwen3-Embedding-0.6B`

Dati verificati sul model card (`config.json`, `1_Pooling/config.json`, `config_sentence_transformers.json`):

| proprietà | valore | impatto su zt-memory |
|---|---|---|
| backbone | `Qwen3ForCausalLM`, 28 layer, hidden 1024 | `encoder.dim = 1024` → rango SVD ≤ 1024 |
| pesi | bf16, 1.19 GB su disco | RAM: ~1.2 GB in bf16, ~2.4 GB in fp32 |
| pooling | ultimo token (`pooling_mode_lasttoken`) | preset `pooling="last"` già impostato |
| istruzione | solo lato query: `Instruct: {task}\nQuery:{query}` | preset `query_prefix` orientato al codice |
| MRL | dimensioni 32–1024 per troncamento | lo script confronta SVD vs troncamento "prefix" |
| transformers | ≥ 4.51 (`KeyError: 'qwen3'` altrimenti) | extra `[hf]` richiede `>=4.51` |

Il padding è forzato a destra: è esatto per qualunque modello causale, quindi il vettore di un chunk non
dipende dal batch e un resync senza modifiche non genera sorpresa (test `test_hf_encoder.py`).

## 2. Installazione e download

```bash
pip install -e ".[hf,treesitter]"
hf download Qwen/Qwen3-Embedding-0.6B          # oppure: huggingface-cli download ... (huggingface_hub < 0.34)
export HF_HUB_OFFLINE=1                         # dopo il download: nessuna rete a runtime
```

Serve accesso a `huggingface.co` solo per il download. In alternativa, puntare a una directory locale:
`--encoder hf:/percorso/Qwen3-Embedding-0.6B`.

## 3. Probe: 10–30 query reali con risposta nota

`probes.json`, preferibilmente da issue/commit veri del repository:

```json
[
  {"query": "retry with exponential backoff on HTTP 503", "expected": "src/http/retry.py"},
  {"query": "ERR_SESSION_EXPIRED handling", "expected": "src/auth/"},
  {"query": "where are invoice totals rounded", "expected": "src/billing/totals.py"}
]
```

`expected` è un prefisso di percorso. I chunk dei file attesi entrano sempre nel campione.

## 4. Calibrazione

```bash
python examples/calibrate_encoder.py --repo ~/code/app \
    --encoder hf:Qwen/Qwen3-Embedding-0.6B --samples 2000 --ranks 64,128,192,256,384,512 \
    --probes probes.json --max-length 512 --threads 8 \
    --save-projector ~/.cache/zt/app-proj.pt --out report.json
```

Un solo passaggio di encoding serve tutti i ranghi (le basi PCA sono annidate). Output per rango:

| colonna | significato | come leggerla |
|---|---|---|
| `EVR` | varianza cumulativa spiegata | indicativa: non misura il retrieval |
| `nbr@10` | quota dei 10 vicini (coseno, spazio pieno dell'encoder) conservati dopo la proiezione | **metrica primaria**: cosa perde il retrieval per la compressione |
| `probe@10` | quota delle probe con il file atteso nei top-10 | verifica sul compito reale |
| `B/node` | byte SSD per nodo (posizione + velocità, float32) | 8 × rango |
| riga `prefix` | primi r valori grezzi (troncamento MRL) | se ≥ SVD, la fase SVD non aggiunge nulla per questo encoder |

Regole pratiche:

- `--samples` ≥ 4 × rango massimo, altrimenti la stima ad alto rango è inaffidabile (lo script avvisa).
- `--max-length`: i chunk arrivano a 120 righe (~1–1.5k token); 512 tronca la coda dei simboli lunghi, 1024
  costa circa il doppio in CPU. Lo stesso valore va usato in calibrazione e in `zt init`.
- `dtype=bfloat16` conviene solo su CPU con AVX512-BF16/AMX; altrimenti `float32` (default) è più veloce.
- Lo script stampa chunks/s e il tempo stimato per l'ingest completo del repository: è il costo una tantum,
  i sync successivi re-embeddano solo i simboli modificati.

Scelta del rango: il più piccolo con `nbr@10 ≥ 0.90` e `probe@10` vicino al massimo. Lo spazio su SSD è
trascurabile (100k chunk × rango 256 ≈ 200 MB); il costo reale del rango è la latenza della scansione esatta
(lineare in N × r). Nel dubbio, rango più alto.

Riferimento con l'encoder di test sulla stdlib (non trasferibile a un encoder neurale, serve a leggere la
tabella): rango 64/128/256/512 → `nbr@10` 0.48/0.62/0.76/0.87; il troncamento prefix non ha senso per
feature hashing (0.07–0.48). Con l'encoder reale questi numeri vanno rimisurati.

## 4b. Risultati misurati con Qwen3-Embedding-0.6B (CPU Windows, codice di zt-memory, 600 chunk)

| metodo | rango 64 | rango 128 | rango 256 |
|---|---|---|---|
| `svd` (centrata), nbr@10 | 0.836 | 0.876 | 0.888 (tetto di centratura 0.886) |
| `svd-raw` (non centrata), nbr@10 | 0.868 | **0.938** | 0.984 |
| `prefix` (Matryoshka), nbr@10 | 0.675 | 0.762 | 0.823 |

Probe: 11/12 per tutti i metodi (non discriminanti). Scelta: `svd-raw` rango 128, ora default (`svd_center=False`).
Velocità 2.3 chunk/s su CPU con batch ordinati per lunghezza (1.6 prima), RAM di picco 5.4 GB in fp32.

Primo ingest di un repository grande su GPU (serve torch con CUDA), poi sync incrementali su CPU:

```bash
zt sync --device cuda --dtype float16 --batch-size 32   # la VRAM viene liberata a fine comando
```

## 5. Inizializzazione con la base calibrata

```bash
zt init --repo ~/code/app --encoder hf:Qwen/Qwen3-Embedding-0.6B --rank 256 --max-length 512 \
        --projector ~/.cache/zt/app-proj.pt
zt sync                 # primo ingest completo (unico passaggio lungo)
zt hook install         # sync in background dopo commit/merge/checkout/rebase
```

Con `--projector` la base non viene ricalcolata al primo ingest (niente doppio encoding del corpus di
calibrazione). Senza, il manager calibra da solo sui chunk del repository.

## 6. Cambiare encoder o rango dopo

Encoder, rango, istruzione di query e curvatura fanno parte dell'impronta dello store: aprirlo con valori
diversi solleva `IncompatibleStoreError` (i vettori salvati non sarebbero confrontabili). Procedura:
nuovo `zt init --force` con i nuovi parametri, cancellare `.zt/store`, `zt sync`. Attenzione: requisiti e
decisioni (nodi SPEC/DECISION) vivono solo nello store e vanno re-inseriti (`seed_requirements`).

## 7. Problemi comuni

| sintomo | causa | rimedio |
|---|---|---|
| `KeyError: 'qwen3'` | transformers < 4.51 | `pip install -U "transformers>=4.51"` |
| download bloccato / 403 | rete senza accesso a huggingface.co | scaricare altrove e usare `hf:/percorso/locale` |
| OOM durante l'encoding | batch × max_length troppo grandi | `--batch-size 4`, `--max-length 512` |
| `IncompatibleStoreError` | encoder/rango cambiati | sezione 6 |
| `zt: ... is locked by pid N` (exit 3) | un altro processo (watch, hook) sta sincronizzando | attendere o `--lock-timeout 60` |
