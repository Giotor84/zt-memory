# Backend Amazon Bedrock (zt 0.4.0)

`zt code`, `zt engine` e il benchmark `examples/benchmark_bugfix.py` possono usare un modello Amazon Bedrock (API
Converse di boto3) al posto di un server compatibile OpenAI (llama.cpp, vLLM). Indice, router, patch SEARCH/REPLACE,
test e `zt undo` non cambiano: cambia solo chi scrive la risposta.

Pensato per ambienti come SageMaker Studio air-gapped (SageMaker senza internet, Bedrock raggiungibile solo tramite proxy).

## Installazione

```bash
pip install "zt-memory[bedrock]"      # aggiunge boto3>=1.34
# senza internet: dalla wheelhouse, es. pip install --no-index --find-links /percorso/wheelhouse "zt-memory[bedrock]"
```

boto3 viene importato solo quando serve: se manca, `zt engine` lo dice ("install the bedrock extra") e il resto di zt
funziona come prima.

## Configurazione (`.zt/config.toml`)

```toml
[engine]
backend = "bedrock"                                # default "openai": il comportamento di sempre
model = "minimax.minimax-m2.5"                     # oppure "qwen.qwen3-235b-a22b-2507-v1:0"
region = "eu-central-1"
proxy = "http://proxy.example.internal:3128"    # vuoto: HTTPS_PROXY dell'ambiente o nessun proxy
# ctx = 196000                                     # facoltativo: senza, vale quello del profilo del modello
max_new_tokens = 8192                              # tagliato al massimo del profilo (8192)
temperature = 0.0
timeout_s = 900
retries = 1                                        # zt code: quante volte richiede la patch con l'errore (ogni volta è una chiamata)
aws_retries = 3                                    # botocore, modo "standard": backoff su throttling ed errori transitori
max_calls = 20                                     # tetto di chiamate per processo (limite di costo)
```

Le credenziali vengono dal ruolo IAM dell'istanza (o dalle variabili `AWS_*`): **non vanno mai nel config**, e zt
non le legge né le scrive. Con `backend = "bedrock"` la chiave `url` viene ignorata.

### Due differenze rispetto alla specifica iniziale

1. **`retries` e `aws_retries` sono due chiavi diverse.** In zt `retries` esisteva già: è il numero di volte in cui
   `zt code` richiede la patch al modello quando un blocco SEARCH non corrisponde, e **ogni volta è una chiamata a
   pagamento con tutto il contesto**. Usare la stessa chiave anche per i tentativi di botocore (come nella bozza:
   `retries = 3`) avrebbe portato `zt code` a fare fino a 4 chiamate per compito. I tentativi di rete stanno quindi
   in `aws_retries` (default 3).
2. **Il benchmark usa 16384 token di contesto anche con Bedrock**, salvo `--served-ctx`. La condizione "file interi"
   riempie il contesto fino al limite: con la finestra di MiniMax (196k token) sarebbe un prompt da ~190k token a
   chiamata e un esperimento diverso da quelli già fatti con il 14B locale.

`zt code` invece non rischia: il contesto inviato è al massimo `[router] token_budget` (12000 di default), qualunque
sia `ctx`.

## Comandi

```bash
zt engine                    # configurazione e profilo del modello; nessuna chiamata
zt engine --probe            # UNA chiamata minima ("Reply only with: OK"): deve rispondere OK
zt code "..." --dry-run      # contesto + patch mostrata, niente scritto
zt code "..." --check "python -m pytest -q" --revert-on-fail
python examples/benchmark_bugfix.py run --workdir DIR --backend bedrock --model minimax.minimax-m2.5 \
    --proxy http://proxy.example.internal:3128 --max-calls 60
```

Esempio di `zt engine`:

```
backend  bedrock  model minimax.minimax-m2.5
profile  minimax-m2.5: context 196000, max output 8192, reasoning yes
aws      region eu-central-1, proxy http://proxy.example.internal:3128, credentials from the environment (IAM role), ...
limits   max_calls 20 per process, max_new_tokens 8192 (sent: 8192), usage log .../.zt/engine_usage.jsonl
budget   router gets up to 12000 tokens ([router] token_budget, within context - max_new_tokens - margin)
boto3    available
probe    not run (zt engine --probe makes one paid call)
```

## Profili dei modelli

| profilo | model ID | contesto | output massimo | ragionamento | campi extra inviati |
|---|---|---|---|---|---|
| minimax-m2.5 | inizia con `minimax.` | 196.000 | 8.192 | sì | nessuno |
| qwen3 | inizia con `qwen.` | 256.000 | 8.192 | no | `enable_thinking: false` |
| generico | tutti gli altri | da `ctx` | 8.192 | no | nessuno |

Il prefisso regionale dei profili di inferenza (`eu.`, `us.`, `apac.`, `global.`) viene ignorato nel confronto.

## Modelli di ragionamento (MiniMax M2.5)

- MiniMax prima "pensa" (blocchi `reasoningContent`) e poi risponde (blocchi `text`). **Solo il testo** arriva al
  parser delle patch: un blocco SEARCH/REPLACE abbozzato nel ragionamento non può essere applicato per sbaglio.
  La lunghezza del ragionamento finisce nel log (`reasoning_chars`).
- Il ragionamento consuma `maxTokens`. Se il modello esaurisce il budget prima di scrivere il testo
  (`stopReason = max_tokens` con risposta vuota) zt si ferma subito con "aumenta max_new_tokens": riprovare con lo
  stesso budget spenderebbe di nuovo senza risultato.
- Se la risposta è vuota per un altro motivo (`end_turn`), con MiniMax zt riprova **una** volta; con gli altri profili
  si ferma.
- Per questo `zt engine --probe` usa 128 token: con 16 MiniMax spende tutto nel ragionamento e il testo resta vuoto.

## Limite di costo

- **Nessuna chiamata nascosta**: `zt engine`, il controllo iniziale di `zt code` e del benchmark costruiscono solo il
  client boto3 (nessuna richiesta al modello). Il conteggio dei token è stimato (3 caratteri per token, +10%): Bedrock
  non espone il tokenizer.
- **`max_calls`** (default 20) è un tetto per processo: oltre, zt si ferma con "max_calls reached". Il benchmark in quel
  caso salva i risultati fatti, non registra il caso interrotto e riprende da lì al lancio successivo.
- **Registro**: ogni chiamata aggiunge una riga a `.zt/engine_usage.jsonl` (per il benchmark: `DIR/engine_usage.jsonl`)
  con ora, modello, token in ingresso e in uscita, `stopReason`, secondi, eventuale codice d'errore AWS. Nessun prezzo
  è scritto nel codice: il costo si calcola dal registro con il listino del momento.
- `zt code` stampa a fine compito quante chiamate e quanti token ha usato.

## Errori

| messaggio | causa probabile |
|---|---|
| `boto3 is not installed: install the bedrock extra` | manca l'extra `bedrock` nell'ambiente |
| `Bedrock call failed (AccessDeniedException)` | il ruolo IAM non ha `bedrock:InvokeModel` su quel modello |
| `Bedrock call failed (ValidationException)` | model ID sbagliato o modello non abilitato nella regione |
| `Bedrock call failed (ThrottlingException)` | troppe richieste: botocore ha già riprovato `aws_retries` volte |
| `Bedrock call failed (EndpointConnectionError / ProxyConnectionError / ConnectTimeoutError)` | proxy o rete: controlla `proxy` e `region` |
| `empty answer ... aumenta max_new_tokens` | il ragionamento ha consumato tutto il budget di output |
| `max_calls reached` | tetto di costo raggiunto: alza `max_calls` se è voluto |

Gli errori di Bedrock fanno uscire `zt code` con il codice 6, come un server locale irraggiungibile.

## Cosa non è ancora misurato

- La **qualità delle patch** di MiniMax M2.5 e Qwen3-235B sul benchmark dei bug di click (con il 14B locale: 1 bug
  risolto su 14, file giusto nel contesto nel 93-100% dei casi).
- Tempi e costi reali per compito in un ambiente SageMaker air-gapped.
- Verifica dal vivo, a carico del proprietario: `zt engine --probe` deve rispondere `OK` con MiniMax, poi
  `zt code --dry-run` su un repository piccolo.

I test automatici (`tests/test_engine_bedrock.py`) usano un client boto3 finto: verificano la traduzione delle
richieste, i profili, il ragionamento ignorato, le risposte vuote, il limite di chiamate, il registro e i comandi,
senza rete e senza boto3 installato.
