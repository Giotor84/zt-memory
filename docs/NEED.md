# Modelli grandi via API: contesto a fuoco + mappa + `NEED`

Obiettivo: modelli da circa 120B parametri in su via API (Bedrock o API dirette: Claude, Llama 405B, Mistral Large,
DeepSeek, Qwen3-235B, MiniMax M2.5). Un modello così può leggere pochi corpi interi, guardare la **mappa** (firme con
le righe) e chiedere il codice che gli manca invece di indovinare.

Attenzione ai modelli MoE: Qwen3-235B-A22B ha 235B parametri ma 22B attivi per token. Che sappia usare `NEED` bene
è proprio quello che misura la fase 2, non un dato di partenza.

## Configurazione

```toml
[router]
preset = "api"        # anti-cloni + 5 corpi interi (e quelli >= 0,6 x il migliore) + fino a 20 firme in più
                      # con un traceback nella domanda: 10 corpi, niente mappa, niente NEED

[engine]
backend = "bedrock"   # o "openai" per qualsiasi API compatibile OpenAI (GPT, DeepSeek, Llama, Mistral...)
need_rounds = -1      # -1 = 1 giro in più con preset "api" (0 = mai)
```

Un solo meccanismo per tutti i modelli e tutti i provider: niente logiche diverse per chi ha la cache del prompt.

`preset = "single"` resta per chi vuole **una chiamata sola, senza secondo giro**: corpi interi finché coprono l'85% del
punteggio (minimo 3), con un traceback 15 corpi. È la variante M85 della fase 1b (-36% / -42% di token, 3 casi persi su
150 per set).

Le chiavi esplicite in `[router]` vincono sul preset (es. `focus_blocks = 3`).

## Il protocollo

**Giro 1**: il contesto a fuoco con la sezione "Other relevant code (signatures only)". Se la modifica sta in una delle
firme, il modello deve rispondere **subito**, senza analisi e senza abbozzi di patch, solo con:

```
NEED src/click/core.py:L2434-2445
KEEP src/click/core.py:L1579-1587
NOTE: il messaggio d'errore si costruisce in get_error_hint, non in add_command
END_NEED
```

- al massimo 3 `NEED`; `KEEP` facoltativo per i corpi del giro 1 che gli servono ancora; al massimo 2 righe `NOTE`;
- `END_NEED` chiude la risposta ed è passato anche come **stop sequence** alla chiamata (Bedrock `stopSequences`, OpenAI
  `stop`): appena il modello lo scrive, la generazione si ferma. Se un modello rifiuta le stop sequence (succede con
  alcuni modelli di ragionamento), zt riprova una volta senza e da lì in poi non le manda più;
- **limite**: nei modelli che ragionano prima di rispondere (MiniMax, DeepSeek R1) lo stop taglia solo la risposta
  visibile, non il ragionamento che la precede. Lì l'unico freno è l'istruzione "rispondi subito".

**Giro 2** (uguale per tutti i modelli): un prompt nuovo e corto con la domanda, le note del modello, il codice chiesto
(letto dal disco, solo dentro il repository, al massimo 300 righe per tratto e 600 in tutto), il codice da tenere
(`KEEP`), le intestazioni di quei file e il contratto. I corpi del giro 1 e la mappa **non** vengono rimandati, quindi la
cache del prompt non serve: con la cache, ripetere il giro 1 costerebbe comunque circa quanto questo prompt corto.

Se il modello chiede file inesistenti o fuori dal repository, la richiesta vale come risposta senza modifiche (il solito
giro di correzione, se `retries` > 0). Con un traceback nella domanda `NEED` non viene offerto.

Ogni giro è una chiamata a pagamento: conta in `max_calls` e finisce in `.zt/engine_usage.jsonl`. `zt code` stampa i
giri usati e i tratti chiesti.

## Quanto può costare (da verificare)

Stima sui numeri della fase 1: giro 1 ~3.700 token (E, -58% rispetto ad A = 8.771); giro 2 ~1.500 token. Sui casi che
fanno due giri il totale in ingresso è circa -40% rispetto ad A; in media si scende sotto la metà solo se il giro 2
serve in una parte dei casi. Attenzione: il punto da correggere è in uno dei 5 corpi nel 49% dei casi con domande di
commit, ma solo nel 14% con domande in italiano: lì `NEED` sarà la regola, non l'eccezione.

**I token in uscita contano di più**: per confrontare modelli diversi il report usa il **costo pesato
`ingresso + 4 × uscita`** (uscita con il ragionamento compreso). Per il costo vero servono i prezzi del modello.

## Fase 2: come la testiamo

`examples/benchmark_need.py`, in tre passi.

1. **`select`** (fatto nel cloud, dove ci sono i cloni): per ognuno dei 150 casi, istantanea del commit precedente la
   correzione, indice nuovo, contesto `api`. Tiene i casi **map-only** (le righe cambiate dalla correzione sono solo in
   una firma della mappa: è lì che `NEED` deve servire) e 6 **controlli** per set (righe già in un corpo: lì `NEED` non
   dovrebbe scattare). Scrive `need_pack.tar.gz`: istantanee senza storia git, domande, righe cambiate. Si porta in
   un ambiente SageMaker air-gapped così com'è.
2. **`run`** (in un ambiente SageMaker air-gapped, con Bedrock, o con qualsiasi API): per ogni compito due bracci, nessuna scrittura sui file:
   - **A**: il contesto di oggi, una chiamata;
   - **E**: contesto a fuoco + mappa + `NEED` (al massimo un giro in più).
3. **`report`**: modifica sul punto giusto (la patch tocca le righe cambiate dalla vera correzione, ±2 righe), patch
   applicabile, `NEED` chiesto e centrato, chiamate, token in ingresso e in uscita, costo pesato, McNemar A contro E.
   Una sezione per modello: lo stesso comando con modelli diversi nella stessa cartella si confronta alla pari.

```bash
# in un ambiente SageMaker air-gapped, dopo aver copiato need_pack.tar.gz e installato zt 0.4.x con l'extra bedrock
python examples/benchmark_need.py run --pack need_pack.tar.gz --workdir need_run --backend bedrock \
    --model minimax.minimax-m2.5 --proxy http://proxy.example.internal:3128 \
    --sets commit --limit 12 --max-calls 40
python examples/benchmark_need.py report --workdir need_run
```

Pilota consigliato: `--limit 12` = per ogni set di domande 8 map-only + 4 controlli, sempre gli stessi (ordine fisso del
pacchetto), al massimo 3 chiamate ciascuno (A 1, E 1-2): `--max-calls 40`. Se si ferma per il tetto, rilanciando
riprende da dove era. Per confrontare un altro modello basta cambiare `--model` (stessa `--workdir`).

**Cosa guardare**

- Nei **map-only**: E chiede con `NEED` il tratto giusto? E poi modifica il punto giusto più spesso di A?
- Nei **controlli**: E chiede `NEED` quando non serve (giro sprecato)?
- **Costo**: token in ingresso **e** in uscita di E contro A, sugli stessi compiti.

Limite: "modifica sul punto giusto" non vuol dire "bug risolto" (non ci sono test per questi casi). La prova con i test
si fa poi con `benchmark_bugfix.py` sui casi che hanno test.
