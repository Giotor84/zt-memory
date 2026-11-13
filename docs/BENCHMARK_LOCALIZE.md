# Benchmark di localizzazione (senza LLM)

Domanda: **il contesto contiene il codice che la correzione vera ha cambiato?** Nessun modello genera nulla: si misura
solo cosa finisce nel contesto.

- **Casi**: commit di correzione (parole tipo fix/bug/error nel titolo) che cambiano da 1 a 3 file di codice esistenti
  (test esclusi), al massimo 100 righe. Domanda = messaggio del commit; codice = commit precedente; obiettivo = le
  righe che la correzione ha cambiato.
- **Metodi**, tutti sullo stesso codice e con lo stesso budget (12 000 token stimati):

  | metodo | cosa fa |
  |---|---|
  | `zt` | router completo: lessicale + embedding + energia, span per simbolo, taglio dinamico |
  | `bm25-chunks` | lo stesso router con la sola lista lessicale (BM25 sui simboli): **la baseline onesta** |
  | `bm25-files` | BM25 su file interi, impacchettati finché il budget è pieno |
  | `grep` | file ordinati per quante parole della domanda contengono (solo ordine, niente contesto) |

- **Metriche**: posizione del file giusto (1°, primi 5, recall@10, MRR) e, per i contesti, se il file giusto c'è,
  quante righe da cambiare ci sono, se c'è almeno una riga da cambiare per ogni file ("punto da correggere nel
  contesto"), token usati.

```bash
python examples/benchmark_localize.py --workdir DIR repo1 repo2 ...   # riprende da dove era
python examples/benchmark_localize.py --workdir DIR --report-only
```

## Risultati: 150 casi, 6 repository, 3 linguaggi (18/10/2026)

click, requests, flask, attrs (Python), cobra (Go), ky (TypeScript); 25 casi ciascuna. **Encoder di test**
(`HashingEncoder`): qui non c'era accesso a Hugging Face, quindi la parte con gli embedding è sottostimata; la parte
lessicale è identica. Tabelle complete: [results/localize_hashing_6repos.md](results/localize_hashing_6repos.md).

| metodo | file giusto 1° | punto da correggere nel contesto | righe da cambiare nel contesto | token medi |
|---|---|---|---|---|
| zt | 40% | **61%** | 61% | 8 771 |
| bm25-chunks | 39% | 58% | 58% | 7 714 |
| bm25-files | 19% | 19% | 21% | 12 320 |
| grep | 12% | – | – | – |

- **zt contro file interi**: punto da correggere nel contesto in 65 casi contro 2 (p < 0.0001), con il 29% di token in
  meno. Vale in tutte e 6 le repo e nei 3 linguaggi.
- **zt contro BM25 sui simboli**: quasi pari. 5 casi a 0 sul punto da correggere (p = 0.06), MRR +0.02 (non
  significativo), ma zt usa il 14% di token in più. Unica differenza netta: cobra (Go), MRR +0.15 [+0.06, +0.25].
- **Il valore viene soprattutto dall'indice per simbolo e dall'impacchettamento**, non (con questo encoder) dagli
  embedding o dall'energia.
- **Anche il metodo migliore manca il punto da correggere nel 39% dei casi** (flask e attrs: circa la metà).

## Stessi 150 casi con Qwen3-Embedding-0.6B (RTX 4060 Ti, 18/10/2026)

Le colonne BM25 sono identiche al giro con l'encoder di test: i casi sono gli stessi, cambia solo la parte embedding
di zt.

| | zt, encoder di test | **zt, Qwen3** | bm25-chunks |
|---|---|---|---|
| file giusto 1° | 40% | 38% | 39% |
| file giusto nei primi 5 | 68% | 71% | 67% |
| MRR | 0.535 | 0.523 | 0.516 |
| punto da correggere nel contesto | 61% | 61% | 58% |
| token medi | 8 771 | 8 446 | 7 724 |

zt con Qwen contro bm25-chunks: punto da correggere 6 casi a 1 (p = 0.125), MRR +0.007 [−0.025, +0.038]: **nessuna
differenza dimostrata**. Per repository (MRR zt-Qwen − bm25-chunks): ky +0.072 [+0.013, +0.142] e cobra +0.089
[+0.001, +0.183] a favore di zt; flask −0.046, requests −0.038, attrs −0.022, click −0.013 (nessuno significativo).

| repo | punto nel contesto: test / Qwen / bm25-chunks | MRR: test / Qwen / bm25-chunks |
|---|---|---|
| attrs | 52% / 56% / 44% | 0.371 / 0.334 / 0.356 |
| click | 72% / 72% / 72% | 0.528 / 0.556 / 0.569 |
| cobra | 76% / 76% / 72% | 0.665 / 0.607 / 0.518 |
| flask | 44% / 44% / 44% | 0.405 / 0.331 / 0.378 |
| ky | 64% / 64% / 60% | 0.759 / 0.837 / 0.765 |
| requests | 60% / 56% / 56% | 0.483 / 0.470 / 0.508 |

**Conclusione**: con domande prese dai messaggi di commit (che spesso nominano funzioni e file) gli embedding veri
non aggiungono nulla di misurabile rispetto al BM25 sui simboli. Coerente con la letteratura (BM25 forte sul codice).
Primo indice di click: 56 s su GPU contro 24 minuti su CPU.

**Domanda ancora aperta**: le domande scritte come le scrive una persona (sintomo, in italiano, senza nomi di
funzioni) sono il caso in cui gli embedding multilingue dovrebbero servire. Non è ancora misurato.

## Domande in italiano, senza nomi di codice (`examples/data/loc_cases_it.json`)

Gli stessi 150 casi, con la domanda riscritta come la scriverebbe una persona: in italiano, descrivendo il sintomo, senza
nomi di funzioni, classi o file (restano nomi di prodotti come Windows, fish, JSON, PowerShell). Le riscritture usano
solo il messaggio di commit originale (conservato in `query_en`), mai la correzione.

```bash
python examples/benchmark_localize.py --workdir DIR_NUOVA --cases-file examples/data/loc_cases_it.json click requests flask attrs cobra ky
```

Encoder di test (nessun modello multilingue):

| metodo | file giusto 1° | file giusto nel contesto | punto da correggere nel contesto | token medi |
|---|---|---|---|---|
| zt | 15% | 72% | 28% | 8 317 |
| bm25-chunks | 12% | 62% | 24% | 6 214 |
| bm25-files | 6% | 13% | 10% | 9 248 |
| grep | 5% | – | – | – |

Tutti i metodi crollano rispetto alle domande in inglese (zt 61% → 28%): la ricerca per parole non trova termini
italiani nel codice inglese. zt resta davanti a bm25-chunks (MRR +0.044 [+0.017, +0.074]; punto da correggere 7 casi
a 1, p = 0.07) ma usa il 34% di token in più. Tabelle complete:
[results/localize_hashing_6repos_it.md](results/localize_hashing_6repos_it.md).

Con **Qwen3-Embedding-0.6B** (RTX 4060 Ti, 18/10/2026):

| metodo | file giusto 1° | file giusto nel contesto | righe da cambiare nel contesto | punto da correggere nel contesto | token medi |
|---|---|---|---|---|---|
| **zt, Qwen3** | 14% | 70% | **38%** | **41%** | 8 328 |
| zt, encoder di test | 15% | 72% | 27% | 28% | 8 317 |
| bm25-chunks | 12% | 60% | 24% | 26% | 6 109 |
| bm25-files | 6% | 13% | 11% | 10% | 9 248 |

- zt-Qwen contro bm25-chunks: punto da correggere **22 casi a 0** (p < 0.0001), righe da cambiare +0.14 [+0.09, +0.20].
  Vale in tutte e 6 le repo (attrs 6 a 0, cobra 5 a 0, click 4 a 0, flask 3 a 0, ky 3 a 0, requests 1 a 0).
- Qwen contro encoder di test, stesso router: punto da correggere 41% contro 28%. La posizione del file giusto non
  cambia (MRR 0.280 contro 0.291): Qwen aiuta a scegliere **quali funzioni** del file mettere nel contesto.
- Attenzione: zt usa il 36% di token in più di bm25-chunks; una parte del vantaggio può venire da lì (manca un
  controllo a parità di token per bm25-chunks).
- Le colonne bm25-chunks differiscono di 1–2 punti tra i due giri: i candidati della parte geometrica entrano
  nell'insieme anche quando il loro peso è zero. Effetto piccolo, da correggere.

**Conclusione**: con domande che nominano il codice gli embedding non servono; con domande in linguaggio naturale e in
un'altra lingua (l'uso reale di chi scrive in italiano) Qwen3 porta il punto da correggere nel contesto dal 26% al 41%.
Resta lontano dal 61% delle domande in inglese: conviene che il client (Claude) aggiunga alla domanda le parole
inglesi e i nomi probabili del codice.

## Controllo a parità di token e domande riscritte dal client (encoder di test, 18/10/2026)

Nuovo metodo `bm25-chunks-matched`: BM25 sui simboli senza taglio dinamico, riempito fino ai token che zt ha usato sullo
stesso caso. Corretto anche il difetto per cui candidati con punteggio zero entravano nel contesto di bm25-chunks.

Quattro versioni della stessa domanda: messaggio di commit originale; italiano senza nomi di codice; traduzione inglese
fedele dell'italiano; traduzione + parole chiave e nomi che il client si aspetta. Le due versioni inglesi sono state
scritte da un agente separato che vedeva **solo** la domanda italiana e il nome del progetto (né il codice né il
messaggio originale), come farebbe Claude prima di chiamare `zt_query_context`
(`examples/data/loc_cases_en_translated.json`, `examples/data/loc_cases_en_plus.json`).

Punto da correggere nel contesto (150 casi, encoder di test):

| domanda | zt | bm25-chunks | bm25-chunks a parità di token | bm25-files |
|---|---|---|---|---|
| commit originale (inglese, con nomi) | 61% | 58% | 60% | 19% |
| italiano, senza nomi | 28% | 24% | 26% | 10% |
| traduzione inglese dell'italiano | 56% | 50% | 55% | 22% |
| traduzione + parole chiave del client | 69% | 69% | 71% | 49% |

- **A parità di token il vantaggio di zt (encoder di test) su BM25 sui simboli sparisce**: 2–0, 3–0, 3–1, 0–2 (nessuno
  significativo). Il vantaggio visto prima veniva soprattutto dal contesto più lungo.
- **La forma della domanda conta più di tutto il resto**: tradurre in inglese porta zt dal 28% al 56%; aggiungere le
  parole chiave al 69%, sopra il messaggio originale. Attenzione: per progetti famosi l'agente conosce i nomi interni
  (in un caso ha scritto il nome esatto della funzione); per codice privato la riga "traduzione" è la stima prudente.
### Con Qwen3-Embedding-0.6B (RTX 4060 Ti, 18/10/2026)

| domanda | zt Qwen3 | zt encoder di test | bm25-chunks a parità di token (giro Qwen) | zt-Qwen contro BM25 a parità di token |
|---|---|---|---|---|
| italiano, senza nomi | **41%** | 28% | 27% | **21 casi a 0** (p < 0.0001) |
| traduzione inglese | 57% | 56% | 51% | 9 casi a 1 (p = 0.02) |

- **Il vantaggio di Qwen sulle domande italiane è reale**: regge anche a parità di token (21 a 0, in tutte le repo).
- **Se la domanda viene tradotta in inglese, Qwen aggiunge poco**: 57% contro 56% dell'encoder di test; resta un piccolo
  vantaggio su BM25 a parità di token (9 a 1).
- Quindi: con un client che riscrive la domanda in inglese (Claude via MCP) l'encoder di test basta; Qwen serve quando
  la domanda arriva così com'è in un'altra lingua (es. `zt code "..."` con il modello locale).
- Le colonne bm25-chunks dei giri con Qwen e con l'encoder di test ora coincidono (difetto dei candidati a peso zero
  corretto); `bm25-chunks-matched` differisce perché il budget è quello di zt, diverso tra i due giri.

Tabelle complete: `results/localize_hashing_6repos_{en3,it3,entr3,enplus3}.md`.

## Da fare

- Progetti con codice in italiano (es. un progetto legale privato): lì tradurre la domanda in inglese potrebbe peggiorare; da misurare.
- Capire i casi mancati in flask e attrs (domande vaghe? file enormi? codice in file inattesi?).
- Capire perché zt usa più token di `bm25-chunks` a parità di risultato (il taglio dinamico interviene meno quando i
  punteggi ibridi sono più piatti).
