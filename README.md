# ipase-pipeline

Code zur Masterarbeit „IPASE-Sprachindex": eine Analyse-Pipeline, die aus Interview-Audio
per Diarisierung/Transkription (Whisper), semantischer Ähnlichkeit zu IPASE-Items
(Sentence-Transformers), Perplexity-Scoring und statistischer Auswertung (u. a. mit
pingouin/statsmodels) einen sprachbasierten Index berechnet.

## Setup

Siehe [`SETUP.md`](SETUP.md) (Mac) bzw. [`SETUP_WINDOWS.md`](SETUP_WINDOWS.md) (Windows/Conda).

## Struktur

- `src/` — Pipeline-Skripte (Transkription, Preprocessing, Similarity, Robustheit, Features, Statistik, Plots, Graphen)
- `tests/` — Unit-Tests
- `data/` — lokale Ein-/Ausgabedaten (nicht Teil dieses Repos, siehe [`data/README.md`](data/README.md))

## Datenschutz

Dieses Repository enthält ausschließlich Pipeline-Code. Patientendaten, Transkripte,
Audiodateien und daraus abgeleitete Zwischen-/Ergebnisdateien sind über `.gitignore`
ausgeschlossen und wurden vor Veröffentlichung geprüft.

## Zitation

Wenn Sie diesen Code verwenden, zitieren Sie ihn bitte gemäß [`CITATION.cff`](CITATION.cff).
