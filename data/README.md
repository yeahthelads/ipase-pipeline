# data/

Dieses Verzeichnis ist absichtlich nicht Teil des öffentlichen Repositories.

Es enthält im lokalen Arbeitsstand u. a. Diarisierungs-/Transkriptdaten, IPASE-/Kontrollitem-Definitionen
und klinische Scores (z. B. PANSS) einzelner Fälle. Da diese Inhalte Patientendaten enthalten oder mit
Fallnummern verknüpft sind, werden sie aus Datenschutzgründen nicht versioniert (siehe `.gitignore`:
`data/*` mit Ausnahme dieser Datei).

Die Pipeline-Skripte in `src/` erwarten unter `data/` u. a. folgende (lokal vorzuhaltende) Dateien:

- `ipase_items.json` / `ipase_items_de.json` — IPASE-Item-Definitionen
- `control_items.json` / `control_items_de.json` — Kontrollitem-Definitionen
- `panss_scores.csv` — klinische Vergleichswerte je Fall
- `diarization/` — Diarisierungs- und Transkriptausgaben je Fall
