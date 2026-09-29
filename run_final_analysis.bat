@echo off
REM ============================================================================
REM Finaler Analyse-Lauf fuer die MA (2026-07)
REM Ausfuehren in "Anaconda Prompt (miniconda3)" nach: conda activate ipase
REM Aus dem Pipeline-Ordner starten (cd .../ipase_pipeline)
REM
REM Laeuft zweimal komplett durch:
REM   1) EN-Referenz (Original-IPASE + engl. Kontrollen)  -> results\
REM   2) DE-Referenz (Arbeitsuebersetzung + dt. Kontrollen) -> results_de\
REM Beide inkl. PMI (log P(s|IPASE) - log P(s|leer)) und
REM Embedding-Robustheit mit BAAI/bge-m3 (M.4: S5; Fallback e5-large, s.u.)
REM ============================================================================

set ALT_EMB=BAAI/bge-m3
REM Bei Download-/Speicherproblemen stattdessen:
REM set ALT_EMB=intfloat/multilingual-e5-large

echo === Schritt 0: Syntax-Check ===
python -m py_compile src\02_preprocess.py src\03_similarity.py src\04_robustness.py src\05_features.py src\06_stats.py src\07_plots.py
if errorlevel 1 goto :fehler

echo.
echo === LAUF 1: Englische IPASE-Referenz (results\) ===
python src\03_similarity.py --ipase data\ipase_items.json --controls data\control_items.json --out results
if errorlevel 1 goto :fehler
python src\04_robustness.py --ipase data\ipase_items.json --out results --alt-embedding %ALT_EMB% --pmi
if errorlevel 1 goto :fehler
python src\05_features.py
if errorlevel 1 goto :fehler
python src\06_stats.py --results results --panss data\panss_scores.csv
if errorlevel 1 goto :fehler
python src\07_plots.py --results results --out results\plots
if errorlevel 1 goto :fehler

echo.
echo === LAUF 2: Deutsche IPASE-Referenz (results_de\) ===
if not exist results_de mkdir results_de
python src\03_similarity.py --ipase data\ipase_items_de.json --controls data\control_items_de.json --out results_de
if errorlevel 1 goto :fehler
python src\04_robustness.py --ipase data\ipase_items_de.json --out results_de --alt-embedding %ALT_EMB% --pmi
if errorlevel 1 goto :fehler
REM Features sind item-unabhaengig: Overview aus Lauf 1 uebernehmen,
REM damit 06_stats die kombinierte Tabelle bauen kann.
copy /Y results\_overview_features.csv results_de\_overview_features.csv >nul
python src\06_stats.py --results results_de --panss data\panss_scores.csv
if errorlevel 1 goto :fehler
python src\07_plots.py --results results_de --out results_de\plots
if errorlevel 1 goto :fehler

echo.
echo ============================================================
echo FERTIG. Ergebnisse:
echo   results\     = englische IPASE-Referenz (Primaeranalyse)
echo   results_de\  = deutsche Arbeitsuebersetzung (Sensitivitaet E3)
echo Wichtigste Dateien je Ordner: _stats_primary_h1h2.csv,
echo _stats_incremental_validity.csv, _overview_similarity.csv
echo (Spalte anchor_source muss "data/panss_scores.csv" zeigen!)
echo ============================================================
goto :ende

:fehler
echo.
echo !!! ABBRUCH: Ein Schritt ist fehlgeschlagen (siehe Meldung oben). !!!

:ende
pause
