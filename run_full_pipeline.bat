@echo off
if "%HF_TOKEN%"=="" (
    echo Bitte vorher setzen: set HF_TOKEN=hf_...
    exit /b 1
)
set HUGGING_FACE_HUB_TOKEN=%HF_TOKEN%
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo [%DATE% %TIME%] === SCHRITT 1: Review-Entscheidungen ===
conda run --no-capture-output -n ipase python src\run_review_decisions.py
echo [%DATE% %TIME%] EXIT S1: %ERRORLEVEL%

echo [%DATE% %TIME%] === SCHRITT 2a: 03_similarity EN ===
conda run --no-capture-output -n ipase python src\03_similarity.py --ipase data\ipase_items.json --controls data\control_items.json --out results
echo [%DATE% %TIME%] EXIT 03EN: %ERRORLEVEL%

echo [%DATE% %TIME%] === SCHRITT 2b: 05_features ===
conda run --no-capture-output -n ipase python src\05_features.py
echo [%DATE% %TIME%] EXIT 05: %ERRORLEVEL%

echo [%DATE% %TIME%] === SCHRITT 2c: 06_stats ===
conda run --no-capture-output -n ipase python src\06_stats.py --results results --panss data\panss_scores.csv
echo [%DATE% %TIME%] EXIT 06: %ERRORLEVEL%

echo [%DATE% %TIME%] === ZWISCHENBERICHT ===
conda run --no-capture-output -n ipase python src\write_intermediate_report.py
echo [%DATE% %TIME%] EXIT REPORT: %ERRORLEVEL%

echo [%DATE% %TIME%] === SCHRITT 3: Robustheit + DE (Langzeit) ===
conda run --no-capture-output -n ipase python src\run_schritt3.py
echo [%DATE% %TIME%] EXIT S3: %ERRORLEVEL%

echo [%DATE% %TIME%] Pipeline vollstaendig beendet.
