@echo off
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo [%TIME%] === 03_similarity ===
conda run --no-capture-output -n ipase python src\03_similarity.py
echo EXIT 03: %ERRORLEVEL%

echo [%TIME%] === 04_robustness (Embeddings, kein LeoLM) ===
conda run --no-capture-output -n ipase python src\04_robustness.py --skip-perplexity
echo EXIT 04: %ERRORLEVEL%

echo [%TIME%] === 05_features ===
conda run --no-capture-output -n ipase python src\05_features.py
echo EXIT 05: %ERRORLEVEL%

echo [%TIME%] === 06_stats ===
conda run --no-capture-output -n ipase python src\06_stats.py --panss data\panss_scores.csv
echo EXIT 06: %ERRORLEVEL%

echo [%TIME%] === 07_plots ===
conda run --no-capture-output -n ipase python src\07_plots.py
echo EXIT 07: %ERRORLEVEL%

echo [%TIME%] Analyse abgeschlossen.
