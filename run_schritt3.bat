@echo off
if "%HF_TOKEN%"=="" (
    echo Bitte vorher setzen: set HF_TOKEN=hf_...
    exit /b 1
)
set HUGGING_FACE_HUB_TOKEN=%HF_TOKEN%
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo [%DATE% %TIME%] Schritt 3 gestartet >> data\diarization\_schritt3.log
conda run --no-capture-output -n ipase python src\run_schritt3.py >> data\diarization\_schritt3.log 2>&1
echo [%DATE% %TIME%] EXIT: %ERRORLEVEL% >> data\diarization\_schritt3.log
