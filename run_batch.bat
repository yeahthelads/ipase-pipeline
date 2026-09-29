@echo off
if "%HF_TOKEN%"=="" (
    echo Bitte vorher setzen: set HF_TOKEN=hf_...
    exit /b 1
)
set HUGGING_FACE_HUB_TOKEN=%HF_TOKEN%
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
conda run --no-capture-output -n ipase python src\auto_batch.py > data\diarization\_auto_batch_stdout.log 2> data\diarization\_auto_batch_stderr.log
