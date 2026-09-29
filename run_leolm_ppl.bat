@echo off
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
conda run --no-capture-output -n ipase python src\04_robustness.py > data\diarization\_leolm_ppl_stdout.log 2> data\diarization\_leolm_ppl_stderr.log
