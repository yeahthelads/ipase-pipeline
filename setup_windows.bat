@echo off
REM ============================================================
REM  IPASE-Pipeline  -  Windows-Setup (Conda)
REM  In der ANACONDA PROMPT ausfuehren, aus dem Ordner
REM  ...\Masterarbeit - Research\ipase_pipeline
REM  Aufruf:  setup_windows.bat
REM ============================================================
setlocal

echo.
echo [1/6] Conda-Umgebung "ipase" anlegen (Python 3.11 + pip, Kanal conda-forge)...
call conda create -y --override-channels -c conda-forge -n ipase python=3.11 pip || goto :error

echo.
echo [2/6] Umgebung aktivieren...
call conda activate ipase || goto :error

echo.
echo [3/6] ffmpeg installieren (Whisper braucht es)...
call conda install -y --override-channels -c conda-forge ffmpeg || goto :error

echo.
echo [4/6] PyTorch mit CUDA installieren (zuerst, damit die CUDA-Version gesetzt ist)...
REM cu124-Wheel laeuft auf der GTX 1650; ab torch 2.6 gibt es keine cu121-Wheels mehr.
pip install torch --index-url https://download.pytorch.org/whl/cu124 || goto :error

echo.
echo [5/6] Pipeline-Dependencies installieren...
pip install -r requirements.txt || goto :error

echo.
echo [6/6] spaCy-Modell de_core_news_lg laden...
python -m spacy download de_core_news_lg || goto :error

echo.
echo ============================================================
echo  Schnelltest:
echo ============================================================
python -c "import torch, whisper, spacy, sentence_transformers; print('torch', torch.__version__); print('CUDA verfuegbar:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'keine')"

echo.
echo Fertig. Wenn oben "CUDA verfuegbar: True" steht und ein GPU-Name
echo erscheint, ist alles bereit. Naechster Schritt: SETUP_WINDOWS.md, Abschnitt 3.
echo.
goto :eof

:error
echo.
echo *** FEHLER in einem Schritt. Bitte die rote/letzte Meldung oben
echo *** komplett kopieren und mir in den Chat schicken.
exit /b 1
