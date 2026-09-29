# Setup auf Windows (IPU-PC-75, Conda)

Diese Anleitung ersetzt die Mac-spezifische `SETUP.md`. Sie ist für den Ambulanz-Rechner (Threadripper 2990WX, GTX 1650 mit 4 GB VRAM, Windows 11). Ziel: lauffähige Python-Umgebung, damit du früh mit dem Transkribieren starten kannst.

Arbeitsteilung: Das eigentliche Installieren läuft auf deinem Windows-Python, nicht in meiner Sandbox. Du startest die Schritte, bei Fehlern kopierst du mir die Meldung in den Chat.

## 1. Anaconda Prompt öffnen

Start-Menü, „Anaconda Prompt" tippen, öffnen. Wichtig: nicht die normale Eingabeaufforderung, sondern die Anaconda Prompt, sonst ist `conda` nicht bekannt.

## 2. In den Pipeline-Ordner wechseln und Setup starten

```bat
cd "<pfad-zum-repo>"
setup_windows.bat
```

Das Skript legt eine Conda-Umgebung `ipase` (Python 3.11) an, installiert ffmpeg, PyTorch mit CUDA 12.1, alle Pipeline-Dependencies und das spaCy-Modell. Dauer rund 10 bis 20 Minuten, vor allem PyTorch und das spaCy-Modell laden einige hundert MB bis ~2 GB.

Am Ende läuft ein Schnelltest. Du willst dort sehen:

```
CUDA verfuegbar: True
GPU: NVIDIA GeForce GTX 1650
```

Wenn dort `False` steht, sag mir Bescheid, dann ist der Treiber- oder CUDA-Pfad das Thema.

## 3. Erste Transkription (Schnelltest auf der GPU)

Umgebung aktivieren und das Modell `small` auf den vorhandenen Test-Audios laufen lassen. `small` passt sicher in die 4 GB der GTX 1650.

```bat
conda activate ipase
python src\01_transcribe.py test_audio\ --model small
```

Beim ersten Mal lädt Whisper das `small`-Modell (~460 MB) einmalig herunter. Ergebnis: für jede Audiodatei ein `.txt`, `.json` und `.meta.json` in `data\transcripts\`.

## 4. Whisper-Modellgröße und die 4-GB-Grenze (wichtig für die echte Studie)

Whisper schiebt das Modell automatisch auf die GPU, sobald CUDA da ist. Auf der GTX 1650 mit 4 GB gilt grob:

- `small` (~2 GB Bedarf): passt auf die GPU, schnell. Gut für Tests.
- `medium` (Default im Skript-Argument, ~5 GB Bedarf): passt NICHT auf die 4 GB, läuft auf der GPU in einen Speicherfehler.
- `large-v3` (~10 GB Bedarf bei fp16): passt ebenfalls nicht auf die GPU.

Für die finale Studie willst du laut Methodenkapitel `large-v3`. Zwei Wege, das auf diesem Rechner zu fahren:

Variante A, sofort nutzbar: `large-v3` auf der CPU. Der Threadripper hat 32 Kerne, das ist offline gut machbar. GPU per Umgebungsvariable verstecken, dann nimmt Whisper automatisch die CPU:

```bat
set CUDA_VISIBLE_DEVICES=
python src\01_transcribe.py test_audio\ --model large-v3
```

(Das `set` gilt nur für dieses Prompt-Fenster. Neues Fenster, neuer Versuch.)

Variante B, später, schneller und sparsamer: Umstieg auf `faster-whisper` (CTranslate2, int8-Quantisierung). Damit passt `large-v3` in rund 2 bis 3 GB und läuft auf der GPU. Das ist eine kleine Code-Änderung an `01_transcribe.py`, die ich baue, wenn du so weit bist. Sie schließt zugleich das Surrogat S4 aus `PROGRESS.md`.

Empfehlung: jetzt mit `small` testen, damit die Kette steht. Für die echten Aufnahmen entscheiden wir zwischen A und B, sobald die Audios da sind.

## 5. Restliche Pipeline (wenn Transkripte da sind)

```bat
conda activate ipase
python src\02_preprocess.py
python src\03_similarity.py
python src\04_robustness.py
python src\05_features.py
python src\06_stats.py
python src\07_plots.py
```

`06_stats.py` läuft auch als reiner Sanity-Check auf den schon vorhandenen Test-Outputs in `results\`.

## 6. Deutsche Perplexität mit llama-cpp-python (für später, Surrogat S1)

Für die konditionale Perplexität auf LeoLM-7B nutzen wir `llama-cpp-python` mit einem GGUF-Modell und partiellem GPU-Offload (passt damit auf die 4 GB). Installation mit dem vorgebauten CUDA-Wheel, in der aktiven `ipase`-Umgebung:

```bat
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

Das GGUF-Modell (z. B. eine quantisierte LeoLM-7B-Variante, Q4_K_M) lädst du separat herunter und legst es unter `models\` ab. Die Erweiterung in `04_robustness.py`, die LeoLM per llama-cpp-python lädt und die Perplexität berechnet, baue ich dann gezielt. Bis dahin bleibt XGLM-564M als multilingualer Fallback aktiv.

## Kurz-Checkliste

- [ ] Anaconda Prompt geöffnet
- [ ] `setup_windows.bat` durchgelaufen, Schnelltest zeigt `CUDA verfuegbar: True`
- [ ] `python src\01_transcribe.py test_audio\ --model small` erzeugt Transkripte
- [ ] (später) Entscheidung large-v3 auf CPU oder faster-whisper
- [ ] (später) llama-cpp-python plus GGUF-LeoLM für die Perplexität
