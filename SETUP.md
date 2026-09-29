# Erstsetup auf deinem Mac

Das musst du *einmal* auf deinem Mac manuell machen. Danach übernehme ich.

## 1. Terminal öffnen

`Cmd + Leertaste` → tippe „Terminal" → Enter.

## 2. In den Projektordner navigieren

```bash
cd "<pfad-zum-repo>"
```

## 3. Python prüfen

```bash
python3 --version
```

Sollte `Python 3.10.x` oder höher zeigen. Falls nicht: `brew install python` (braucht Homebrew, https://brew.sh).

## 4. Virtual Environment anlegen

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Erkennbar: dein Prompt zeigt jetzt `(.venv)` vorne.

## 5. ffmpeg installieren (Whisper braucht es)

```bash
brew install ffmpeg
```

(Falls Homebrew fehlt: https://brew.sh/de/ oder sag mir Bescheid.)

## 6. Python-Dependencies installieren

```bash
pip install -r requirements.txt
```

Dauert 2–5 Min, ca. 2 GB werden runtergeladen (vor allem `torch`).

## 7. spaCy-Modell laden

```bash
python -m spacy download de_core_news_lg
```

Nochmal ~ 500 MB.

## 8. Erstes Audio testen

Leg eine kurze deutsche Audio-Datei (m4a, mp3, wav) in `test_audio/`. Dann:

```bash
python src/01_transcribe.py test_audio/ --model small
```

Beim ersten Mal lädt Whisper das Modell runter (small ~ 460 MB). Nächstes Mal sofort startklar.

## Check-Liste

- [ ] Terminal funktioniert
- [ ] `(.venv)` ist im Prompt sichtbar
- [ ] `pip install` lief durch
- [ ] `python -m spacy download de_core_news_lg` lief durch
- [ ] Mindestens eine Audio-Datei in `test_audio/`
- [ ] `python src/01_transcribe.py test_audio/ --model small` läuft und erzeugt `data/transcripts/*.txt`

Wenn an einer Stelle was klemmt, kopier mir die Fehlermeldung in den Chat — ich helfe.
