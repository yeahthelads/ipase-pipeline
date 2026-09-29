"""Liest auf IPU-PC-75 alle Angaben aus, die in 3.4 und 3.5.1 als [Version] markiert sind.

Aufruf im Ordner ipase-pipeline, in der Umgebung, mit der der finale Lauf gerechnet wurde:
    python versionen_auslesen.py
"""
import json
import platform
from collections import Counter
from importlib import metadata
from pathlib import Path

print("Python", platform.python_version())
for pkg in ["scipy", "pandas", "numpy", "statsmodels", "pingouin",
            "openai-whisper", "faster-whisper", "ctranslate2", "pyannote.audio",
            "spacy", "sentence-transformers", "torch", "transformers"]:
    try:
        print(f"{pkg:22s}", metadata.version(pkg))
    except metadata.PackageNotFoundError:
        print(f"{pkg:22s}", "nicht installiert")

try:
    import spacy
    nlp = spacy.load("de_core_news_lg")
    print(f"{'de_core_news_lg':22s}", nlp.meta.get("version"))
except Exception as e:  # noqa: BLE001
    print("de_core_news_lg        nicht ladbar:", e)

# Welches Backend und welches Modell hat der finale Lauf pro Fall verwendet?
backends, models = Counter(), Counter()
for f in Path("data/diarization").glob("*.full_transcript.json"):
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        continue
    meta = d.get("meta", d) if isinstance(d, dict) else {}
    backends[str(meta.get("whisper_backend", "?"))] += 1
    models[str(meta.get("whisper_model", meta.get("model", "?")))] += 1
print("\nwhisper_backend je Fall:", dict(backends))
print("Modell je Fall:        ", dict(models))
