"""Whisper-Transkription für deutsche Audio-Dateien.

Verwendung:
    python src/01_transcribe.py test_audio/
    python src/01_transcribe.py test_audio/mein_file.m4a

Output:
    Für jedes Audio test_audio/X.m4a wird data/transcripts/X.txt erzeugt.
    Zusätzlich data/transcripts/X.json mit Zeitstempeln pro Segment.

Modellgrößen (Trade-off Geschwindigkeit / Qualität):
    tiny   ~  70 MB, sehr schnell, geringe Qualität — nur für ersten Test
    base   ~ 140 MB, schnell, akzeptabel
    small  ~ 460 MB, mittlere Qualität
    medium ~ 1.5 GB, gute deutsche Performance — DEFAULT
    large-v3 ~ 3 GB, beste Qualität, GPU empfohlen
"""
import os
# Windows/conda: tolerate duplicate OpenMP runtime (MKL + libiomp), sonst bricht
# der Prozess auf manchen Setups kommentarlos bei 0 % ab. Muss vor torch/whisper.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import argparse
from pathlib import Path

import whisper


def transcribe_file(model, audio_path: Path, out_dir: Path) -> None:
    print(f"  → Transkribiere {audio_path.name}...")
    result = model.transcribe(str(audio_path), language="de", verbose=False)

    text_out = out_dir / f"{audio_path.stem}.txt"
    json_out = out_dir / f"{audio_path.stem}.json"
    meta_out = out_dir / f"{audio_path.stem}.meta.json"

    text_out.write_text(result["text"].strip(), encoding="utf-8")
    json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Audio-Dauer aus dem letzten Segment-Ende ableiten. Wenn keine Segmente,
    # versuche es über die Whisper-internen Werte.
    duration_sec = 0.0
    if result.get("segments"):
        duration_sec = float(result["segments"][-1].get("end", 0.0))

    n_words = len(result["text"].split())
    n_segments = len(result["segments"])
    wpm = (n_words / duration_sec * 60.0) if duration_sec > 0 else float("nan")

    meta = {
        "person_id": audio_path.stem,
        "audio_path": str(audio_path),
        "duration_sec": duration_sec,
        "n_words": n_words,
        "n_segments": n_segments,
        "words_per_minute": wpm,
        "whisper_model": getattr(model, "name", None) or getattr(model, "model_name", None),
        "language": "de",
    }
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    ✓ {n_words} Wörter, {n_segments} Segmente, "
          f"Dauer {duration_sec:.1f}s, WPM {wpm:.1f} → {text_out.name}")


def main():
    parser = argparse.ArgumentParser(description="Whisper-Transkription DE")
    parser.add_argument("input", type=str, help="Audio-Datei oder -Ordner")
    parser.add_argument("--model", default="medium", help="Modellgröße (default: medium)")
    parser.add_argument("--out", default="data/transcripts", help="Output-Ordner")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Audiodateien sammeln
    if input_path.is_file():
        files = [input_path]
    else:
        # rglob: findet Audios auch in Unterordnern (z. B. Audios/Nr_XXX/...).
        exts = {".m4a", ".mp3", ".wav", ".mp4", ".flac", ".wma", ".aac", ".ogg", ".m4b"}
        files = sorted(
            p for p in input_path.rglob("*")
            if p.is_file()
            and p.suffix.lower() in exts
            and not p.name.startswith(".")
        )

    if not files:
        print(f"Keine Audio-Dateien in {input_path}")
        sys.exit(1)

    print(f"Lade Whisper-{args.model}-Modell...")
    model = whisper.load_model(args.model)
    print(f"Modell geladen. Transkribiere {len(files)} Datei(en).\n")

    for f in files:
        transcribe_file(model, f, out_dir)

    print(f"\nFertig. Transkripte in {out_dir}/")


if __name__ == "__main__":
    main()
