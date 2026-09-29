"""Sprecher-Diarisierung (pyannote 3.1) + Whisper-Transkription mit Patient-only-Filter.

Zwei-Phasen-Workflow nach M.4.1:

  Phase 1 - Diarisierung & Übersicht (kein --patient angegeben):
      python src/01b_diarize_transcribe.py "Audios/<fall_id>/audio.wma"

      -> Konvertiert zu WAV, diarisiert mit pyannote, transkribiert mit Whisper
        (word_timestamps=True), zeigt Sprecher-Übersicht (Redezeit + Beispielsätze).
      -> Speichert:
            data/diarization/<id>.diarization.json   Rohdiarisat
            data/diarization/<id>.full_transcript.json  Mehrsprecher-Transkript
            data/diarization/<id>.speakers.json      Template (patient_speaker: null)
      -> Kein automatisches Raten! User wählt manuell.

  Phase 2 - Patient-only-Transkript (nach manueller Auswahl):
      python src/01b_diarize_transcribe.py "Audios/<fall_id>/audio.wma" --patient SPEAKER_00

      -> Filtert auf Patient-Sprache, schreibt:
            data/transcripts/<id>.txt        ← weiter zu 02_preprocess
            data/transcripts/<id>.meta.json  ← kompatibel mit 01_transcribe.py
            data/diarization/<id>.speakers.json  (bestätigt)

Voraussetzungen:
    HF_TOKEN-Umgebungsvariable gesetzt ODER `huggingface-cli login` ausgeführt.
    Zwei Modell-Lizenzen auf HuggingFace akzeptiert:
      https://hf.co/pyannote/speaker-diarization-3.1
      https://hf.co/pyannote/segmentation-3.0
"""
import os
# Windows/conda: doppelten OpenMP-Runtime tolerieren. Muss vor torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import math
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path

import torch
import whisper

DIARIZE_MODEL = "pyannote/speaker-diarization-3.1"
SUPPORTED_EXTS = {".m4a", ".mp3", ".wav", ".mp4", ".flac", ".wma", ".aac", ".ogg", ".m4b"}

# Pause (Sekunden) zwischen Patient-Turns, ab der ein Absatz eingefügt wird.
# Hilft spaCy bei der Satzsegmentierung (M.4.3).
TURN_GAP_PARAGRAPH_SEC = 3.0


# ──────────────────────────────────────────────────────────────────────────────
# HF-Token
# ──────────────────────────────────────────────────────────────────────────────

def get_hf_token() -> str:
    """HF-Token aus Umgebungsvariable HF_TOKEN oder huggingface_hub-Cache."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    # huggingface_hub >= 0.17: get_token() statt HfFolder.get_token()
    try:
        from huggingface_hub import get_token as hf_get_token
        cached = hf_get_token()
        if cached:
            return cached
    except Exception:
        pass
    # Ältere API (< 0.17)
    try:
        from huggingface_hub import HfFolder
        cached = HfFolder.get_token()
        if cached:
            return cached
    except Exception:
        pass
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Audio-Konvertierung
# ──────────────────────────────────────────────────────────────────────────────

def _find_ffmpeg() -> str:
    """Sucht ffmpeg im PATH; Fallback: conda-env Library/bin.
    Fügt das Verzeichnis auch zu PATH hinzu, damit Whisper ffmpeg findet."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # Fallback: conda env (Windows) — sys.executable = envs/ipase/python.exe
    conda_ffmpeg = Path(sys.executable).parent / "Library" / "bin" / "ffmpeg.exe"
    if conda_ffmpeg.exists():
        # PATH erweitern, damit Whisper intern ffmpeg findet
        ff_dir = str(conda_ffmpeg.parent)
        if ff_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ff_dir + os.pathsep + os.environ.get("PATH", "")
        return str(conda_ffmpeg)
    return "ffmpeg"


def to_wav(audio_path: Path, tmp_dir: Path) -> Path:
    """Konvertiert beliebige Audio-Datei via ffmpeg -> 16 kHz Mono WAV."""
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    wav_path = tmp_dir / f"{audio_path.stem}_16k.wav"
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-i", str(audio_path),
        "-ar", "16000", "-ac", "1", "-f", "wav",
        str(wav_path), "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg-Fehler bei der WAV-Konvertierung:\n{result.stderr}"
        )
    return wav_path


# ──────────────────────────────────────────────────────────────────────────────
# Diarisierung
# ──────────────────────────────────────────────────────────────────────────────

def _patch_pyannote_compat():
    """Zwei Compat-Patches für pyannote 3.3.x auf torch 2.6 + huggingface_hub >= 0.24:

    A) hf_hub_download: pyannote übergibt 'use_auth_token', das >= 0.24 nicht mehr
       kennt. Patch: umbenennen in 'token' in allen pyannote-Modul-Namespaces.

    B) torch.load weights_only: torch 2.6 setzt weights_only=True als Default.
       pyannote-Checkpoints enthalten pyannote-spezifische Globals (Specifications,
       TorchVersion usw.), die nicht im Default-Allowlist sind. Patch: lightning_fabric's
       _load() explizit auf weights_only=False setzen (pyannote-Quelle ist HF-verifiziert,
       daher vertrauenswürdig; dies entspricht der offiziellen Empfehlung im Fehlertext).
    """
    import sys as _sys
    import huggingface_hub as _hfhub

    # --- Patch A: hf_hub_download use_auth_token -> token ---
    _orig_dl = _hfhub.hf_hub_download

    def _compat_dl(*a, **kw):
        if "use_auth_token" in kw:
            kw["token"] = kw.pop("use_auth_token")
        return _orig_dl(*a, **kw)

    _hfhub.hf_hub_download = _compat_dl
    for _mod in list(_sys.modules.values()):
        if getattr(_mod, "__name__", "").startswith("pyannote."):
            if hasattr(_mod, "hf_hub_download"):
                setattr(_mod, "hf_hub_download", _compat_dl)

    # --- Patch B: torch.load weights_only=False für pyannote-Checkpoints ---
    # pyannote.audio.core.model importiert lightning_fabric._load als pl_load.
    # Das ist eine lokale Bindung, daher patchen wir den Verweis direkt im Modul.
    import torch as _torch
    _orig_tload = _torch.load

    def _compat_tload(f, *a, **kw):
        kw["weights_only"] = False  # pytorch_lightning übergibt True explizit -> überschreiben
        return _orig_tload(f, *a, **kw)

    _torch.load = _compat_tload

    # Falls lightning_fabric bereits importiert ist, auch dessen _load patchen
    try:
        import lightning_fabric.utilities.cloud_io as _lf_io

        def _compat_lf_load(path_or_url, map_location=None):
            return _orig_tload(path_or_url, map_location=map_location,
                               weights_only=False)

        _lf_io._load = _compat_lf_load
    except Exception:
        pass

    # pyannote.audio.core.model.pl_load direkt patchen (lokale Bindung)
    try:
        import pyannote.audio.core.model as _pann_model

        def _compat_pl_load(path, map_location=None):
            return _orig_tload(path, map_location=map_location, weights_only=False)

        if hasattr(_pann_model, "pl_load"):
            _pann_model.pl_load = _compat_pl_load
    except Exception:
        pass


def run_diarization(wav_path: Path, hf_token: str, device: torch.device,
                    num_speakers: int | None = None,
                    min_speakers: int | None = None,
                    max_speakers: int | None = None):
    """Führt pyannote-Diarisierung aus. Gibt pyannote Annotation zurück."""
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        sys.exit(
            "pyannote.audio nicht installiert.\n"
            "-> conda run -n ipase pip install pyannote.audio"
        )

    if not hf_token:
        sys.exit(
            "\nFehler: Kein HuggingFace-Token gefunden.\n"
            "Bitte setze die Umgebungsvariable HF_TOKEN, z. B.:\n"
            "  $env:HF_TOKEN = 'hf_...'\n"
            "oder führe einmalig aus:\n"
            "  conda run -n ipase huggingface-cli login\n\n"
            "Außerdem müssen folgende zwei Modell-Lizenzen auf HuggingFace\n"
            "akzeptiert sein (HF-Account erforderlich):\n"
            "  https://hf.co/pyannote/speaker-diarization-3.1\n"
            "  https://hf.co/pyannote/segmentation-3.0\n"
        )

    # Token als Umgebungsvariable setzen (wird von huggingface_hub automatisch gelesen)
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # Compat-Patches für pyannote 3.3.x auf torch 2.6 + huggingface_hub >= 0.24
    _patch_pyannote_compat()

    print(f"Lade pyannote-Modell ({DIARIZE_MODEL})...")
    pipeline = Pipeline.from_pretrained(DIARIZE_MODEL)

    pipeline.to(device)
    print(f"  Device: {device}")
    print(f"  Diarisiere {wav_path.name}...")

    kwargs = {}
    if min_speakers is not None or max_speakers is not None:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
    elif num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(str(wav_path), **kwargs)
    return diarization


def diarization_to_segments(diarization) -> list[dict]:
    """Konvertiert pyannote Annotation -> serialisierbare Segment-Liste."""
    segs = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segs.append({
            "start": round(float(turn.start), 3),
            "end":   round(float(turn.end),   3),
            "speaker": speaker,
        })
    return sorted(segs, key=lambda x: x["start"])


# ──────────────────────────────────────────────────────────────────────────────
# Wort-Sprecher-Alignment
# ──────────────────────────────────────────────────────────────────────────────

def speaker_at(segments: list[dict], word_start: float, word_end: float) -> str | None:
    """Sprecher mit der größten zeitlichen Überlappung zum Wort [word_start, word_end]."""
    best_speaker = None
    best_overlap = 0.0
    for seg in segments:
        overlap = max(0.0, min(word_end, seg["end"]) - max(word_start, seg["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]
    return best_speaker


def align_words_to_speakers(whisper_result: dict, segments: list[dict]) -> list[dict]:
    """Fügt jedem Whisper-Wort das Sprecher-Label (max-Überlappung) hinzu."""
    aligned = []
    for seg in whisper_result.get("segments", []):
        for w in seg.get("words", []):
            sp = speaker_at(segments, float(w["start"]), float(w["end"]))
            aligned.append({
                "word":        w["word"],
                "start":       round(float(w["start"]), 3),
                "end":         round(float(w["end"]),   3),
                "probability": round(float(w.get("probability", 0.0)), 4),
                "speaker":     sp,
            })
    return aligned


# ──────────────────────────────────────────────────────────────────────────────
# faster-whisper Backend
# ──────────────────────────────────────────────────────────────────────────────

def transcribe_with_faster_whisper(audio_path: Path, model_size: str,
                                   device_str: str) -> dict:
    """faster-whisper (CTranslate2) Transkription mit int8-Quantisierung.

    Gibt ein openai-whisper-kompatibles Dict zurück:
        {"segments": [{"id", "start", "end", "text", "words": [...]}], "language": "de"}
    Damit kann align_words_to_speakers() unverändert verwendet werden.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit(
            "faster-whisper nicht installiert. Bitte ausfuehren:\n"
            "  pip install faster-whisper"
        )

    compute_type = "int8"
    print(f"  Device: {device_str}, compute_type={compute_type}")
    try:
        fw_model = WhisperModel(model_size, device=device_str, compute_type=compute_type)
    except Exception as exc:
        if device_str == "cuda":
            print(f"  CUDA-Fehler ({exc}), Fallback auf CPU...")
            fw_model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        else:
            raise

    # WMA/M4A -> WAV konvertieren; faster-whisper liest den Generator lazy,
    # daher Generator innerhalb des temp-Kontexts materialisieren.
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = to_wav(audio_path, Path(tmp_dir))
        segments_gen, info = fw_model.transcribe(
            str(wav_path), language="de", word_timestamps=True,
            vad_filter=True,
        )
        segments_out = []
        for seg in segments_gen:
            words_out = [
                {
                    "word":        w.word,
                    "start":       w.start,
                    "end":         w.end,
                    "probability": w.probability,
                }
                for w in (seg.words or [])
            ]
            segments_out.append({
                "id":    seg.id,
                "start": seg.start,
                "end":   seg.end,
                "text":  seg.text,
                "words": words_out,
            })

    return {"segments": segments_out, "language": getattr(info, "language", "de")}


# ──────────────────────────────────────────────────────────────────────────────
# Sprecher-Übersicht (M.4.1: nur anzeigen, nie automatisch raten)
# ──────────────────────────────────────────────────────────────────────────────

def build_speaker_overview(segments: list[dict], aligned_words: list[dict]) -> dict:
    """
    Pro Sprecher: Redezeit, Wortanzahl, Turn-Anzahl, 3 Beispiel-Äußerungen.
    Beispiele: erste, mittlere und letzte informative Äußerung (≥ 4 Wörter).
    """
    speakers = sorted(set(s["speaker"] for s in segments))
    overview = {}
    for sp in speakers:
        duration = sum(s["end"] - s["start"] for s in segments if s["speaker"] == sp)
        words_sp = [w for w in aligned_words if w["speaker"] == sp]

        # Turns: zusammenhängende Wortblöcke (Pause > 2 s -> neuer Turn)
        turns: list[list[dict]] = []
        current: list[dict] = []
        for w in words_sp:
            if current and w["start"] - current[-1]["end"] > 2.0:
                turns.append(current)
                current = [w]
            else:
                current.append(w)
        if current:
            turns.append(current)

        # Beispiele: erste, mittlere, letzte Turn mit ≥ 4 Wörtern
        long_turns = [t for t in turns if len(t) >= 4]
        sample_indices = []
        if long_turns:
            sample_indices = list({0, len(long_turns) // 2, len(long_turns) - 1})
        examples = [
            " ".join(w["word"] for w in long_turns[i]).strip()
            for i in sorted(sample_indices)
        ]

        overview[sp] = {
            "duration_sec": round(duration, 1),
            "n_words":      len(words_sp),
            "n_turns":      len(turns),
            "examples":     examples,
        }
    return overview


def print_speaker_overview(overview: dict):
    total_dur = sum(v["duration_sec"] for v in overview.values())
    print()
    print("=" * 68)
    print("SPRECHER-ÜBERSICHT  —  M.4.1: Manuelle Patient:innen-Auswahl")
    print("=" * 68)
    for sp, info in sorted(overview.items()):
        pct = 100 * info["duration_sec"] / total_dur if total_dur > 0 else 0
        print(f"\n  {sp}  |  {info['duration_sec']:.0f}s ({pct:.0f} %)  "
              f"|  {info['n_words']} Wörter  |  {info['n_turns']} Turns")
        for i, ex in enumerate(info["examples"], 1):
            snippet = (ex[:115] + "...") if len(ex) > 118 else ex
            print(f"    Bsp. {i}: {snippet}")
    print("=" * 68)


# ──────────────────────────────────────────────────────────────────────────────
# Patient-only-Transkript aufbauen
# ──────────────────────────────────────────────────────────────────────────────

def patient_only_text(aligned_words: list[dict], patient_speaker: str) -> str:
    """
    Baut aus den Patientenwörtern einen zusammenhängenden Text.
    Pausen > TURN_GAP_PARAGRAPH_SEC -> Absatz (hilft spaCy-Satzsegmentierung).
    """
    pat_words = [w for w in aligned_words if w["speaker"] == patient_speaker]
    if not pat_words:
        return ""
    chunks: list[str] = []
    current: list[str] = [pat_words[0]["word"]]
    for prev_w, w in zip(pat_words, pat_words[1:]):
        gap = w["start"] - prev_w["end"]
        if gap > TURN_GAP_PARAGRAPH_SEC:
            text = " ".join(current).strip()
            if text:
                chunks.append(text)
            current = [w["word"]]
        else:
            current.append(w["word"])
    text = " ".join(current).strip()
    if text:
        chunks.append(text)
    return "\n\n".join(chunks)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diarisierung (pyannote) + Whisper-Transkription mit Patient-Filter (M.4.1)"
    )
    parser.add_argument("audio",
                        help="Audio-Datei (.wma, .m4a, .wav, ...)")
    parser.add_argument("--patient", default=None, metavar="SPEAKER_XX",
                        help="Sprecher-Label der Patient:in (z. B. SPEAKER_00). "
                             "Ohne Angabe: nur Phase 1 (Diarisierung + Übersicht).")
    parser.add_argument("--person-id", default=None,
                        help="Kurzname für Output-Dateien (default: Datei-Stem).")
    parser.add_argument("--model", default="small",
                        help="Whisper-Modellgröße (default: small). "
                             "Mit --backend faster-whisper: large-v3 empfohlen "
                             "(int8, passt in 4 GB VRAM).")
    parser.add_argument("--backend", default="openai-whisper",
                        choices=["openai-whisper", "faster-whisper"],
                        help="Transkriptions-Backend (default: openai-whisper). "
                             "faster-whisper nutzt CTranslate2 + int8-Quantisierung.")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Feste Sprecheranzahl (default: None → pyannote auto). "
                             "Nicht zusammen mit --min-speakers/--max-speakers verwenden.")
    parser.add_argument("--min-speakers", type=int, default=None,
                        help="Minimale Sprecheranzahl für Bereichs-Modus (z.B. 2).")
    parser.add_argument("--max-speakers", type=int, default=None,
                        help="Maximale Sprecheranzahl für Bereichs-Modus (z.B. 3).")
    parser.add_argument("--out-transcripts", default="data/transcripts",
                        help="Ausgabe-Ordner für Transkripte (default: data/transcripts).")
    parser.add_argument("--out-diarization", default="data/diarization",
                        help="Ausgabe-Ordner für Diarisierungs-Daten (default: data/diarization).")
    parser.add_argument("--skip-diarize", action="store_true",
                        help="Vorhandenes Rohdiarisat wiederverwenden.")
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Vorhandenes Volltranskript wiederverwenden.")
    parser.add_argument("--force", action="store_true",
                        help="Alle Zwischenschritte neu berechnen.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        sys.exit(f"Datei nicht gefunden: {audio_path}")
    if audio_path.name.startswith("."):
        sys.exit(f"macOS-Metadaten-Datei übersprungen: {audio_path.name}")
    if audio_path.suffix.lower() not in SUPPORTED_EXTS:
        sys.exit(f"Nicht unterstütztes Format '{audio_path.suffix}'. "
                 f"Unterstützt: {sorted(SUPPORTED_EXTS)}")

    person_id = args.person_id or audio_path.stem
    out_transcripts  = Path(args.out_transcripts)
    out_diarization  = Path(args.out_diarization)
    out_transcripts.mkdir(parents=True, exist_ok=True)
    out_diarization.mkdir(parents=True, exist_ok=True)

    _find_ffmpeg()  # PATH um conda-ffmpeg erweitern, damit Whisper intern ffmpeg findet
    hf_token = get_hf_token()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diarization_json    = out_diarization / f"{person_id}.diarization.json"
    full_transcript_json = out_diarization / f"{person_id}.full_transcript.json"
    speakers_json       = out_diarization / f"{person_id}.speakers.json"

    # ── Schritt 1: Diarisierung ───────────────────────────────────────────────
    # --skip-diarize hat immer Vorrang vor --force (Diarisat ist modellunabhängig)
    need_diarize = not args.skip_diarize and (
        args.force or not diarization_json.exists()
    )

    if need_diarize:
        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f"Konvertiere {audio_path.name} -> 16 kHz WAV...")
            wav_path = to_wav(audio_path, Path(tmp_dir))
            print(f"  -> {wav_path.name} ({wav_path.stat().st_size // 1024} KB)")
            diarization = run_diarization(
                wav_path, hf_token, device,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
        segments = diarization_to_segments(diarization)
        diarization_json.write_text(
            json.dumps({"person_id": person_id, "model": DIARIZE_MODEL,
                        "segments": segments}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"  Rohdiarisat -> {diarization_json.name} "
              f"({len(segments)} Segmente, "
              f"{len(set(s['speaker'] for s in segments))} Sprecher)")
    else:
        if not diarization_json.exists():
            sys.exit(f"--skip-diarize gesetzt, aber kein Diarisat gefunden:\n"
                     f"  {diarization_json}\nErst ohne --skip-diarize laufen lassen.")
        data = json.loads(diarization_json.read_text(encoding="utf-8"))
        segments = data["segments"]
        print(f"Diarisat geladen: {diarization_json.name} "
              f"({len(segments)} Segmente)")

    # ── Schritt 2: Whisper-Transkription ─────────────────────────────────────
    # --skip-transcribe hat Vorrang vor --force
    need_transcribe = not args.skip_transcribe and (
        args.force or not full_transcript_json.exists()
    )

    if need_transcribe:
        if args.backend == "faster-whisper":
            print(f"\nLade faster-whisper ({args.model}, int8)...")
            result = transcribe_with_faster_whisper(audio_path, args.model, str(device))
        else:
            print(f"\nLade Whisper-{args.model}-Modell...")
            model = whisper.load_model(args.model)
            print(f"  Transkribiere {audio_path.name} (word_timestamps=True)...")
            result = model.transcribe(str(audio_path), language="de",
                                      word_timestamps=True, verbose=False)

        aligned_words = align_words_to_speakers(result, segments)
        segs_no_words = [
            {k: v for k, v in s.items() if k != "words"}
            for s in result.get("segments", [])
        ]
        full_transcript_json.write_text(
            json.dumps({
                "person_id":        person_id,
                "whisper_model":    args.model,
                "whisper_backend":  args.backend,
                "aligned_words":    aligned_words,
                "whisper_segments": segs_no_words,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"  Volltranskript -> {full_transcript_json.name} "
              f"({len(aligned_words)} Wörter)")
    else:
        if not full_transcript_json.exists():
            sys.exit(f"--skip-transcribe gesetzt, aber kein Volltranskript gefunden:\n"
                     f"  {full_transcript_json}")
        full_data = json.loads(full_transcript_json.read_text(encoding="utf-8"))
        aligned_words = full_data["aligned_words"]
        print(f"Volltranskript geladen: {full_transcript_json.name} "
              f"({len(aligned_words)} Wörter)")
        if need_diarize:
            # Neue Diarisierung → Speaker-Labels in aligned_words aktualisieren
            aligned_words = [
                {**w, "speaker": speaker_at(segments, w["start"], w["end"])}
                for w in aligned_words
            ]
            full_data["aligned_words"] = aligned_words
            _tmp = full_transcript_json.with_suffix(".tmp")
            _tmp.write_text(
                json.dumps(full_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(_tmp, full_transcript_json)
            print(f"  Speaker-Labels mit neuer Diarisierung neu zugewiesen.")

    # ── Schritt 3: Sprecher-Übersicht anzeigen ────────────────────────────────
    overview = build_speaker_overview(segments, aligned_words)
    print_speaker_overview(overview)

    # ── speakers.json schreiben (Phase-1-Template oder Update) ────────────────
    existing = {}
    if speakers_json.exists():
        try:
            existing = json.loads(speakers_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    known_speakers = sorted(overview.keys())

    if args.patient is None:
        # Phase 1 abgeschlossen — kein automatisches Raten
        speakers_json.write_text(
            json.dumps({
                "person_id":      person_id,
                "speakers":       known_speakers,
                "patient_speaker": existing.get("patient_speaker"),
                "confirmed":      False,
                "note": (
                    f"Bitte manuell patient_speaker eintragen. "
                    f"Verfügbare Labels: {known_speakers}"
                ),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\n-> Phase 1 abgeschlossen.")
        print(f"  Sprecher-Template: {speakers_json}")
        print(f"\n  Nächster Schritt — Patient:in angeben und Phase 2 starten:")
        print(f'  python src/01b_diarize_transcribe.py "{audio_path}" '
              f'--patient SPEAKER_XX --skip-diarize --skip-transcribe')
        return

    # ── Phase 2: Patient-only-Transkript ─────────────────────────────────────
    patient_speaker = args.patient
    if patient_speaker not in known_speakers:
        sys.exit(
            f"\nFehler: '{patient_speaker}' nicht in Diarisat-Sprechern.\n"
            f"Verfügbare Labels: {known_speakers}\n"
        )

    print(f"\nPatient:in = {patient_speaker}  OK")

    pat_text = patient_only_text(aligned_words, patient_speaker)
    if not pat_text.strip():
        print("WARNUNG: Kein Text für Patient-Sprecher gefunden.")

    pat_words   = [w for w in aligned_words if w["speaker"] == patient_speaker]
    n_words_pat = len(pat_words)
    dur_pat     = sum(
        s["end"] - s["start"] for s in segments if s["speaker"] == patient_speaker
    )
    dur_total   = max((s["end"] for s in segments), default=0.0)
    wpm_pat     = (n_words_pat / dur_pat * 60.0) if dur_pat > 0 else float("nan")

    # Transkript schreiben (gleicher Ort/Format wie 01_transcribe.py)
    txt_out = out_transcripts / f"{person_id}.txt"
    txt_out.write_text(pat_text, encoding="utf-8")

    # meta.json (kompatibel mit 01_transcribe.py, erweitert um Diarisierungs-Felder)
    meta = {
        "person_id":           person_id,
        "audio_path":          str(audio_path),
        "duration_sec":        round(dur_pat,   2),
        "duration_sec_total":  round(dur_total, 2),
        "n_words":             n_words_pat,
        "n_words_total":       len(aligned_words),
        "words_per_minute":    round(wpm_pat, 2) if not math.isnan(wpm_pat) else None,
        "whisper_model":       args.model,
        "diarization_model":   DIARIZE_MODEL,
        "patient_speaker":     patient_speaker,
        "n_speakers":          len(known_speakers),
        "language":            "de",
        "pipeline_step":       "01b_diarize_transcribe",
    }
    meta_out = out_transcripts / f"{person_id}.meta.json"
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # speakers.json bestätigen (Qualitätssicherungs-Nachweis für M.4.7)
    speakers_json.write_text(
        json.dumps({
            "person_id":      person_id,
            "speakers":       known_speakers,
            "patient_speaker": patient_speaker,
            "confirmed":      True,
            "speaker_overview": {
                sp: {
                    "duration_sec": v["duration_sec"],
                    "n_words":      v["n_words"],
                    "n_turns":      v["n_turns"],
                }
                for sp, v in overview.items()
            },
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\nPatient-only-Transkript:")
    print(f"  Wörter:   {n_words_pat}")
    print(f"  Redezeit: {dur_pat:.1f} s  ({wpm_pat:.1f} WPM)")
    print(f"  Ausgabe:  {txt_out}")
    print(f"\nVollständige Diarisierungs-Daten: {out_diarization}/")
    print(f"Sprecher-Zuordnung (bestätigt):   {speakers_json.name}")
    print(f"\n-> Bereit für 02_preprocess.py:")
    print(f'  python src/02_preprocess.py --in "{txt_out}"')


if __name__ == "__main__":
    main()
