"""
Autonomer 48h-Batch: Diarisierung, Transkription, Sprecherzuweisung und Analyse
für alle Fälle in Audios/Audios/Nr_XXX.

Ausführung:
    conda run -n ipase python src/auto_batch.py

State-Datei:  data/diarization/_batch_state.json
Log:          data/diarization/_batch_log.md
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ── Pfade ────────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent.parent.resolve()
AUDIO_BASE   = Path(os.environ.get("IPASE_AUDIO_BASE", str(PIPELINE_DIR / "audio_input")))
DIAR_DIR     = PIPELINE_DIR / "data" / "diarization"
TRANS_DIR    = PIPELINE_DIR / "data" / "transcripts"
STATE_FILE   = DIAR_DIR / "_batch_state.json"
LOG_FILE     = DIAR_DIR / "_batch_log.md"
PANSS_CSV    = PIPELINE_DIR / "data" / "panss_scores.csv"

# ── Konstanten ────────────────────────────────────────────────────────────────
MIN_PATIENT_TIME_SHARE = 0.60   # Kriterium A: Redezeit-Anteil
MIN_SECOND_SPK_TIME    = 0.05   # Kriterium C: mind. 5% Redezeit für 2. Sprecher
MAX_THIRD_SPK_TIME     = 0.05   # 3. Sprecher ignorierbar wenn <5%
MAX_QUESTION_RATIO_PAT = 0.50   # Heuristik: Patient selten >50% Fragen
MIN_WORDS_PATIENT      = 100    # Sanity-Check Phase 2

INTERVIEW_RE = re.compile(
    r"\b(wie|was|wann|wo|wer|warum|weshalb|inwiefern|können sie|könnten sie|"
    r"haben sie|hatten sie|würden sie|dürfte ich|erzählen sie|schildern sie|"
    r"beschreiben sie|erinnern sie|wissen sie)\b",
    re.IGNORECASE,
)

# Fallbezogene Zuordnungen liegen lokal in data/case_config.json (per .gitignore
# ausgeschlossen, da an Fallnummern gebunden). Erwartetes Schema:
# {
#   "preconfirmed": {"<fall_id>": "<speaker_label>"},
#   "already_done_analysis": ["<fall_id>"],
#   "step2_cases": ["<fall_id>"]
# }
CASE_CONFIG_FILE = PIPELINE_DIR / "data" / "case_config.json"


def _load_case_config() -> dict:
    if CASE_CONFIG_FILE.exists():
        return json.loads(CASE_CONFIG_FILE.read_text("utf-8"))
    return {}


_case_config = _load_case_config()

# Vorbelegte Fälle (manuell bestätigt, schon in Phase 1)
PRECONFIRMED = _case_config.get("preconfirmed", {})

ALREADY_DONE_ANALYSIS = set(_case_config.get("already_done_analysis", []))  # 02-07 komplett

# Schritt-2-Fälle: erzwungene Loudnorm-Re-Diarisierung
STEP2_CASES = set(_case_config.get("step2_cases", []))


# ═══════════════════════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ═══════════════════════════════════════════════════════════════════════════════

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text("utf-8"))
        except Exception:
            pass
    return {"cases": {}, "consecutive_errors": 0}


def _save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
    os.replace(tmp, STATE_FILE)


def _disk_free_gb() -> float:
    import shutil
    return shutil.disk_usage(PIPELINE_DIR).free / 1e9


def _find_audio_file(nr: str) -> Path | None:
    """Screening bevorzugt; sonst frühester Diagnostik-Termin; Duplikate ignorieren."""
    folder = AUDIO_BASE / nr
    if not folder.exists():
        return None
    # Hidden-Dateien (macOS ._*) und System-Dateien ausschließen
    files = sorted(
        [f for f in folder.iterdir() if not f.name.startswith(".")],
        key=lambda p: p.name,
    )
    audio_exts = {".wma", ".wav", ".mp3", ".m4a"}
    # Screening bevorzugt
    for f in files:
        if re.search(r"screening", f.name, re.IGNORECASE) and f.suffix.lower() in audio_exts:
            return f
    # Frühester Diagnostik-Termin (erstes verfügbares)
    for f in files:
        if f.suffix.lower() in audio_exts:
            return f
    return None


def _loudnorm_wav(audio: Path, tmp_dir: Path, nr: str) -> Path:
    """Konvertiert Audio nach WAV mit EBU R128-Lautheitsnormalisierung."""
    out_wav = tmp_dir / f"{nr}_norm.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "16000",
        "-ac", "1",
        str(out_wav),
    ]
    log(f"  ffmpeg loudnorm: {audio.name} → {out_wav.name}")
    rc, out = _run(cmd, timeout=600)
    if rc != 0:
        raise RuntimeError(f"ffmpeg loudnorm fehlgeschlagen (rc={rc}):\n{out[-1000:]}")
    if not out_wav.exists():
        raise RuntimeError(f"ffmpeg erzeugte keine Ausgabedatei: {out_wav}")
    size_mb = out_wav.stat().st_size / 1e6
    log(f"  Loudnorm WAV erzeugt: {out_wav.name} ({size_mb:.1f} MB)")
    return out_wav


def _run(cmd: list[str], env: dict | None = None, timeout: int = 7200) -> tuple[int, str]:
    """Führt einen Befehl aus und gibt (returncode, combined_output) zurück."""
    merged_env = {**os.environ}
    if env:
        merged_env.update(env)
    merged_env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=merged_env,
        timeout=timeout,
        cwd=str(PIPELINE_DIR),
    )
    out = (result.stdout or "") + (result.stderr or "")
    return result.returncode, out


def _conda_python(script_args: list[str], env: dict | None = None, timeout: int = 7200) -> tuple[int, str]:
    # sys.executable = Python im aktiven conda-Env (da auto_batch selbst per conda run gestartet)
    cmd = [sys.executable] + script_args
    return _run(cmd, env=env, timeout=timeout)


# ═══════════════════════════════════════════════════════════════════════════════
# Speaker-Statistiken
# ═══════════════════════════════════════════════════════════════════════════════

def compute_speaker_stats(nr: str) -> dict | None:
    """Berechnet Redezeit, Wortanzahl, Frageanteil, Turn-Länge je Sprecher."""
    diar_path = DIAR_DIR / f"{nr}.diarization.json"
    ft_path   = DIAR_DIR / f"{nr}.full_transcript.json"
    if not diar_path.exists() or not ft_path.exists():
        return None

    # Redezeit aus diarization.json
    diar = json.loads(diar_path.read_text("utf-8"))
    time_per_spk: dict[str, float] = {}
    for seg in diar.get("segments", []):
        spk = seg["speaker"]
        time_per_spk[spk] = time_per_spk.get(spk, 0.0) + (seg["end"] - seg["start"])

    # Wörter/Turns/Fragen aus full_transcript.json
    ft = json.loads(ft_path.read_text("utf-8"))
    words = ft.get("aligned_words", [])

    turns: list[tuple[str, list]] = []
    cur_spk, cur_ws = None, []
    for w in words:
        spk = w.get("speaker") or "?"
        if spk != cur_spk:
            if cur_ws:
                turns.append((cur_spk, cur_ws))
            cur_spk, cur_ws = spk, [w["word"]]
        else:
            cur_ws.append(w["word"])
    if cur_ws:
        turns.append((cur_spk, cur_ws))

    word_stats: dict[str, dict] = {}
    for spk, ws in turns:
        text = " ".join(ws)
        s = word_stats.setdefault(spk, dict(turns=0, words=0, q_turns=0, iv_turns=0))
        s["turns"]    += 1
        s["words"]    += len(ws)
        s["q_turns"]  += int("?" in text)
        s["iv_turns"] += int(bool(INTERVIEW_RE.search(text)))

    # Zusammenführen
    all_spks = set(time_per_spk) | set(word_stats)
    stats: dict[str, dict] = {}
    for spk in all_spks:
        ws = word_stats.get(spk, {})
        stats[spk] = {
            "time_sec":     round(time_per_spk.get(spk, 0.0), 2),
            "words":        ws.get("words", 0),
            "turns":        ws.get("turns", 0),
            "q_turns":      ws.get("q_turns", 0),
            "q_ratio":      round(ws.get("q_turns", 0) / max(ws.get("turns", 1), 1), 3),
            "avg_turn_len": round(ws.get("words", 0) / max(ws.get("turns", 1), 1), 1),
        }
    return stats


def auto_decide_patient(stats: dict) -> tuple[str | None, str, dict]:
    """
    Wendet Regeln A+B+C an. Gibt zurück: (patient_speaker | None, reason, metrics).
    reason: "auto_confirmed" | "review_A" | "review_B" | "review_C" | "review_3spk"
    """
    # Sprecher filtern (ohne "?")
    candidates = {k: v for k, v in stats.items() if k != "?"}
    if not candidates:
        return None, "review_no_speakers", {}

    # 3 Sprecher: prüfe ob 3. <5%
    total_time = sum(v["time_sec"] for v in candidates.values())
    if total_time == 0:
        return None, "review_no_time", {}

    sorted_spks = sorted(candidates, key=lambda s: candidates[s]["time_sec"], reverse=True)
    if len(sorted_spks) > 2:
        third_share = candidates[sorted_spks[2]]["time_sec"] / total_time
        if third_share >= MAX_THIRD_SPK_TIME:
            return None, "review_3spk", {
                "spk": sorted_spks[2],
                "share": round(third_share, 3),
            }
        # 3. Sprecher ignorieren
        sorted_spks = sorted_spks[:2]
        candidates  = {k: candidates[k] for k in sorted_spks}
        total_time  = sum(v["time_sec"] for v in candidates.values())

    if len(sorted_spks) < 2:
        return None, "review_single_speaker", {}

    spk_a, spk_b = sorted_spks[0], sorted_spks[1]  # a dominiert

    # Kriterium C: 2. Sprecher ≥5% Redezeit
    second_share = candidates[spk_b]["time_sec"] / total_time
    if second_share < MIN_SECOND_SPK_TIME:
        return None, "criterion_C_fail", {
            "dominant": spk_a,
            "minor": spk_b,
            "minor_share": round(second_share, 3),
        }

    # Kriterium A: Kandidat Patient = Sprecher mit mehr Redezeit + längere Turns
    # Beide Sprecher einzeln als Kandidat prüfen
    for patient_cand, other_cand in [(spk_a, spk_b), (spk_b, spk_a)]:
        time_share = candidates[patient_cand]["time_sec"] / total_time
        longer_turns = (
            candidates[patient_cand]["avg_turn_len"] > candidates[other_cand]["avg_turn_len"]
        )
        lower_q = candidates[patient_cand]["q_ratio"] < candidates[other_cand]["q_ratio"]

        if time_share >= MIN_PATIENT_TIME_SHARE and longer_turns and lower_q:
            metrics = {
                "patient":         patient_cand,
                "time_share":      round(time_share, 3),
                "avg_turn_patient": candidates[patient_cand]["avg_turn_len"],
                "avg_turn_other":   candidates[other_cand]["avg_turn_len"],
                "q_ratio_patient":  candidates[patient_cand]["q_ratio"],
                "q_ratio_other":    candidates[other_cand]["q_ratio"],
                "second_share":     round(second_share, 3),
            }
            return patient_cand, "auto_confirmed", metrics

    # A oder B nicht erfüllt
    metrics = {
        "spk_a":          spk_a,
        "spk_b":          spk_b,
        "time_share_a":   round(candidates[spk_a]["time_sec"] / total_time, 3),
        "q_ratio_a":      candidates[spk_a]["q_ratio"],
        "q_ratio_b":      candidates[spk_b]["q_ratio"],
        "avg_turn_a":     candidates[spk_a]["avg_turn_len"],
        "avg_turn_b":     candidates[spk_b]["avg_turn_len"],
        "second_share":   round(second_share, 3),
    }
    return None, "review_AB", metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline-Schritte
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase1(audio: Path, nr: str, min_spk: int | None = None,
               max_spk: int | None = None, force: bool = False) -> bool:
    """Führt Phase 1 (Diarisierung + Transkription) aus."""
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        log("FEHLER: HF_TOKEN nicht gesetzt!")
        return False

    args = [
        "src/01b_diarize_transcribe.py",
        str(audio),
        "--person-id", nr,
        "--backend", "faster-whisper",
        "--model", "large-v3",
    ]
    if min_spk is not None:
        args += ["--min-speakers", str(min_spk)]
    if max_spk is not None:
        args += ["--max-speakers", str(max_spk)]
    if force:
        args.append("--force")
    if min_spk is not None or max_spk is not None:
        # Nur Diarisierung neu, Transkript behalten wenn vorhanden
        if (DIAR_DIR / f"{nr}.full_transcript.json").exists():
            args.append("--skip-transcribe")

    env = {"HF_TOKEN": hf_token, "HUGGING_FACE_HUB_TOKEN": hf_token}
    log(f"  Phase 1: {audio.name} (min={min_spk}, max={max_spk}, force={force})")
    rc, out = _conda_python(args, env=env, timeout=10800)
    if rc != 0:
        log(f"  Phase 1 FEHLER (rc={rc}):\n{out[-2000:]}")
        return False
    # Output-Verifikation (conda run propagiert rc nicht immer korrekt)
    speakers_json = DIAR_DIR / f"{nr}.speakers.json"
    if not speakers_json.exists():
        log(f"  Phase 1 FEHLER: speakers.json nicht erzeugt. Ausgabe:\n{out[-1500:]}")
        return False
    log(f"  Phase 1 OK")
    return True


def run_phase2(audio: Path, nr: str, patient: str) -> bool:
    """Führt Phase 2 (Patient-only-Transkript) aus."""
    hf_token = os.environ.get("HF_TOKEN", "")
    args = [
        "src/01b_diarize_transcribe.py",
        str(audio),
        "--person-id", nr,
        "--patient", patient,
        "--backend", "faster-whisper",
        "--model", "large-v3",
        "--skip-diarize",
        "--skip-transcribe",
    ]
    env = {"HF_TOKEN": hf_token, "HUGGING_FACE_HUB_TOKEN": hf_token}
    log(f"  Phase 2: {nr} → {patient}")
    rc, out = _conda_python(args, env=env, timeout=600)
    if rc != 0:
        log(f"  Phase 2 FEHLER (rc={rc}):\n{out[-1000:]}")
        return False
    # Sanity-Check: Transkript vorhanden und >100 Wörter?
    txt = TRANS_DIR / f"{nr}.txt"
    if not txt.exists():
        log(f"  Phase 2 FEHLER: {txt} nicht erzeugt")
        return False
    n_words = len(txt.read_text("utf-8", errors="replace").split())
    if n_words < MIN_WORDS_PATIENT:
        log(f"  Phase 2 WARNUNG: nur {n_words} Wörter im Patienten-Transkript")
    log(f"  Phase 2 OK ({n_words} Wörter, Patient={patient})")
    return True


def run_preprocess(nr: str) -> bool:
    """Führt 02_preprocess.py für einen einzelnen Fall aus."""
    txt = TRANS_DIR / f"{nr}.txt"
    rc, out = _conda_python(
        ["src/02_preprocess.py", "--in", str(txt)], timeout=600
    )
    if rc != 0:
        log(f"  02_preprocess FEHLER (rc={rc}):\n{out[-1000:]}")
        return False
    log(f"  02_preprocess OK")
    return True


def _count_ich_saetze(ich_path: Path) -> int:
    """Zählt Ich-Sätze aus <name>.ich_saetze.json.

    02_preprocess.py schreibt {"summary": ..., "ich_saetze": [...]}, kein
    bloßes Array — ein reiner isinstance(..., list)-Check liefert hier immer 0.
    """
    ich_data = json.loads(ich_path.read_text("utf-8"))
    if isinstance(ich_data, dict) and isinstance(ich_data.get("ich_saetze"), list):
        return len(ich_data["ich_saetze"])
    if isinstance(ich_data, list):
        return len(ich_data)
    return 0


def run_analysis() -> bool:
    """Führt Schritte 03-07 aus."""
    steps = [
        (["src/03_similarity.py"],                      "03_similarity"),
        (["src/04_robustness.py"],                      "04_robustness"),
        (["src/05_features.py"],                        "05_features"),
        (["src/06_stats.py", "--panss", str(PANSS_CSV)], "06_stats"),
        (["src/07_plots.py"],                           "07_plots"),
    ]
    for args, name in steps:
        log(f"  Analyse: {name} ...")
        rc, out = _conda_python(args, timeout=14400)
        if rc != 0:
            log(f"  {name} FEHLER (rc={rc}):\n{out[-2000:]}")
            return False
        log(f"  {name} OK")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Log-Ausgabe
# ═══════════════════════════════════════════════════════════════════════════════

def _append_log(text: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _init_log() -> None:
    if not LOG_FILE.exists():
        LOG_FILE.write_text(
            f"# Auto-Batch-Log\n\nGestartet: {datetime.now().isoformat()}\n\n",
            "utf-8",
        )
    _append_log(f"\n\n---\n## Auto-Batch-Lauf: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")


def _log_case(nr: str, status: str, metrics: dict, note: str = "") -> None:
    _append_log(
        f"### {nr} — {status}\n"
        + (f"**Kennzahlen:** {json.dumps(metrics, ensure_ascii=False)}\n" if metrics else "")
        + (f"**Hinweis:** {note}\n" if note else "")
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Hauptlogik
# ═══════════════════════════════════════════════════════════════════════════════

def process_case(nr: str, state: dict) -> str:
    """
    Verarbeitet einen Fall vollständig. Gibt Status zurück:
    'done' | 'phase2_done' | 'review' | 'error'
    """
    case_state = state["cases"].setdefault(nr, {})

    # Bereits analysiert
    if case_state.get("status") in ("done", "skip"):
        return case_state["status"]

    # Audio-Datei finden
    audio = _find_audio_file(nr)
    if audio is None:
        log(f"[{nr}] FEHLER: Keine Audio-Datei gefunden")
        case_state.update({"status": "error", "note": "Keine Audio-Datei"})
        return "error"

    case_state["audio_file"] = str(audio)
    log(f"[{nr}] Audio: {audio.name}")

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    if case_state.get("status") not in ("phase1_done", "phase2_done", "review"):
        # Phase 1 noch nötig?
        speakers_json = DIAR_DIR / f"{nr}.speakers.json"
        need_p1 = not speakers_json.exists() or case_state.get("status") == "phase1_needed"

        # Manuell vorbestätigte Fälle: Phase 1 ist schon durch
        if nr in PRECONFIRMED and speakers_json.exists():
            need_p1 = False
            case_state["patient_speaker"] = PRECONFIRMED[nr]
            case_state["confirmed_by"] = "manual"

        if need_p1:
            ok = run_phase1(audio, nr)
            if not ok:
                case_state["status"] = "error"
                return "error"
            case_state["phase1_done"] = True

    # ── Speaker-Stats & Entscheidung ──────────────────────────────────────────
    if case_state.get("status") not in ("phase2_done", "review") and "patient_speaker" not in case_state:
        stats = compute_speaker_stats(nr)
        if stats is None:
            log(f"[{nr}] FEHLER: Keine Stats berechenbar")
            case_state.update({"status": "error", "note": "Stats fehlen"})
            return "error"

        patient, reason, metrics = auto_decide_patient(stats)

        if reason == "criterion_C_fail":
            # Re-Diarisierung mit Bereichs-Modus
            log(f"[{nr}] Kriterium C verletzt ({metrics}) → Re-Diarisierung min=2, max=3")
            ok = run_phase1(audio, nr, min_spk=2, max_spk=3, force=True)
            if not ok:
                case_state.update({"status": "error", "note": "Re-Diarisierung fehlgeschlagen"})
                return "error"

            # Stats neu berechnen
            stats = compute_speaker_stats(nr)
            if stats is None:
                case_state.update({"status": "error", "note": "Stats nach Re-Diar fehlen"})
                return "error"
            patient, reason, metrics = auto_decide_patient(stats)

            if reason == "criterion_C_fail":
                # Immer noch verschmolzen → REVIEW
                reason = "review_C_persistent"

        case_state["speaker_stats"] = stats
        case_state["decision_metrics"] = metrics
        case_state["decision_reason"] = reason

        if reason == "auto_confirmed":
            case_state["patient_speaker"] = patient
            case_state["confirmed_by"] = "auto"
            log(f"[{nr}] Auto-Bestätigung: Patient={patient} (Metriken: {metrics})")
            _log_case(nr, f"auto_confirmed: {patient}", metrics)
        else:
            log(f"[{nr}] REVIEW: {reason} — Kennzahlen: {metrics}")
            _log_case(nr, f"REVIEW ({reason})", metrics)
            case_state["status"] = "review"
            return "review"

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if case_state.get("status") != "phase2_done":
        patient = case_state.get("patient_speaker")
        if not patient:
            case_state.update({"status": "review", "note": "Kein Patient-Sprecher"})
            return "review"

        ok = run_phase2(audio, nr, patient)
        if not ok:
            case_state["status"] = "error"
            return "error"

        # Wortanzahl aus Transkript
        txt = TRANS_DIR / f"{nr}.txt"
        if txt.exists():
            case_state["words_patient"] = len(txt.read_text("utf-8", errors="replace").split())

    # ── 02_preprocess ─────────────────────────────────────────────────────────
    processed = PIPELINE_DIR / "data" / "processed" / f"{nr}.sentences.json"
    if not processed.exists():
        ok = run_preprocess(nr)
        if not ok:
            case_state["status"] = "error"
            return "error"
        # Ich-Sätze zählen
        ich_path = PIPELINE_DIR / "data" / "processed" / f"{nr}.ich_saetze.json"
        if ich_path.exists():
            case_state["ich_saetze"] = _count_ich_saetze(ich_path)

    case_state["status"] = "phase2_done"
    _log_case(nr, "phase2_done", {"patient": case_state.get("patient_speaker")},
              f"Wörter: {case_state.get('words_patient', '?')}")
    return "phase2_done"


def process_case_loudnorm(nr: str, state: dict) -> str:
    """
    Loudnorm-Re-Diarisierung: ffmpeg EBU R128 → Phase 1 min=2 max=3 --force.
    Gibt zurück: 'phase2_done' | 'review-final' | 'error'
    """
    case_state = state["cases"].setdefault(nr, {})

    audio = _find_audio_file(nr)
    if audio is None:
        log(f"[{nr}] FEHLER: Keine Audio-Datei")
        case_state.update({"status": "error", "note": "Keine Audio-Datei (loudnorm)"})
        return "error"

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"ipase_{nr}_"))
    try:
        norm_wav = _loudnorm_wav(audio, tmp_dir, nr)

        ok = run_phase1(norm_wav, nr, min_spk=2, max_spk=3, force=True)
        if not ok:
            case_state.update({"status": "error", "note": "Phase 1 loudnorm fehlgeschlagen"})
            return "error"

        stats = compute_speaker_stats(nr)
        if stats is None:
            case_state.update({"status": "error", "note": "Stats nach loudnorm fehlen"})
            return "error"

        patient, reason, metrics = auto_decide_patient(stats)

        case_state["speaker_stats_loudnorm"] = stats
        case_state["decision_metrics_loudnorm"] = metrics
        case_state["decision_reason_loudnorm"] = reason
        case_state["loudnorm_applied"] = True

        if reason == "auto_confirmed":
            case_state["patient_speaker"] = patient
            case_state["confirmed_by"] = "auto_loudnorm"
            log(f"[{nr}] Loudnorm-Erfolg: Patient={patient}")
            _log_case(nr, f"loudnorm_confirmed: {patient}", metrics)

            ok = run_phase2(audio, nr, patient)
            if not ok:
                case_state["status"] = "error"
                return "error"

            txt = TRANS_DIR / f"{nr}.txt"
            if txt.exists():
                case_state["words_patient"] = len(txt.read_text("utf-8", errors="replace").split())

            processed = PIPELINE_DIR / "data" / "processed" / f"{nr}.sentences.json"
            if not processed.exists():
                ok = run_preprocess(nr)
                if not ok:
                    case_state["status"] = "error"
                    return "error"

            ich_path = PIPELINE_DIR / "data" / "processed" / f"{nr}.ich_saetze.json"
            if ich_path.exists():
                case_state["ich_saetze"] = _count_ich_saetze(ich_path)

            case_state["status"] = "phase2_done"
            _log_case(nr, "phase2_done (loudnorm)", {"patient": patient},
                      f"Wörter: {case_state.get('words_patient', '?')}")
            return "phase2_done"

        else:
            log(f"[{nr}] Loudnorm ohne Erfolg: {reason} → REVIEW-FINAL")
            _log_case(nr, f"REVIEW-FINAL (loudnorm, {reason})", metrics)
            case_state["status"] = "review-final"
            case_state["decision_reason"] = reason
            return "review-final"

    finally:
        # Temporäre WAVs löschen
        try:
            for f in tmp_dir.iterdir():
                f.unlink()
            tmp_dir.rmdir()
            log(f"  Temp-Verzeichnis {tmp_dir.name} gelöscht")
        except Exception as e:
            log(f"  Warnung: Temp-Verzeichnis nicht gelöscht: {e}")


def main():
    _init_log()
    log("=" * 70)
    log("Auto-Batch Schritt 2+3 gestartet")
    log(f"Freier Speicher: {_disk_free_gb():.1f} GB")

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        sys.exit("Fehler: HF_TOKEN nicht gesetzt! Abbruch.")

    state = _load_state()

    all_cases = sorted(
        [d.name for d in AUDIO_BASE.iterdir() if d.is_dir() and re.match(r"Nr_\d+$", d.name)],
        key=lambda n: int(re.search(r"\d+", n).group()),
    )
    log(f"Gefundene Fälle: {len(all_cases)}")

    # Vorbestätigte und bereits fertige Fälle in State eintragen
    for nr, patient in PRECONFIRMED.items():
        cs = state["cases"].setdefault(nr, {})
        if "patient_speaker" not in cs:
            cs["patient_speaker"] = patient
            cs["confirmed_by"] = "manual"
    for nr in ALREADY_DONE_ANALYSIS:
        state["cases"].setdefault(nr, {})["status"] = "done"
    _save_state(state)

    # ── Schritt 2: Loudnorm-Re-Diarisierung für spezielle Fälle ─────────────
    log(f"\n{'='*60}")
    step2_sorted = sorted(STEP2_CASES, key=lambda n: int(re.search(r"\d+", n).group()))
    log(f"SCHRITT 2: Loudnorm Re-Diarisierung ({', '.join(step2_sorted) or '—'})")
    for nr in step2_sorted:
        cs = state["cases"].get(nr, {})
        status = cs.get("status", "pending")
        if status in ("done", "phase2_done"):
            log(f"[{nr}] Schritt 2 übersprungen (bereits fertig: {status})")
            continue
        if cs.get("loudnorm_done"):
            log(f"[{nr}] Schritt 2 übersprungen (loudnorm bereits versucht)")
            continue
        if _disk_free_gb() < 10:
            log("KRITISCH: Weniger als 10 GB frei — Batch angehalten!")
            _append_log("\n**ABBRUCH: Zu wenig Speicher (<10 GB)**\n")
            break
        log(f"\n{'─'*60}")
        log(f"[{nr}] Schritt 2: Loudnorm Re-Diarisierung (Status bisher: {status})")
        try:
            result = process_case_loudnorm(nr, state)
            state["cases"][nr]["loudnorm_done"] = True
            _save_state(state)
            log(f"[{nr}] Schritt 2 → {result}")
        except subprocess.TimeoutExpired:
            log(f"[{nr}] TIMEOUT in Schritt 2")
            state["cases"].setdefault(nr, {}).update({"status": "error", "note": "Timeout loudnorm"})
            _save_state(state)
        except Exception as exc:
            log(f"[{nr}] Ausnahme in Schritt 2: {exc}")
            state["cases"].setdefault(nr, {}).update({"status": "error", "note": str(exc)[:200]})
            _save_state(state)

    # ── Schritt 3: Verbleibende untranskribiete Fälle ────────────────────────
    log(f"\n{'='*60}")
    log("SCHRITT 3: Verbleibende Fälle")

    REVIEW_AB_REASONS = {
        "review_AB", "review_3spk", "review_single_speaker",
        "review_no_speakers", "review_no_time",
    }
    consecutive_errors = 0

    for nr in all_cases:
        if nr in STEP2_CASES:
            continue

        cs = state["cases"].setdefault(nr, {})
        status = cs.get("status", "pending")
        reason = cs.get("decision_reason", "")

        if status in ("done", "skip", "review-final"):
            log(f"[{nr}] Übersprungen (Status: {status})")
            continue

        if status == "phase2_done":
            log(f"[{nr}] Übersprungen (phase2_done)")
            continue

        # Transkript vorhanden aber kein phase2_done-Status → als solches markieren
        if (TRANS_DIR / f"{nr}.txt").exists():
            log(f"[{nr}] Transkript vorhanden → phase2_done")
            cs["status"] = "phase2_done"
            _save_state(state)
            continue

        if _disk_free_gb() < 10:
            log("KRITISCH: Weniger als 10 GB frei — Batch angehalten!")
            _append_log("\n**ABBRUCH: Zu wenig Speicher (<10 GB)**\n")
            break

        log(f"\n{'─'*60}")
        log(f"[{nr}] Status: {status}, Entscheidungsgrund: {reason or '(keiner)'}")

        try:
            if status == "review" and reason in REVIEW_AB_REASONS:
                # A/B-Kriterien sind inhaltlich — loudnorm hilft nicht → REVIEW-FINAL
                log(f"[{nr}] REVIEW-FINAL (A/B-Kriterium, loudnorm nicht anwendbar)")
                cs["status"] = "review-final"
                _log_case(nr, f"REVIEW-FINAL ({reason})", cs.get("decision_metrics", {}),
                          "A/B-Kriterien inhaltlich — loudnorm hilft nicht")
                _save_state(state)

            elif status == "review" and reason == "review_C_persistent":
                # C-Kriterium trotz min/max persistent → Loudnorm versuchen
                if cs.get("loudnorm_done"):
                    log(f"[{nr}] Loudnorm bereits versucht → REVIEW-FINAL")
                    cs["status"] = "review-final"
                    _save_state(state)
                    continue
                log(f"[{nr}] C-persistent → Loudnorm Re-Diarisierung")
                result = process_case_loudnorm(nr, state)
                cs["loudnorm_done"] = True
                _save_state(state)
                if result == "error":
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        log("3 aufeinanderfolgende Fehler — Batch angehalten!")
                        break
                else:
                    consecutive_errors = 0

            else:
                # Normaler Lauf: pending, error→retry, unbekannter Status
                result = process_case(nr, state)
                _save_state(state)
                if result == "error":
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        log("3 aufeinanderfolgende Fehler — Batch angehalten!")
                        _append_log("\n**ABBRUCH: 3 aufeinanderfolgende Fehler**\n")
                        break
                else:
                    consecutive_errors = 0

        except subprocess.TimeoutExpired:
            log(f"[{nr}] TIMEOUT — als ERROR markiert")
            cs.update({"status": "error", "note": "Timeout"})
            _save_state(state)
            consecutive_errors += 1
        except Exception as exc:
            log(f"[{nr}] Unbehandelte Ausnahme: {exc}")
            cs.update({"status": "error", "note": str(exc)[:200]})
            _save_state(state)
            consecutive_errors += 1

    log(f"\n{'─'*60}")
    log("Schritt 2+3 abgeschlossen.")

    # ── Abschlussbericht ──────────────────────────────────────────────────────
    _write_final_report(state, all_cases)
    log("\nBatch vollständig beendet.")


def _write_final_report(state: dict, all_cases: list[str]) -> None:
    _append_log("\n\n---\n## Abschlussbericht\n")

    # Tabelle
    header = "| Fall | Status | Aufnahme | Patient-Label | Wörter | Ich-Sätze |"
    sep    = "|------|--------|----------|--------------|--------|-----------|"
    rows   = [header, sep]
    review_final_cases = []
    review_cases = []
    error_cases  = []
    done_cases   = 0
    loudnorm_tried   = []
    loudnorm_success = []

    for nr in all_cases:
        cs = state["cases"].get(nr, {})
        status = cs.get("status", "pending")
        audio  = Path(cs.get("audio_file", "?")).name if cs.get("audio_file") else "?"
        pat    = cs.get("patient_speaker", "—")
        words  = cs.get("words_patient", "—")
        ich    = cs.get("ich_saetze", "—")
        rows.append(f"| {nr} | {status} | {audio} | {pat} | {words} | {ich} |")

        if cs.get("loudnorm_applied"):
            loudnorm_tried.append(nr)
            if cs.get("confirmed_by") == "auto_loudnorm":
                loudnorm_success.append(nr)

        if status == "review-final":
            review_final_cases.append((nr, cs.get("decision_reason", "?"), cs.get("decision_metrics", {})))
        elif status == "review":
            review_cases.append((nr, cs.get("decision_reason", "?"), cs.get("decision_metrics", {})))
        elif status == "error":
            error_cases.append((nr, cs.get("note", "?")))
        elif status in ("done", "phase2_done"):
            done_cases += 1

    _append_log("\n".join(rows))

    # Zusammenfassung
    total_c        = len(all_cases)
    review_final_c = len(review_final_cases)
    review_c       = len(review_cases)
    error_c        = len(error_cases)
    _append_log(
        f"\n**Gesamt:** {total_c} | **Verarbeitet:** {done_cases} | "
        f"**REVIEW-FINAL:** {review_final_c} | **REVIEW:** {review_c} | **ERROR:** {error_c}"
    )

    # Loudnorm-Wirksamkeit (M.4.7)
    _append_log("\n### Loudnorm-Wirksamkeit\n")
    if loudnorm_tried:
        success_rate = len(loudnorm_success) / len(loudnorm_tried) * 100
        _append_log(f"- Loudnorm angewendet: {len(loudnorm_tried)} Fälle")
        _append_log(f"- Davon erfolgreich (auto_confirmed): {len(loudnorm_success)} ({success_rate:.0f}%)")
        _append_log(f"- Erfolge: {', '.join(loudnorm_success) if loudnorm_success else '—'}")
        _append_log(f"- Ohne Erfolg: {', '.join(nr for nr in loudnorm_tried if nr not in loudnorm_success) or '—'}")
    else:
        _append_log("- Loudnorm wurde nicht angewendet.")

    # REVIEW-FINAL-Details
    if review_final_cases:
        _append_log("\n### REVIEW-FINAL (keine automatische Lösung möglich)\n")
        for nr, reason, metrics in review_final_cases:
            _append_log(f"- **{nr}** ({reason}): {json.dumps(metrics, ensure_ascii=False)}")

    # REVIEW-Details (noch offen)
    if review_cases:
        _append_log("\n### REVIEW (noch offen)\n")
        for nr, reason, metrics in review_cases:
            _append_log(f"- **{nr}** ({reason}): {json.dumps(metrics, ensure_ascii=False)}")

    # ERROR-Details
    if error_cases:
        _append_log("\n### ERROR-Fälle\n")
        for nr, note in error_cases:
            _append_log(f"- **{nr}**: {note}")

    _append_log(f"\n*Bericht erstellt: {datetime.now().isoformat()}*\n")


if __name__ == "__main__":
    main()
