"""
Schritt 1: Manuelle Review-Entscheidungen umsetzen.
Phase 2 (Patient-only-Transkript) + Sanity-Check + 02_preprocess.
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent.parent.resolve()
AUDIO_BASE   = Path(os.environ.get("IPASE_AUDIO_BASE", str(PIPELINE_DIR / "audio_input")))
DIAR_DIR     = PIPELINE_DIR / "data" / "diarization"
TRANS_DIR    = PIPELINE_DIR / "data" / "transcripts"
STATE_FILE   = DIAR_DIR / "_batch_state.json"
LOG_FILE     = DIAR_DIR / "_batch_log.md"

# Fallbezogene Zuordnungen liegen lokal in data/case_config.json (per .gitignore
# ausgeschlossen, da an Fallnummern gebunden). Erwartetes Schema:
# {
#   "confirmed": {"<fall_id>": "<speaker_label>"},
#   "confirmed_notes": {"<fall_id>": "<Freitext-Anmerkung>"},
#   "manual_pending": ["<fall_id>"],
#   "c_fail_skip": ["<fall_id>"]
# }
CASE_CONFIG_FILE = PIPELINE_DIR / "data" / "case_config.json"


def _load_case_config() -> dict:
    if CASE_CONFIG_FILE.exists():
        return json.loads(CASE_CONFIG_FILE.read_text("utf-8"))
    return {}


_case_config  = _load_case_config()
CONFIRMED     = _case_config.get("confirmed", {})
CONFIRMED_NOTES = _case_config.get("confirmed_notes", {})
MANUAL_PENDING  = set(_case_config.get("manual_pending", []))
C_FAIL_SKIP     = set(_case_config.get("c_fail_skip", []))

MAX_Q_RATIO_SANITY = 0.35  # Warnung wenn >35% Frageturn im Patienten-Transkript


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text("utf-8"))
    except Exception:
        return {"cases": {}}


def _save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
    os.replace(tmp, STATE_FILE)


def _append_log(text: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _find_audio_file(nr: str) -> Path | None:
    folder = AUDIO_BASE / nr
    if not folder.exists():
        return None
    audio_exts = {".wma", ".wav", ".mp3", ".m4a"}
    files = sorted(
        [f for f in folder.iterdir() if not f.name.startswith(".")],
        key=lambda p: p.name,
    )
    for f in files:
        if re.search(r"screening", f.name, re.IGNORECASE) and f.suffix.lower() in audio_exts:
            return f
    for f in files:
        if f.suffix.lower() in audio_exts:
            return f
    return None


def run_phase2(audio: Path, nr: str, patient: str) -> tuple[int, str]:
    hf = os.environ.get("HF_TOKEN", "")
    args = [
        sys.executable,
        "src/01b_diarize_transcribe.py",
        str(audio),
        "--person-id", nr,
        "--patient", patient,
        "--backend", "faster-whisper",
        "--model", "large-v3",
        "--skip-diarize",
        "--skip-transcribe",
    ]
    env = {**os.environ, "HF_TOKEN": hf, "HUGGING_FACE_HUB_TOKEN": hf,
           "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=600,
                       cwd=str(PIPELINE_DIR))
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def run_preprocess(nr: str) -> tuple[int, str]:
    txt = TRANS_DIR / f"{nr}.txt"
    args = [sys.executable, "src/02_preprocess.py", "--in", str(txt)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=600,
                       cwd=str(PIPELINE_DIR))
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def sanity_check(nr: str) -> tuple[bool, str]:
    txt = TRANS_DIR / f"{nr}.txt"
    if not txt.exists():
        return False, "Transkript fehlt"
    content = txt.read_text("utf-8", errors="replace")
    n_words = len(content.split())
    if n_words < 100:
        return False, f"Zu wenig Woerter: {n_words}"
    sentences = [s.strip() for s in re.split(r"[.!?]+", content) if s.strip()]
    q_ratio = sum(1 for s in sentences if "?" in s) / max(len(sentences), 1)
    warn = (f" WARNUNG q_ratio={q_ratio:.2f}" if q_ratio > MAX_Q_RATIO_SANITY else "")
    return True, f"{n_words} Woerter, {len(sentences)} Saetze, q={q_ratio:.2f}{warn}"


def main() -> None:
    log("=" * 60)
    log("Schritt 1: Review-Entscheidungen")

    hf = os.environ.get("HF_TOKEN", "")
    if not hf:
        sys.exit("FEHLER: HF_TOKEN nicht gesetzt!")

    state = _load_state()

    # Log-Header
    _append_log(
        f"\n\n---\n## Manuelle Review-Session {datetime.now().strftime('%Y-%m-%d')}\n"
    )
    _append_log(
        "**Begruendungsprinzip**: Frageanteil statt Redezeit. "
        "Fälle mit atypischer Redezeitverteilung nur nach manueller Pruefung bestaetigt. "
        "Auto-Regel A (>=60% Redezeit) systematisch falsch bei Negativsymptomatik.\n"
    )
    _append_log(
        f"**Bestaetigt**: {len(CONFIRMED)} | "
        f"**MANUAL_PENDING**: {len(MANUAL_PENDING)} | "
        f"**C_fail (keine Aktion)**: {len(C_FAIL_SKIP)}\n"
    )

    # MANUAL_PENDING markieren
    _append_log("\n### MANUAL_PENDING\n")
    for nr in sorted(MANUAL_PENDING):
        cs = state["cases"].setdefault(nr, {})
        cs["status"] = "manual_pending"
        cs["note"] = "Zurueckgestellt — manuelle Pruefung (atypische Redezeit oder knappes Signal)"
        log(f"[{nr}] → MANUAL_PENDING")
        _append_log(f"- **{nr}**: atypische Redezeitverteilung / knappes Signal")

    # C_fail — keine Aktion, nur loggen
    _append_log("\n### C_fail (keine Aktion, Schritt 5 spaeter)\n")
    for nr in sorted(C_FAIL_SKIP):
        log(f"[{nr}] → C_fail, keine Aktion")
        _append_log(f"- **{nr}**: verschmolzener Sprecher-Cluster, manuell zu entscheiden")

    # Confirmed cases
    _append_log("\n### Verarbeitete Faelle\n")
    _append_log("| Fall | Patient | Woerter | Sanity | Note |")
    _append_log("|------|---------|---------|--------|------|")

    errors_consecutive = 0
    done_new = 0

    for nr in sorted(CONFIRMED, key=lambda n: int(re.search(r"\d+", n).group())):
        patient = CONFIRMED[nr]
        cs = state["cases"].setdefault(nr, {})

        # Bereits vorhanden?
        txt_exists = (TRANS_DIR / f"{nr}.txt").exists()
        sentences_json = PIPELINE_DIR / "data" / "processed" / f"{nr}.sentences.json"

        if txt_exists and cs.get("status") == "phase2_done":
            ok, sanity_msg = sanity_check(nr)
            # preprocess nachholen falls fehlt
            if not sentences_json.exists():
                rc2, _ = run_preprocess(nr)
                if rc2 != 0:
                    log(f"[{nr}] preprocess FEHLER (nachgeholt)")
            n_words = len((TRANS_DIR / f"{nr}.txt").read_text("utf-8", errors="replace").split())
            cs["patient_speaker"] = patient
            cs["confirmed_by"] = "manual_review"
            note = CONFIRMED_NOTES.get(nr, "Phase 2 bereits vorhanden")
            _append_log(f"| {nr} | {patient} | {n_words} | {sanity_msg[:60]} | {note} |")
            log(f"[{nr}] Vorhanden: {sanity_msg}")
            _save_state(state)
            done_new += 1
            continue

        audio = _find_audio_file(nr)
        if audio is None:
            log(f"[{nr}] FEHLER: Audio nicht gefunden")
            cs["status"] = "error"
            cs["note"] = "Audio nicht gefunden"
            _append_log(f"| {nr} | {patient} | — | ERROR | Audio nicht gefunden |")
            errors_consecutive += 1
            _save_state(state)
            if errors_consecutive >= 3:
                log("3 Fehler — Abbruch!")
                break
            continue

        log(f"\n[{nr}] Audio: {audio.name} → Patient={patient}")

        try:
            rc, out = run_phase2(audio, nr, patient)
            if rc != 0:
                log(f"[{nr}] Phase 2 FEHLER (rc={rc}): {out[-400:]}")
                cs["status"] = "error"
                cs["note"] = f"Phase 2 rc={rc}"
                _append_log(f"| {nr} | {patient} | — | ERROR | Phase 2 rc={rc} |")
                errors_consecutive += 1
                _save_state(state)
                if errors_consecutive >= 3:
                    log("3 Fehler — Abbruch!")
                    break
                continue

            ok, sanity_msg = sanity_check(nr)
            log(f"[{nr}] Sanity: {sanity_msg}")

            if not ok:
                cs["status"] = "error"
                cs["note"] = f"Sanity: {sanity_msg}"
                _append_log(f"| {nr} | {patient} | — | FAILED | {sanity_msg} |")
                errors_consecutive += 1
                _save_state(state)
                if errors_consecutive >= 3:
                    break
                continue

            n_words = len((TRANS_DIR / f"{nr}.txt").read_text("utf-8", errors="replace").split())

            rc2, out2 = run_preprocess(nr)
            if rc2 != 0:
                log(f"[{nr}] preprocess FEHLER (rc={rc2}): {out2[-300:]}")
                cs["status"] = "error"
                cs["note"] = f"preprocess rc={rc2}"
                _append_log(f"| {nr} | {patient} | {n_words} | {sanity_msg[:40]} | preprocess FEHLER |")
                errors_consecutive += 1
                _save_state(state)
                if errors_consecutive >= 3:
                    break
                continue

            ich_path = PIPELINE_DIR / "data" / "processed" / f"{nr}.ich_saetze.json"
            ich_n = 0
            if ich_path.exists():
                try:
                    ich_data = json.loads(ich_path.read_text("utf-8"))
                    ich_n = len(ich_data) if isinstance(ich_data, list) else 0
                except Exception:
                    pass

            note = CONFIRMED_NOTES.get(nr, "")
            cs.update({
                "status": "phase2_done",
                "patient_speaker": patient,
                "confirmed_by": "manual_review",
                "words_patient": n_words,
                "ich_saetze": ich_n,
            })
            _append_log(f"| {nr} | {patient} | {n_words} | {sanity_msg[:60]} | {note} |")
            log(f"[{nr}] OK — {n_words} Woerter, {ich_n} Ich-Saetze")
            errors_consecutive = 0
            done_new += 1
            _save_state(state)

        except subprocess.TimeoutExpired:
            log(f"[{nr}] TIMEOUT")
            cs.update({"status": "error", "note": "Timeout Phase 2"})
            _append_log(f"| {nr} | {patient} | — | ERROR | Timeout |")
            errors_consecutive += 1
            _save_state(state)
        except Exception as exc:
            log(f"[{nr}] Ausnahme: {exc}")
            cs.update({"status": "error", "note": str(exc)[:200]})
            _append_log(f"| {nr} | {patient} | — | ERROR | {str(exc)[:80]} |")
            errors_consecutive += 1
            _save_state(state)

    # Statusuebersicht
    done_total = sum(1 for c in state["cases"].values()
                     if c.get("status") in ("phase2_done", "done"))
    pending    = sum(1 for c in state["cases"].values()
                     if c.get("status") == "manual_pending")
    errors_all = sum(1 for c in state["cases"].values()
                     if c.get("status") == "error")

    _append_log(
        f"\n**Status**: Verarbeitet={done_total} | "
        f"MANUAL_PENDING={pending} | C_fail={len(C_FAIL_SKIP)} | ERROR={errors_all}\n"
    )
    log(
        f"\nSchritt 1 fertig: verarbeitet={done_total}, "
        f"MANUAL_PENDING={pending}, C_fail={len(C_FAIL_SKIP)}, error={errors_all}"
    )


if __name__ == "__main__":
    main()
