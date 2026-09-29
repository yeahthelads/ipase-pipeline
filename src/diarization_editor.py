"""Manuelle Diarisierungskorrektur via CSV-Export / -Import.

Export: Erzeugt eine editierbare CSV pro Person — eine Zeile = ein Sprecher-Turn
        mit Zeitstempel, aktuellem Speaker-Label und Anfang des Textes.

Import: Liest die korrigierte CSV ein und überschreibt die Speaker-Labels in
        data/diarization/<person_id>.full_transcript.json (aligned_words).
        Die diarization.json bleibt unberührt.

Verwendung:
    # Export (vor der Korrektur)
    python src/diarization_editor.py --export <fall_id>

    # Import (nach der Korrektur)
    python src/diarization_editor.py --import <fall_id>

    # Alle auf einmal exportieren
    python src/diarization_editor.py --export-all
"""
import argparse
import csv
import json
import sys
from pathlib import Path

DIAR_DIR  = Path("data/diarization")
EDIT_DIR  = Path("data/diarization/corrections")
PREVIEW_WORDS = 20


def _build_turns(aligned_words: list) -> list:
    """Gruppiert aligned_words in zusammenhängende Sprecher-Turns."""
    turns = []
    cur_spk, cur_words, cur_start, cur_end = None, [], 0.0, 0.0
    for w in aligned_words:
        spk = w.get("speaker") or "?"
        if spk != cur_spk:
            if cur_words:
                turns.append({
                    "turn_id":  len(turns),
                    "speaker":  cur_spk,
                    "start":    round(cur_start, 2),
                    "end":      round(cur_end,   2),
                    "preview":  " ".join(cur_words[:PREVIEW_WORDS]),
                    "n_words":  len(cur_words),
                })
            cur_spk   = spk
            cur_words = [w["word"]]
            cur_start = w.get("start", 0.0)
            cur_end   = w.get("end",   cur_start)
        else:
            cur_words.append(w["word"])
            cur_end = w.get("end", cur_end)
    if cur_words:
        turns.append({
            "turn_id":  len(turns),
            "speaker":  cur_spk,
            "start":    round(cur_start, 2),
            "end":      round(cur_end,   2),
            "preview":  " ".join(cur_words[:PREVIEW_WORDS]),
            "n_words":  len(cur_words),
        })
    return turns


def export_person(person_id: str) -> Path:
    src = DIAR_DIR / f"{person_id}.full_transcript.json"
    if not src.exists():
        sys.exit(f"Nicht gefunden: {src}")

    t = json.loads(src.read_text("utf-8"))
    turns = _build_turns(t["aligned_words"])

    EDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = EDIT_DIR / f"{person_id}_corrections.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["turn_id","speaker","start","end","n_words","preview"])
        w.writeheader()
        w.writerows(turns)

    speakers = sorted(set(r["speaker"] for r in turns))
    print(f"Exportiert: {out}")
    print(f"  {len(turns)} Turns, Sprecher: {speakers}")
    print(f"  Spalte 'speaker' korrigieren (z.B. SPEAKER_00 ↔ SPEAKER_01),")
    print(f"  dann: python src/diarization_editor.py --import {person_id}")
    return out


def import_person(person_id: str) -> None:
    csv_path = EDIT_DIR / f"{person_id}_corrections.csv"
    if not csv_path.exists():
        sys.exit(f"Korrektur-CSV nicht gefunden: {csv_path}\nZuerst exportieren.")

    src = DIAR_DIR / f"{person_id}.full_transcript.json"
    t = json.loads(src.read_text("utf-8"))

    # Korrekturen aus CSV laden
    corrections: dict[int, str] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            corrections[int(row["turn_id"])] = row["speaker"].strip()

    # Turns aus aligned_words rekonstruieren → Turn-IDs vergeben
    turns_map: dict[int, list[int]] = {}  # turn_id → Wort-Indizes
    cur_spk, cur_tid = None, -1
    for wi, w in enumerate(t["aligned_words"]):
        spk = w.get("speaker") or "?"
        if spk != cur_spk:
            cur_tid += 1
            cur_spk = spk
        turns_map.setdefault(cur_tid, []).append(wi)

    # Korrekturen anwenden
    changed = 0
    for tid, indices in turns_map.items():
        new_spk = corrections.get(tid)
        if new_spk is None:
            continue
        old_spk = t["aligned_words"][indices[0]].get("speaker") or "?"
        if new_spk != old_spk:
            for wi in indices:
                t["aligned_words"][wi]["speaker"] = new_spk
            changed += 1

    # Atomar zurückschreiben
    tmp = src.with_suffix(".tmp")
    tmp.write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")
    import os; os.replace(tmp, src)

    print(f"Importiert: {person_id} — {changed} Turns geändert.")
    print(f"Weiter mit Phase 2:")
    print(f"  python src/01b_diarize_transcribe.py <audio> --person-id {person_id} "
          f"--patient SPEAKER_XX --skip-diarize --skip-transcribe")


def main():
    p = argparse.ArgumentParser(description="Diarisierungs-Editor: CSV Export/Import")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--export",     metavar="PERSON_ID")
    grp.add_argument("--import",     metavar="PERSON_ID", dest="import_id")
    grp.add_argument("--export-all", action="store_true")
    args = p.parse_args()

    if args.export_all:
        for f in sorted(DIAR_DIR.glob("*.full_transcript.json")):
            pid = f.stem.replace(".full_transcript", "")
            if pid.startswith("Nr_"):
                print()
                export_person(pid)
    elif args.export:
        export_person(args.export)
    else:
        import_person(args.import_id)


if __name__ == "__main__":
    main()
