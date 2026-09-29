"""
Zwischenbericht nach Schritt 2: liest results/_stats_*.csv und _overview_similarity.csv,
schreibt Zahlen in _batch_log.md und auf stdout.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent.parent.resolve()
RESULTS_DIR  = PIPELINE_DIR / "results"
DIAR_DIR     = PIPELINE_DIR / "data" / "diarization"
LOG_FILE     = DIAR_DIR / "_batch_log.md"
STATE_FILE   = DIAR_DIR / "_batch_state.json"

EXPECTED_ANCHOR_SOURCE = "data/panss_scores.csv"


def _append_log(text: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists():
        return [], []
    lines = path.read_text("utf-8", errors="replace").splitlines()
    if not lines:
        return [], []
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        if line.strip():
            parts = line.split(",")
            rows.append(dict(zip(header, parts)))
    return header, rows


def main() -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Zwischenbericht ...", flush=True)

    _append_log(f"\n\n---\n## Zwischenbericht Schritt 2 — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # 1. Echte Fälle aus _overview_similarity.csv
    _, ov_rows = _read_csv(RESULTS_DIR / "_overview_similarity.csv")
    real_cases = [r for r in ov_rows if r.get("person_id", "").startswith("Nr_")]
    n_real = len(real_cases)
    n_total = len(ov_rows)

    _append_log(f"**Ähnlichkeits-Outputs**: {n_total} Personen gesamt, davon {n_real} echte Fälle (Nr_XXX)\n")
    print(f"  Similarity-Outputs: {n_total} gesamt, {n_real} echte Fälle")

    # 2. n_ich_saetze < 30
    low_m = []
    for r in real_cases:
        try:
            m = int(float(r.get("n_ich_saetze", "0")))
            if m < 30:
                low_m.append((r["person_id"], m))
        except (ValueError, KeyError):
            pass
    if low_m:
        _append_log(f"**Fälle m < 30 Ich-Sätze** ({len(low_m)}):\n")
        for pid, m in sorted(low_m):
            _append_log(f"- {pid}: m={m}")
        print(f"  Fälle m<30: {low_m}")
    else:
        _append_log("**Fälle m < 30**: keine\n")
        print("  Fälle m<30: keine")

    # 3. _stats_primary_h1h2.csv
    hdr_h1h2, rows_h1h2 = _read_csv(RESULTS_DIR / "_stats_primary_h1h2.csv")
    if rows_h1h2:
        n_ok = sum(1 for r in rows_h1h2 if r.get("n", "").strip())
        ns = [r.get("n", "?") for r in rows_h1h2[:2]]
        anchor_srcs = {r.get("anchor_source", "?") for r in rows_h1h2}
        src_ok = all(s.strip() == EXPECTED_ANCHOR_SOURCE for s in anchor_srcs)
        _append_log(
            f"**_stats_primary_h1h2.csv**: {len(rows_h1h2)} Zeilen, n={ns[0] if ns else '?'}, "
            f"anchor_source={'OK' if src_ok else 'FALSCH: ' + str(anchor_srcs)}\n"
        )
        print(f"  h1h2: {len(rows_h1h2)} Zeilen, n={ns[0] if ns else '?'}, anchor_source={'OK' if src_ok else anchor_srcs}")
    else:
        _append_log("**_stats_primary_h1h2.csv**: FEHLT oder leer\n")
        print("  h1h2: FEHLT")

    # 4. _stats_incremental_validity.csv
    hdr_iv, rows_iv = _read_csv(RESULTS_DIR / "_stats_incremental_validity.csv")
    if rows_iv:
        anchor_srcs_iv = {r.get("anchor_source", "?") for r in rows_iv}
        src_ok_iv = all(s.strip() == EXPECTED_ANCHOR_SOURCE for s in anchor_srcs_iv)
        _append_log(
            f"**_stats_incremental_validity.csv**: {len(rows_iv)} Zeilen, "
            f"anchor_source={'OK' if src_ok_iv else 'FALSCH: ' + str(anchor_srcs_iv)}\n"
        )
        print(f"  inkrementell: {len(rows_iv)} Zeilen, anchor_source={'OK' if src_ok_iv else anchor_srcs_iv}")
    else:
        _append_log("**_stats_incremental_validity.csv**: FEHLT oder leer\n")
        print("  inkrementell: FEHLT")

    # 5. State-Zählung
    try:
        state = json.loads(STATE_FILE.read_text("utf-8"))
        done    = sum(1 for c in state["cases"].values() if c.get("status") in ("phase2_done", "done"))
        pending = sum(1 for c in state["cases"].values() if c.get("status") == "manual_pending")
        cfail   = sum(1 for c in state["cases"].values() if c.get("decision_reason") in
                      ("criterion_C_fail", "review_C_persistent") and c.get("status") == "review-final")
        errors  = sum(1 for c in state["cases"].values() if c.get("status") == "error")
        _append_log(
            f"**State**: Verarbeitet={done} | MANUAL_PENDING={pending} | C_fail={cfail} | ERROR={errors}\n"
        )
        print(f"  State: done={done}, manual_pending={pending}, cfail={cfail}, error={errors}")
    except Exception as e:
        _append_log(f"State-Lesen fehlgeschlagen: {e}\n")

    _append_log(f"*Zwischenbericht: {datetime.now().isoformat()}*\n")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Zwischenbericht geschrieben.")


if __name__ == "__main__":
    main()
