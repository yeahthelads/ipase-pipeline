"""
Schritt 3: 04_robustness EN + 03_similarity DE + 04_robustness DE.
Wählt Alt-Embedding einmalig (BAAI/bge-m3 → intfloat/multilingual-e5-large Fallback),
verwendet dieselbe Wahl für EN und DE. Schreibt Abschlussbericht.
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PIPELINE_DIR  = Path(__file__).parent.parent.resolve()
DIAR_DIR      = PIPELINE_DIR / "data" / "diarization"
LOG_FILE      = DIAR_DIR / "_batch_log.md"
STATE_FILE    = DIAR_DIR / "_batch_state.json"
CHOICE_FILE   = DIAR_DIR / "_embedding_choice.json"
RESULTS_EN    = PIPELINE_DIR / "results"
RESULTS_DE    = PIPELINE_DIR / "results_de"

PRIMARY_EMBEDDING  = "BAAI/bge-m3"
FALLBACK_EMBEDDING = "intfloat/multilingual-e5-large"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _append_log(text: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _run_py(args: list[str], timeout: int = 86400) -> tuple[int, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(
        [sys.executable] + args,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=timeout, cwd=str(PIPELINE_DIR),
    )
    combined = (r.stdout or "") + (r.stderr or "")
    return r.returncode, combined


def _test_embedding(model_name: str) -> bool:
    """Versucht, das Modell zu laden (nur tokenizer + config, kein full inference)."""
    log(f"  Teste Embedding: {model_name} ...")
    rc, out = _run_py([
        "-c",
        f"from sentence_transformers import SentenceTransformer; "
        f"m = SentenceTransformer('{model_name}'); "
        f"print('OK:', m.get_sentence_embedding_dimension())"
    ], timeout=300)
    if rc == 0 and "OK:" in out:
        log(f"  Embedding {model_name} verfügbar.")
        return True
    log(f"  Embedding {model_name} nicht verfügbar: {out[-300:]}")
    return False


def _choose_embedding() -> str:
    if CHOICE_FILE.exists():
        try:
            choice = json.loads(CHOICE_FILE.read_text("utf-8"))["embedding"]
            log(f"Embedding-Wahl aus Cache: {choice}")
            return choice
        except Exception:
            pass

    if _test_embedding(PRIMARY_EMBEDDING):
        chosen = PRIMARY_EMBEDDING
    else:
        log(f"Fallback auf {FALLBACK_EMBEDDING}")
        _append_log(
            f"\n**Embedding-Fallback**: {PRIMARY_EMBEDDING} nicht verfügbar → "
            f"{FALLBACK_EMBEDDING} für ALLE Läufe (EN + DE).\n"
        )
        chosen = FALLBACK_EMBEDDING

    CHOICE_FILE.write_text(
        json.dumps({"embedding": chosen, "chosen_at": datetime.now().isoformat()},
                   ensure_ascii=False), "utf-8"
    )
    return chosen


def _run_robustness(ipase: str, out_dir: str, embedding: str, pmi: bool = True,
                    skip_embedding: bool = False) -> bool:
    args = [
        "src/04_robustness.py",
        "--ipase", ipase,
        "--out", out_dir,
        "--alt-embedding", embedding,
    ]
    if pmi:
        args.append("--pmi")
    if skip_embedding:
        args.append("--skip-embedding")
    log(f"  04_robustness: ipase={ipase}, out={out_dir}, embedding={embedding}, pmi={pmi}, skip_embedding={skip_embedding}")
    rc, out = _run_py(args, timeout=259200)  # 3 Tage max
    if rc != 0:
        log(f"  04_robustness FEHLER (rc={rc}): {out[-1000:]}")
        _append_log(f"\n**04_robustness FEHLER** (rc={rc}): {out[-500:]}\n")
        return False
    log("  04_robustness OK")
    return True


def _run_similarity(ipase: str, controls: str, out_dir: str) -> bool:
    args = [
        "src/03_similarity.py",
        "--ipase", ipase,
        "--controls", controls,
        "--out", out_dir,
    ]
    log(f"  03_similarity: ipase={ipase}, out={out_dir}")
    rc, out = _run_py(args, timeout=7200)
    if rc != 0:
        log(f"  03_similarity FEHLER (rc={rc}): {out[-500:]}")
        _append_log(f"\n**03_similarity DE FEHLER** (rc={rc}): {out[-300:]}\n")
        return False
    log("  03_similarity OK")
    return True


def _read_csv_col(path: Path, col: str) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text("utf-8", errors="replace").splitlines()
    if not lines:
        return []
    header = lines[0].split(",")
    if col not in header:
        return []
    idx = header.index(col)
    return [line.split(",")[idx] for line in lines[1:] if line.strip()]


def write_final_report() -> None:
    log("Schreibe Abschlussbericht ...")
    _append_log(f"\n\n---\n## Abschlussbericht Schritt 3 — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    try:
        choice = json.loads(CHOICE_FILE.read_text("utf-8"))["embedding"] if CHOICE_FILE.exists() else "?"
        _append_log(f"**Verwendetes Alt-Embedding**: {choice}\n")
    except Exception:
        pass

    # Zählung
    try:
        state = json.loads(STATE_FILE.read_text("utf-8"))
        done    = sum(1 for c in state["cases"].values() if c.get("status") in ("phase2_done", "done"))
        pending = sum(1 for c in state["cases"].values() if c.get("status") == "manual_pending")
        cfail   = sum(1 for c in state["cases"].values()
                      if c.get("status") == "review-final" and
                      c.get("decision_reason") in ("criterion_C_fail", "review_C_persistent"))
        errors  = sum(1 for c in state["cases"].values() if c.get("status") == "error")
        _append_log(
            f"**Finale Zählung**: Verarbeitet={done} | MANUAL_PENDING={pending} | "
            f"C_fail={cfail} | ERROR={errors}\n"
        )

        # Tabelle: Fall | Wörter | Ich-Sätze | m>=30
        _append_log("\n### Fallübersicht\n")
        _append_log("| Fall | Wörter | Ich-Sätze | m>=30 |")
        _append_log("|------|--------|-----------|-------|")
        for nr, cs in sorted(state["cases"].items(),
                              key=lambda x: int(__import__("re").search(r"\d+", x[0]).group())
                              if __import__("re").search(r"\d+", x[0]) else 0):
            if not nr.startswith("Nr_"):
                continue
            st = cs.get("status", "?")
            if st not in ("phase2_done", "done"):
                continue
            w = cs.get("words_patient", "—")
            m = cs.get("ich_saetze", "—")
            m_ok = "ja" if isinstance(m, int) and m >= 30 else ("nein" if isinstance(m, int) else "?")
            _append_log(f"| {nr} | {w} | {m} | {m_ok} |")
    except Exception as e:
        _append_log(f"State-Auswertung fehlgeschlagen: {e}\n")

    # EN vs DE Vergleich
    for label, rdir in [("EN", RESULTS_EN), ("DE", RESULTS_DE)]:
        ov = rdir / "_overview_similarity.csv"
        if not ov.exists():
            _append_log(f"\n**{label}**: _overview_similarity.csv fehlt\n")
            continue
        lines = ov.read_text("utf-8", errors="replace").splitlines()
        header = lines[0].split(",") if lines else []
        rows = [dict(zip(header, l.split(","))) for l in lines[1:] if l.strip()]
        real = [r for r in rows if r.get("person_id", "").startswith("Nr_")]
        def _vals(col):
            try:
                return [float(r[col]) for r in real if r.get(col, "").strip()]
            except Exception:
                return []
        import statistics
        mean_vals  = _vals("mean")
        msm_vals   = _vals("mean_sent_max")
        shr_vals   = _vals("specific_hit_rate")
        dsh_vals   = _vals("drift_share_of_hits")
        def _sd(lst):
            return round(statistics.stdev(lst), 4) if len(lst) > 1 else "n.a."
        def _mn(lst):
            return round(sum(lst)/len(lst), 4) if lst else "n.a."
        _append_log(f"\n### {label}: {len(real)} echte Fälle\n")
        _append_log(f"- mean cosine: M={_mn(mean_vals)}, SD={_sd(mean_vals)}")
        _append_log(f"- mean_sent_max: M={_mn(msm_vals)}, SD={_sd(msm_vals)}")
        _append_log(f"- specific_hit_rate: M={_mn(shr_vals)}, SD={_sd(shr_vals)}")
        _append_log(f"- drift_share_of_hits: M={_mn(dsh_vals)}, SD={_sd(dsh_vals)}")

        # Perplexität + PMI aus _overview_robustness_perplexity.csv
        ppl_ov = rdir / "_overview_robustness_perplexity.csv"
        if ppl_ov.exists():
            ppl_lines = ppl_ov.read_text("utf-8", errors="replace").splitlines()
            ppl_h = ppl_lines[0].split(",") if ppl_lines else []
            ppl_rows = [dict(zip(ppl_h, l.split(","))) for l in ppl_lines[1:] if l.strip()]
            ppl_real = [r for r in ppl_rows if r.get("person_id", "").startswith("Nr_")]
            _append_log(f"- Perplexität verfügbar: {len(ppl_real)} Fälle")
            pmi_col = next((c for c in ppl_h if "pmi" in c.lower()), None)
            if pmi_col:
                pmi_vals = [float(r[pmi_col]) for r in ppl_real if r.get(pmi_col, "").strip()]
                _append_log(f"- PMI ({pmi_col}): M={_mn(pmi_vals)}, SD={_sd(pmi_vals)}")

        # stats h1h2
        h12 = rdir / "_stats_primary_h1h2.csv"
        if h12.exists():
            h12_lines = h12.read_text("utf-8", errors="replace").splitlines()
            h12_h = h12_lines[0].split(",") if h12_lines else []
            h12_rows = [dict(zip(h12_h, l.split(","))) for l in h12_lines[1:] if l.strip()]
            _append_log(f"\n**{label} H1/H2-Ergebnisse** ({len(h12_rows)} Zeilen):\n")
            for r in h12_rows:
                rho = r.get("spearman_rho", "?")
                p   = r.get("spearman_p_uncorrected", "?")
                n   = r.get("n", "?")
                anc = r.get("anchor_label", "?")
                _append_log(f"- {anc}: ρ={rho[:7]}, p={p[:6]}, n={n}")

    _append_log(f"\n*Abschlussbericht: {datetime.now().isoformat()}*\n")
    log("Abschlussbericht geschrieben.")


def main() -> None:
    log("=" * 60)
    log("Schritt 3: Robustheit + DE-Variante")

    hf = os.environ.get("HF_TOKEN", "")
    if not hf:
        log("WARNUNG: HF_TOKEN nicht gesetzt (perplexity model download könnte scheitern)")

    _append_log(f"\n\n---\n## Schritt 3 gestartet — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Embedding einmalig wählen
    embedding = _choose_embedding()
    _append_log(f"**Alt-Embedding**: {embedding}\n")

    # 3a: 04_robustness EN — Embedding bereits fertig (gestern), nur Perplexität+PMI
    emb_en_done = (RESULTS_EN / "_overview_robustness_embedding.csv").exists()
    log("\n=== 3a: 04_robustness EN (perplexity+PMI) ===")
    _append_log(f"\n### 3a: 04_robustness EN ({datetime.now().strftime('%H:%M')})\n")
    if emb_en_done:
        _append_log("Embedding-CSV vorhanden → --skip-embedding\n")
    ok_en = _run_robustness(
        ipase="data/ipase_items.json",
        out_dir="results",
        embedding=embedding,
        pmi=True,
        skip_embedding=emb_en_done,
    )
    _append_log(f"04_robustness EN: {'OK' if ok_en else 'FEHLER'}\n")

    # 3b: 03_similarity DE — bereits fertig, überspringen
    de_sim_done = (RESULTS_DE / "_overview_similarity.csv").exists()
    if de_sim_done:
        log("\n=== 3b: 03_similarity DE — bereits vorhanden, übersprungen ===")
        _append_log(f"\n### 3b: 03_similarity DE — bereits vorhanden ({datetime.now().strftime('%H:%M')})\n")
        ok_de_sim = True
    else:
        log("\n=== 3b: 03_similarity DE ===")
        _append_log(f"\n### 3b: 03_similarity DE ({datetime.now().strftime('%H:%M')})\n")
        ok_de_sim = _run_similarity(
            ipase="data/ipase_items_de.json",
            controls="data/control_items_de.json",
            out_dir="results_de",
        )
        _append_log(f"03_similarity DE: {'OK' if ok_de_sim else 'FEHLER'}\n")

    # 3c: 04_robustness DE — Embedding bereits fertig, nur Perplexität+PMI
    emb_de_done = (RESULTS_DE / "_overview_robustness_embedding.csv").exists()
    log("\n=== 3c: 04_robustness DE (perplexity+PMI) ===")
    _append_log(f"\n### 3c: 04_robustness DE ({datetime.now().strftime('%H:%M')})\n")
    if emb_de_done:
        _append_log("Embedding-CSV vorhanden → --skip-embedding\n")
    ok_de_rob = _run_robustness(
        ipase="data/ipase_items_de.json",
        out_dir="results_de",
        embedding=embedding,
        pmi=True,
        skip_embedding=emb_de_done,
    )
    _append_log(f"04_robustness DE: {'OK' if ok_de_rob else 'FEHLER'}\n")

    # Abschlussbericht
    write_final_report()

    log("\nSchritt 3 vollständig.")


if __name__ == "__main__":
    main()
