"""Auswertung der graphbasierten Connectedness (09_graph.py) gegen klinische Anker.

Ergaenzung zu 06_stats.py, analog zu dessen combined_table()/panss_correlations()-
Logik, aber als eigenes Skript, um 06_stats.py (getestet) nicht anzufassen.
Benjamini-Hochberg und die Bootstrap-CI-Funktion werden aus 06_stats.py per
importlib geladen und 1:1 wiederverwendet (Modulname beginnt mit einer Ziffer,
daher kein normales "import 06_stats" moeglich).

Primaerhypothese (vorab festgehalten, vor Sichtung der Ergebnisse):
  Hoehere Negativsymptomatik (SANS gesamt, PANSS-N) geht einher mit
    - niedrigerem LCC_mean  (kleinere groesste Komponente -> weniger Konnektivitaet)
    - niedrigerem LSC_mean  (kleinere groesste starke Komponente)
    - hoeherem   ATD_mean   (mehr Wortwiederholungen/staerkere lokale Bindung)
  Sechs Tests (3 Masse x 2 Anker), Benjamini-Hochberg ueber alle sechs.
  PANSS-P2 und SAPS-FTD: Bodeneffekte (siehe unten), rein deskriptiv, kein Test.

Liest:
  - results/_overview_graph.csv          (Hauptlauf, extended, --save-windows)
  - results_lemma/_overview_graph.csv    (Lemma-Sensitivitaet)
  - data/panss_scores.csv
  - results/_overview_features.csv       (fuer die min_coherence-Robustheitspruefung)

Schreibt (alle nach results/):
  - _stats_graph_primary.csv         6 Tests, Hauptstichprobe, BH-korrigiert
  - _stats_graph_sensitivity.csv     6 Tests, ohne die 7 Diarisierungs-Vorbehalt-Faelle
  - _stats_graph_lemma.csv           6 Tests, Lemma-Tokenisierung
  - _stats_graph_floor_anchors.csv   PANSS-P2/SAPS-FTD, deskriptiv, kein Test
  - _stats_graph_exploratory.csv     13 Masse x 4 Anker, explorativ
  - _stats_graph_length_confound.csv *_mean vs. *_full gegen n_words
  - _overview_coherence_robust.csv   min_coherence vs. min_coherence_p10 (Aufgabe 3a)

Verwendung:
    python src/10_graph_analysis.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SRC_DIR = Path(__file__).parent
ROOT = SRC_DIR.parent

PRIMARY_MEASURES = ["LCC_mean", "LSC_mean", "ATD_mean"]
PRIMARY_ANCHORS = ["sans_total", "panss_negative"]
FLOOR_ANCHORS = ["panss_p2", "saps_thought_disorder"]
ALL_MEASURES = ["N", "E", "PE", "RE", "ATD", "LCC", "LSC",
                "L1", "L2", "L3", "density", "ASP", "CC"]
ALL_ANCHORS = PRIMARY_ANCHORS + FLOOR_ANCHORS

# Richtungserwartung, vorab festgehalten (siehe Docstring oben). +1 = positiver,
# -1 = negativer Zusammenhang mit hoeherer Negativsymptomatik erwartet.
EXPECTED_DIRECTION = {"LCC_mean": -1, "LSC_mean": -1, "ATD_mean": 1}

# Fallbezogene Liste liegt lokal in data/case_config.json (per .gitignore
# ausgeschlossen, da an Fallnummern gebunden). Erwartetes Schema:
# {"sensitivity_exclude_ids": ["<fall_id>", ...]}
CASE_CONFIG_FILE = ROOT / "data" / "case_config.json"


def _load_case_config() -> dict:
    if CASE_CONFIG_FILE.exists():
        return json.loads(CASE_CONFIG_FILE.read_text("utf-8"))
    return {}


SENSITIVITY_EXCLUDE_IDS = set(_load_case_config().get("sensitivity_exclude_ids", []))

N_BOOT = 10_000
BOOT_SEED = 42


# ------------------------------------------------------- 06_stats.py wiederverwenden

def _load_stats06():
    path = SRC_DIR / "06_stats.py"
    spec = importlib.util.spec_from_file_location("ipase_06_stats", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stats06 = _load_stats06()
benjamini_hochberg = _stats06.benjamini_hochberg
bootstrap_spearman_ci = _stats06.bootstrap_spearman_ci
is_real_case = _stats06.is_real_case


# --------------------------------------------------------------------- Partialkorrelation

def partial_spearman(x, y, cov):
    """Echte Partialkorrelation auf Raengen, Ein-Kovariaten-Fall in geschlossener
    Form: partial_r = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2)) auf den
    Raengen von x, y, cov. Aequivalent zur OLS-Residualisierung beider Variablen
    an cov mit anschliessender Korrelation der Residuen, aber ohne lstsq pro
    Bootstrap-Ziehung -- wichtig bei 10 000 Wiederholungen.

    Bewusst keine Semipartialkorrelation (wie die semipartial()-Hilfsfunktion in
    06_stats.py::incremental_validity, die nur x residualisiert) -- hier ist nach
    Vorgabe eine vollstaendige Partialkorrelation verlangt (Kontrolle von n_words
    auf beiden Seiten der Korrelation).
    """
    xr, yr, cr = stats.rankdata(x), stats.rankdata(y), stats.rankdata(cov)
    r_xy = np.corrcoef(xr, yr)[0, 1]
    r_xz = np.corrcoef(xr, cr)[0, 1]
    r_yz = np.corrcoef(yr, cr)[0, 1]
    denom = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return float((r_xy - r_xz * r_yz) / denom)


def partial_spearman_p(x, y, cov):
    """p-Wert der Partialkorrelation via pingouin (Spearman, ein Kovariat)."""
    import pingouin as pg
    df = pd.DataFrame({"x": x, "y": y, "cov": cov})
    res = pg.partial_corr(data=df, x="x", y="y", covar="cov", method="spearman")
    return float(res["r"].iloc[0]), float(res["p_val"].iloc[0])


def bootstrap_partial_spearman_ci(x, y, cov, n_boot=N_BOOT, alpha=0.05, seed=BOOT_SEED):
    if len(x) < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb, cb = x[idx], y[idx], cov[idx]
        if np.std(xb) == 0 or np.std(yb) == 0 or np.std(cb) == 0:
            vals[i] = np.nan
            continue
        vals[i] = partial_spearman(xb, yb, cb)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    point = partial_spearman(x, y, cov)
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return point, lo, hi


# ------------------------------------------------------------------------- Daten laden

def load_merged(graph_csv: Path, panss_csv: Path) -> pd.DataFrame:
    graph = pd.read_csv(graph_csv)
    panss = pd.read_csv(panss_csv)
    graph = graph[graph["person_id"].apply(is_real_case)].copy()
    merged = graph.merge(panss, on="person_id", how="inner")
    return merged


# --------------------------------------------------------------------- Testfamilien

def run_family(df: pd.DataFrame, measures, anchors, apply_bh: bool,
               with_partial: bool, expected_dir: dict | None = None) -> list[dict]:
    rows = []
    p_values = []
    for measure in measures:
        for anchor in anchors:
            sub = df[[measure, anchor, "n_words"]].dropna()
            x = sub[measure].to_numpy(dtype=float)
            y = sub[anchor].to_numpy(dtype=float)
            n = len(sub)
            rho, ci_lo, ci_hi = bootstrap_spearman_ci(x, y, n_boot=N_BOOT, seed=BOOT_SEED)
            _, p_raw = stats.spearmanr(x, y)
            row = {
                "measure": measure, "anchor": anchor, "n": n,
                "spearman_rho": rho, "ci95_lower": ci_lo, "ci95_upper": ci_hi,
                "spearman_p_uncorrected": float(p_raw),
            }
            if expected_dir is not None and measure in expected_dir:
                exp = expected_dir[measure]
                row["direction_expected"] = "negativ" if exp < 0 else "positiv"
                row["direction_matches_hypothesis"] = bool(
                    np.sign(rho) == exp) if not np.isnan(rho) else None
            if with_partial:
                cov = sub["n_words"].to_numpy(dtype=float)
                p_rho, p_p = partial_spearman_p(x, y, cov)
                p_lo, p_hi = bootstrap_partial_spearman_ci(x, y, cov)[1:]
                row.update({
                    "partial_rho_n_words": p_rho,
                    "partial_p_uncorrected": p_p,
                    "partial_ci95_lower": p_lo,
                    "partial_ci95_upper": p_hi,
                })
            rows.append(row)
            p_values.append(float(p_raw))
    if apply_bh:
        adj, rejected = benjamini_hochberg(p_values, alpha=0.05)
        for row, a, r in zip(rows, adj, rejected):
            row["p_bh_adjusted"] = a
            row["significant_bh_05"] = bool(r)
    return rows


def run_floor_table(df: pd.DataFrame) -> list[dict]:
    """PANSS-P2 / SAPS-FTD: rein deskriptiv, bewusst kein p-Wert/kein Test."""
    rows = []
    for anchor in FLOOR_ANCHORS:
        floor_val = df[anchor].min()
        n_at_floor = int((df[anchor] == floor_val).sum())
        n_total = int(df[anchor].notna().sum())
        for measure in PRIMARY_MEASURES:
            sub = df[[measure, anchor, "n_words"]].dropna()
            x = sub[measure].to_numpy(dtype=float)
            y = sub[anchor].to_numpy(dtype=float)
            rho, ci_lo, ci_hi = bootstrap_spearman_ci(x, y, n_boot=N_BOOT, seed=BOOT_SEED)
            cov = sub["n_words"].to_numpy(dtype=float)
            p_rho, _ = partial_spearman_p(x, y, cov)
            p_lo, p_hi = bootstrap_partial_spearman_ci(x, y, cov)[1:]
            rows.append({
                "measure": measure, "anchor": anchor, "n": len(sub),
                "floor_value": float(floor_val),
                "n_at_floor": n_at_floor, "n_total": n_total,
                "share_at_floor": round(n_at_floor / n_total, 3) if n_total else float("nan"),
                "spearman_rho_descriptive": rho,
                "ci95_lower": ci_lo, "ci95_upper": ci_hi,
                "partial_rho_n_words_descriptive": p_rho,
                "partial_ci95_lower": p_lo, "partial_ci95_upper": p_hi,
                "note": "Bodeneffekt -- deskriptiv, kein Hypothesentest, kein p-Wert berichtet.",
            })
    return rows


def run_exploratory_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    for measure in ALL_MEASURES:
        col = f"{measure}_mean"
        if col not in df.columns:
            continue
        for anchor in ALL_ANCHORS:
            sub = df[[col, anchor]].dropna()
            x = sub[col].to_numpy(dtype=float)
            y = sub[anchor].to_numpy(dtype=float)
            rho, ci_lo, ci_hi = bootstrap_spearman_ci(x, y, n_boot=N_BOOT, seed=BOOT_SEED)
            _, p_raw = stats.spearmanr(x, y)
            rows.append({
                "measure": col, "anchor": anchor, "n": len(sub),
                "spearman_rho": rho, "ci95_lower": ci_lo, "ci95_upper": ci_hi,
                "spearman_p_uncorrected": float(p_raw),
                "note": "explorativ, unadjustiert -- nicht fuer Signifikanzaussagen verwenden",
            })
    return rows


def run_length_confound_table(df: pd.DataFrame) -> list[dict]:
    """Vergleich *_mean (Fenster-gemittelt) vs. *_full (Gesamttranskript) gegen n_words."""
    rows = []
    for measure in PRIMARY_MEASURES + ["PE_mean"]:
        base = measure.replace("_mean", "")
        full_col = f"{base}_full"
        for variant, col in [("windowed (*_mean)", measure), ("full transcript (*_full)", full_col)]:
            if col not in df.columns:
                continue
            sub = df[[col, "n_words"]].dropna()
            rho, p = stats.spearmanr(sub[col], sub["n_words"])
            rows.append({
                "measure": base, "variant": variant, "column": col, "n": len(sub),
                "spearman_rho_vs_n_words": float(rho), "spearman_p": float(p),
            })
    return rows


def run_coherence_robustness(features_csv: Path) -> tuple[list[dict], dict]:
    feat = pd.read_csv(features_csv)
    feat = feat[feat["person_id"].apply(is_real_case)].copy()
    feat = feat.dropna(subset=["min_coherence", "min_coherence_p10", "n_pairs"])
    rows = feat[["person_id", "n_pairs", "min_coherence", "min_coherence_p10",
                "mean_coherence"]].to_dict("records")

    rho_old, p_old = stats.spearmanr(feat["n_pairs"], feat["min_coherence"])
    rho_new, p_new = stats.spearmanr(feat["n_pairs"], feat["min_coherence_p10"])
    summary = {
        "n_cases": len(feat),
        "n_pairs_min": int(feat["n_pairs"].min()),
        "n_pairs_max": int(feat["n_pairs"].max()),
        "min_coherence_vs_n_pairs_rho": float(rho_old),
        "min_coherence_vs_n_pairs_p": float(p_old),
        "min_coherence_p10_vs_n_pairs_rho": float(rho_new),
        "min_coherence_p10_vs_n_pairs_p": float(p_new),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    pd.DataFrame(rows, columns=keys).to_csv(path, index=False)


# --------------------------------------------------------------------------------- Main

def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    panss_csv = ROOT / "data" / "panss_scores.csv"
    main_graph_csv = ROOT / "results" / "_overview_graph.csv"
    lemma_graph_csv = ROOT / "results_lemma" / "_overview_graph.csv"
    features_csv = ROOT / "results" / "_overview_features.csv"

    df_main = load_merged(main_graph_csv, panss_csv)
    df_lemma = load_merged(lemma_graph_csv, panss_csv)
    df_sens = df_main[~df_main["person_id"].isin(SENSITIVITY_EXCLUDE_IDS)].copy()

    print(f"Hauptstichprobe (echte Faelle, Graph x PANSS): n = {len(df_main)}")
    print(f"Sensitivitaetsstichprobe (ohne 7 Diarisierungs-Vorbehalt-Faelle): n = {len(df_sens)}")
    print(f"Lemma-Stichprobe: n = {len(df_lemma)}")

    primary = run_family(df_main, PRIMARY_MEASURES, PRIMARY_ANCHORS,
                         apply_bh=True, with_partial=True,
                         expected_dir=EXPECTED_DIRECTION)
    write_csv(out_dir / "_stats_graph_primary.csv", primary)

    sensitivity = run_family(df_sens, PRIMARY_MEASURES, PRIMARY_ANCHORS,
                             apply_bh=True, with_partial=True,
                             expected_dir=EXPECTED_DIRECTION)
    write_csv(out_dir / "_stats_graph_sensitivity.csv", sensitivity)

    lemma = run_family(df_lemma, PRIMARY_MEASURES, PRIMARY_ANCHORS,
                       apply_bh=True, with_partial=True,
                       expected_dir=EXPECTED_DIRECTION)
    write_csv(out_dir / "_stats_graph_lemma.csv", lemma)

    floor_table = run_floor_table(df_main)
    write_csv(out_dir / "_stats_graph_floor_anchors.csv", floor_table)

    exploratory = run_exploratory_table(df_main)
    write_csv(out_dir / "_stats_graph_exploratory.csv", exploratory)

    length_confound = run_length_confound_table(df_main)
    write_csv(out_dir / "_stats_graph_length_confound.csv", length_confound)

    coherence_rows, coherence_summary = run_coherence_robustness(features_csv)
    write_csv(out_dir / "_overview_coherence_robust.csv", coherence_rows)

    print("\nGeschrieben:")
    for name in ["_stats_graph_primary.csv", "_stats_graph_sensitivity.csv",
                "_stats_graph_lemma.csv", "_stats_graph_floor_anchors.csv",
                "_stats_graph_exploratory.csv", "_stats_graph_length_confound.csv",
                "_overview_coherence_robust.csv"]:
        print(f"  results/{name}")

    print("\n--- min_coherence Laengenkonfund (Aufgabe 3a) ---")
    for k, v in coherence_summary.items():
        print(f"  {k}: {v}")

    return {
        "n_main": len(df_main), "n_sensitivity": len(df_sens), "n_lemma": len(df_lemma),
        "primary": primary, "sensitivity": sensitivity, "lemma": lemma,
        "floor_table": floor_table, "coherence_summary": coherence_summary,
    }


if __name__ == "__main__":
    main()
