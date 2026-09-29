"""Statistische Auswertungen über die Pipeline-Outputs hinweg.

Liest alle Personen-Aggregate aus results/ und berechnet:
  1) Deskriptive Statistik der IPASE-Cosine-Werte (pro Person + global)
  2) Konvergenz mpnet ↔ e5: Korrelation, Bland-Altman-Kennzahlen
  3) Konvergenz Cosine ↔ Perplexität: Spearman + Pearson über Personen
  4) Faktor-Profil-Statistiken (welcher IPASE-Faktor variiert am stärksten?)
  5) Korrelationen mit klinischen Ankern (PANSS, falls panss_scores.csv vorhanden,
     sonst Demo-Lauf mit zufälligen Platzhaltern, set.seed-fixed)
  6) Zusammenführung mit Sprach-Features aus 05_features.py

Schreibt:
  - results/_stats_descriptives.csv
  - results/_stats_convergence_embedding.csv
  - results/_stats_convergence_perplexity.csv
  - results/_stats_factor_profiles.csv
  - results/_stats_panss_correlations.csv
  - results/_stats_combined_table.csv (alle Personen-Metriken nebeneinander)
  - results/_stats_report.txt (Klartext-Zusammenfassung)

Verwendung:
    python src/06_stats.py
    python src/06_stats.py --panss data/panss_scores.csv
"""
import argparse
import json
import csv
from pathlib import Path
from statistics import mean, median, stdev
import math

import numpy as np
from scipy import stats

# Synthetische/Test-Fälle: NIEMALS in klinische Korrelationen (H1/H2, M.5.5)
# einbeziehen. Nur echte Fall-IDs (Nr_XXX) gehören in die Anker-Tests.
SYNTHETIC_IDS = {"me2_pathologisch", "psychose_referenz",
                 "Test audio für NLP pipeline"}


def is_real_case(person_id: str) -> bool:
    """Echte Fälle heißen Nr_XXX; alles andere ist Synthetik/Testmaterial."""
    return person_id.startswith("Nr_") and person_id not in SYNTHETIC_IDS


def load_overview(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_aggregates(results_dir: Path) -> dict[str, dict]:
    """Lade alle *.aggregates.json Files."""
    out = {}
    for f in results_dir.glob("*.aggregates.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        out[data["person_id"]] = data
    return out


def load_similarity_matrices(results_dir: Path) -> dict[str, np.ndarray]:
    """Lade alle *.similarity_matrix.csv Files als numpy arrays."""
    out = {}
    for f in results_dir.glob("*.similarity_matrix.csv"):
        person_id = f.stem.replace(".similarity_matrix", "")
        rows = []
        with open(f, "r", encoding="utf-8") as cf:
            reader = csv.reader(cf)
            header = next(reader)
            item_cols = [i for i, h in enumerate(header) if h.startswith("item_")]
            for row in reader:
                rows.append([float(row[i]) for i in item_cols])
        if rows:
            out[person_id] = np.array(rows)
    return out


def descriptive_stats(sim_matrices: dict[str, np.ndarray]) -> list[dict]:
    """Verteilungs-Kennzahlen der Cosine-Werte pro Person + global."""
    rows = []
    all_vals = []
    for pid, mat in sim_matrices.items():
        flat = mat.flatten()
        all_vals.extend(flat.tolist())
        rows.append({
            "person_id": pid,
            "n_obs": int(flat.size),
            "n_sentences": int(mat.shape[0]),
            "mean": float(flat.mean()),
            "sd": float(flat.std(ddof=1)) if flat.size > 1 else 0.0,
            "median": float(np.median(flat)),
            "q25": float(np.percentile(flat, 25)),
            "q75": float(np.percentile(flat, 75)),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "skewness": float(stats.skew(flat)),
            "kurtosis": float(stats.kurtosis(flat)),
        })
    if all_vals:
        v = np.array(all_vals)
        rows.append({
            "person_id": "_global",
            "n_obs": int(v.size),
            "n_sentences": sum(m.shape[0] for m in sim_matrices.values()),
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)),
            "median": float(np.median(v)),
            "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75)),
            "min": float(v.min()),
            "max": float(v.max()),
            "skewness": float(stats.skew(v)),
            "kurtosis": float(stats.kurtosis(v)),
        })
    return rows


def embedding_convergence(mpnet_csv: list[dict], e5_csv: list[dict]) -> dict:
    """Konvergenz-Kennzahlen für mpnet vs. e5 (über Personen-Aggregate)."""
    by_pid_mpnet = {r["person_id"]: float(r["mean"]) for r in mpnet_csv}
    by_pid_e5 = {r["person_id"]: float(r["mean_overall"]) for r in e5_csv}
    common = sorted(set(by_pid_mpnet) & set(by_pid_e5))
    if len(common) < 2:
        return {
            "n_persons": len(common),
            "note": "Zu wenige Personen für Korrelation. Bei n≥10 verlässlich.",
            "per_person": [{"person_id": p,
                            "mpnet_mean": by_pid_mpnet[p],
                            "e5_mean": by_pid_e5[p],
                            "diff": by_pid_e5[p] - by_pid_mpnet[p]} for p in common],
        }
    a = np.array([by_pid_mpnet[p] for p in common])
    b = np.array([by_pid_e5[p] for p in common])
    pearson_r, pearson_p = stats.pearsonr(a, b)
    spearman_r, spearman_p = stats.spearmanr(a, b)
    diff = b - a
    return {
        "n_persons": len(common),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "bland_altman_mean_diff": float(diff.mean()),
        "bland_altman_sd_diff": float(diff.std(ddof=1)),
        "bland_altman_loa_lower": float(diff.mean() - 1.96 * diff.std(ddof=1)),
        "bland_altman_loa_upper": float(diff.mean() + 1.96 * diff.std(ddof=1)),
        "per_person": [{"person_id": p,
                        "mpnet_mean": by_pid_mpnet[p],
                        "e5_mean": by_pid_e5[p],
                        "diff": by_pid_e5[p] - by_pid_mpnet[p]} for p in common],
    }


def perplexity_convergence(mpnet_csv: list[dict], pp_csv: list[dict]) -> dict:
    """Konvergenz: niedrige Perplexität ↔ hoher Cosine?"""
    by_pid_cos = {r["person_id"]: float(r["mean"]) for r in mpnet_csv}
    by_pid_pp = {r["person_id"]: float(r["mean_perplexity"]) for r in pp_csv}
    common = sorted(set(by_pid_cos) & set(by_pid_pp))
    if len(common) < 2:
        return {
            "n_persons": len(common),
            "note": "Zu wenige Personen für Korrelation.",
            "per_person": [{"person_id": p,
                            "cosine": by_pid_cos[p],
                            "perplexity": by_pid_pp[p],
                            "neg_log_pp": -math.log(by_pid_pp[p]) if by_pid_pp[p] > 0 else None}
                           for p in common],
        }
    cos = np.array([by_pid_cos[p] for p in common])
    pp = np.array([by_pid_pp[p] for p in common])
    neg_log_pp = -np.log(pp)
    pearson_r, pearson_p = stats.pearsonr(cos, neg_log_pp)
    spearman_r, spearman_p = stats.spearmanr(cos, pp)
    return {
        "n_persons": len(common),
        "pearson_r_cos_vs_neglogpp": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r_cos_vs_pp": float(spearman_r),
        "spearman_p": float(spearman_p),
        "interpretation": (
            "Erwartet: Hoher Cosine ↔ niedrige Perplexität "
            "(positiv mit -log(PP), negativ mit PP)."
        ),
        "per_person": [{"person_id": p,
                        "cosine": by_pid_cos[p],
                        "perplexity": by_pid_pp[p],
                        "neg_log_pp": float(-math.log(by_pid_pp[p])) if by_pid_pp[p] > 0 else None}
                       for p in common],
    }


def factor_profile_stats(aggregates: dict[str, dict]) -> list[dict]:
    """Pro Person: Faktor-Profil (5 Means + Range + welcher Faktor am höchsten)."""
    factor_names = {
        "1": "Cognition",
        "2": "Self-Awareness/Presence",
        "3": "Consciousness",
        "4": "Somatization/Embodiment",
        "5": "Demarcation/Transitivism",
    }
    rows = []
    for pid, agg in aggregates.items():
        sub = agg["factor_subindices"]
        means = {fid: sub[fid]["mean"] for fid in ("1", "2", "3", "4", "5")}
        vals = list(means.values())
        max_fid = max(means, key=means.get)
        min_fid = min(means, key=means.get)
        rows.append({
            "person_id": pid,
            **{f"f{fid}_{factor_names[fid].split('/')[0].lower()}": v for fid, v in means.items()},
            "factor_range": float(max(vals) - min(vals)),
            "factor_sd": float(np.std(vals, ddof=1)),
            "highest_factor": f"F{max_fid} {factor_names[max_fid]}",
            "lowest_factor": f"F{min_fid} {factor_names[min_fid]}",
        })
    return rows


def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray,
                          n_boot: int = 10_000, alpha: float = 0.05,
                          seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap-Konfidenzintervall für Spearman ρ.

    Implementiert nach M.5.3 des Methodenkapitels: 10 000 Bootstrap-Ziehungen,
    Percentile-Methode für das CI. Gibt (ρ, ci_lower, ci_upper) zurück.
    """
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        x_b, y_b = x[idx], y[idx]
        # spearmanr kann bei konstanten Bootstrap-Stichproben NaN liefern
        if np.std(x_b) == 0 or np.std(y_b) == 0:
            rhos[i] = np.nan
            continue
        rho, _ = stats.spearmanr(x_b, y_b)
        rhos[i] = rho
    rhos = rhos[~np.isnan(rhos)]
    if len(rhos) == 0:
        return float("nan"), float("nan"), float("nan")
    point_rho, _ = stats.spearmanr(x, y)
    lo = float(np.percentile(rhos, 100 * alpha / 2))
    hi = float(np.percentile(rhos, 100 * (1 - alpha / 2)))
    return float(point_rho), lo, hi


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg FDR-Korrektur.

    Returns (adjusted_p_values, rejected_bool_list) parallel zur Eingabe.
    """
    n = len(p_values)
    if n == 0:
        return [], []
    order = sorted(range(n), key=lambda i: p_values[i])
    adj_sorted = [0.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        rank_in_sorted = n - rank + 1
        adj = p_values[idx] * n / rank_in_sorted
        adj = min(adj, prev)
        adj_sorted[idx] = adj
        prev = adj
    rejected = [p <= alpha for p in adj_sorted]
    return adj_sorted, rejected


def primary_hypothesis_tests(combined: list[dict],
                             panss_rows: list[dict]) -> dict:
    """Primäre Hypothesentests H1 und H2 nach M.5.3.

    H1: IPASE-Similarity-Index korreliert positiv mit PANSS-P2 und SAPS-Denkstörung
    H2: IPASE-Similarity-Index korreliert positiv mit SANS-Total und PANSS-N

    Spearman-Rangkorrelation, Bootstrap-CI (10 000), Bonferroni über 4 Tests
    (2 pro Familie). Adjustiertes α = 0.05 / 2 = 0.025 nach M.5.3.
    """
    by_pid = {r["person_id"]: r for r in combined}
    by_pid_panss = {r["person_id"]: r for r in panss_rows}
    # Nur echte Fälle: synthetisches/Test-Material verzerrt klinische Korrelationen.
    common = sorted(p for p in (set(by_pid) & set(by_pid_panss)) if is_real_case(p))

    if len(common) < 5:
        return {
            "n_persons": len(common),
            "note": "Zu wenige Personen für H1/H2 mit Bootstrap. Mindestens n=5; "
                    "M.5.3 spezifiziert N=30 bis 40 für die echte Studie.",
            "tests": [],
        }

    x = np.array([float(by_pid[p]["mean"]) for p in common])

    tests = []
    test_specs = [
        ("H1", "ipase_mean", "panss_p2", "PANSS-P2 (Conceptual disorganization)"),
        ("H1", "ipase_mean", "saps_thought_disorder", "SAPS Thought Disorder"),
        ("H2", "ipase_mean", "sans_total", "SANS Total"),
        ("H2", "ipase_mean", "panss_negative", "PANSS-N"),
    ]

    p_values = []
    for fam, _xkey, ykey, ylabel in test_specs:
        if ykey not in by_pid_panss[common[0]]:
            tests.append({
                "family": fam,
                "anchor_key": ykey,
                "anchor_label": ylabel,
                "note": f"Anchor {ykey} fehlt in PANSS-Datei, Test übersprungen.",
            })
            p_values.append(float("nan"))
            continue
        y = np.array([float(by_pid_panss[p][ykey]) for p in common])
        rho, ci_lo, ci_hi = bootstrap_spearman_ci(x, y)
        _, p_val = stats.spearmanr(x, y)
        tests.append({
            "family": fam,
            "anchor_key": ykey,
            "anchor_label": ylabel,
            "n": len(common),
            "spearman_rho": float(rho),
            "spearman_p_uncorrected": float(p_val),
            "ci95_lower": float(ci_lo),
            "ci95_upper": float(ci_hi),
        })
        p_values.append(float(p_val))

    # Bonferroni über die 4 Tests (Family-Wise α = 0.05, adjustiert auf 0.0125
    # innerhalb jeder Familie mit 2 Tests; alternativ über alle 4: 0.0125).
    # Methodenkapitel M.5.3: zwei Test-Familien, je adjustiertes α = 0.05/2 = 0.025
    # → das gilt PRO FAMILIE, nicht über alle 4. Wir berichten beide Lesarten.
    bonferroni_alpha_per_family = 0.025   # nach M.5.3
    bonferroni_alpha_global = 0.05 / 4    # konservative globale Lesart

    for t in tests:
        if "spearman_p_uncorrected" not in t:
            continue
        p = t["spearman_p_uncorrected"]
        t["significant_at_alpha_per_family_025"] = bool(p < bonferroni_alpha_per_family)
        t["significant_at_alpha_global_0125"] = bool(p < bonferroni_alpha_global)

    return {
        "n_persons": len(common),
        "bonferroni_alpha_per_family": bonferroni_alpha_per_family,
        "bonferroni_alpha_global": bonferroni_alpha_global,
        "method_notes": (
            "Spearman ρ mit 10 000-Bootstrap-Perzentil-95%-CI. "
            "Bonferroni nach M.5.3: α/2 = .025 pro Familie (H1, H2). "
            "Globale konservative Lesart: α/4 = .0125."
        ),
        "tests": tests,
    }


def incremental_validity(combined: list[dict], panss_rows: list[dict]) -> dict:
    """Inkrementelle Validität des IPASE-Index (kritischer Test, fable-Audit P4).

    Frage: Klärt ein IPASE-Maß Varianz in den klinischen Ankern auf, die über
    die konventionellen Marker (Kohärenz, Sprachmenge, Ich-Pronomenfrequenz)
    hinausgeht? Methode: Rang-Transformation aller Variablen (Spearman-Logik),
    dann Semipartialkorrelation: Index an den Kovariaten residualisiert (OLS
    auf Rängen), Korrelation der Residuen mit dem Anker-Rang.

    Stichproben-Guards: Volles Modell (3 Kovariaten) ab n >= 10; einzelne
    Kovariaten ab n >= 6. Darunter nur Zero-Order-Werte mit Warnhinweis.
    """
    covariate_keys = ["mean_coherence", "n_tokens_total", "freq_ich_per100"]
    index_keys = ["mean", "mean_sent_max", "specific_hit_rate"]
    anchor_keys = ["panss_p2", "saps_thought_disorder", "sans_total",
                   "panss_negative"]

    by_pid = {r["person_id"]: r for r in combined}
    by_pid_panss = {r["person_id"]: r for r in panss_rows}
    common = sorted(p for p in (set(by_pid) & set(by_pid_panss)) if is_real_case(p))
    n = len(common)

    def col(source, key):
        vals = []
        for p in common:
            v = source[p].get(key, "")
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                return None
        return np.array(vals)

    def semipartial(x_rank, y_rank, cov_ranks):
        """Korrelation von y mit dem an cov residualisierten x (auf Rängen)."""
        X = np.column_stack([np.ones(len(x_rank))] + cov_ranks)
        beta, *_ = np.linalg.lstsq(X, x_rank, rcond=None)
        resid = x_rank - X @ beta
        if np.allclose(resid.std(), 0):
            return float("nan")
        return float(np.corrcoef(resid, y_rank)[0, 1])

    results = []
    for ikey in index_keys:
        x = col(by_pid, ikey)
        if x is None:
            results.append({"index": ikey,
                            "note": "Metrik fehlt (03_similarity neu laufen lassen)."})
            continue
        x_rank = stats.rankdata(x)
        covs = {}
        for ck in covariate_keys:
            c = col(by_pid, ck)
            if c is not None:
                covs[ck] = stats.rankdata(c)
        for akey in anchor_keys:
            y = col(by_pid_panss, akey)
            if y is None:
                continue
            y_rank = stats.rankdata(y)
            row = {"index": ikey, "anchor": akey, "n": n,
                   "zero_order_spearman": float(stats.spearmanr(x, y)[0])}
            # Einzel-Kovariaten-Semipartials
            if n >= 6:
                for ck, c_rank in covs.items():
                    row[f"semipartial_vs_{ck}"] = semipartial(x_rank, y_rank, [c_rank])
            # Volles Modell
            if n >= 10 and len(covs) == len(covariate_keys):
                row["semipartial_full_model"] = semipartial(
                    x_rank, y_rank, list(covs.values()))
            if n < 6:
                row["note"] = f"n = {n} zu klein für Semipartials (min. 6)."
            results.append(row)

    return {
        "n_persons": n,
        "covariates": covariate_keys,
        "method_notes": (
            "Semipartial auf Rängen: Index OLS-residualisiert an Kovariaten, "
            "Residuen korreliert mit Anker-Rang. Interpretierbar erst ab n >= 10; "
            "bei kleinem n rein deskriptiv. Kritischer Test: bleibt vom "
            "Zero-Order-Zusammenhang nach Kontrolle konventioneller Marker "
            "etwas übrig?"
        ),
        "results": results,
    }


def m_moderator_analysis(combined: list[dict], panss_rows: list[dict],
                         sim_matrices: dict) -> dict:
    """m-SANS-Konfund (M.4.3): Moderator- und Reliabilitätsanalysen.

    m = Zahl der Ich-Sätze pro Person. Da SE(Index) ∝ 1/√m und m mit SANS
    korreliert, ist die Messpräzision mit der H2-Zielvariablen konfundiert.
    Vier Bausteine: (1) Median-Split nach m, (2) √m-gewichtete Rang-Korrelation,
    (3) Interaktionstest Anker ~ Index × m auf Rängen (deskriptiv, low power),
    (4) Disattenuation via per-Person-SE aus den Similarity-Matrizen.
    """
    anchor_keys = ["panss_p2", "saps_thought_disorder", "sans_total",
                   "panss_negative"]
    index_keys = ["mean", "mean_sent_max"]

    by_pid = {r["person_id"]: r for r in combined}
    by_pid_panss = {r["person_id"]: r for r in panss_rows}
    common = sorted(p for p in (set(by_pid) & set(by_pid_panss)) if is_real_case(p))
    n = len(common)
    if n < 10:
        return {"n_persons": n, "note": "n < 10, Moderator-Analyse übersprungen.",
                "results": []}

    m_arr = np.array([float(by_pid[p]["n_ich_saetze"]) for p in common])

    def col(source, key):
        vals = []
        for p in common:
            try:
                vals.append(float(source[p].get(key, "")))
            except (TypeError, ValueError):
                return None
        return np.array(vals)

    def weighted_rank_corr(x, y, w):
        rx, ry = stats.rankdata(x), stats.rankdata(y)
        mx = np.average(rx, weights=w)
        my = np.average(ry, weights=w)
        cov = np.average((rx - mx) * (ry - my), weights=w)
        sx = np.sqrt(np.average((rx - mx) ** 2, weights=w))
        sy = np.sqrt(np.average((ry - my) ** 2, weights=w))
        return float(cov / (sx * sy)) if sx > 0 and sy > 0 else float("nan")

    def interaction_beta(x, y, mod):
        """OLS auf z-standardisierten Rängen: y ~ x + mod + x*mod."""
        def zr(v):
            r = stats.rankdata(v)
            return (r - r.mean()) / r.std()
        xr, yr, mr = zr(x), zr(y), zr(mod)
        X = np.column_stack([np.ones(len(xr)), xr, mr, xr * mr])
        beta, *_ = np.linalg.lstsq(X, yr, rcond=None)
        resid = yr - X @ beta
        dof = len(xr) - X.shape[1]
        if dof <= 0:
            return float("nan"), float("nan")
        s2 = float(resid @ resid) / dof
        try:
            cov = s2 * np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        se = np.sqrt(np.diag(cov))[3]
        t = beta[3] / se if se > 0 else float("nan")
        p = 2 * (1 - stats.t.cdf(abs(t), dof)) if np.isfinite(t) else float("nan")
        return float(beta[3]), float(p)

    def per_person_se(ikey):
        """SE des Personen-Index aus den per-Satz-Werten der Similarity-Matrix."""
        ses = []
        for p in common:
            mat = sim_matrices.get(p)
            if mat is None or mat.shape[0] < 2:
                return None
            vals = mat.mean(axis=1) if ikey == "mean" else mat.max(axis=1)
            ses.append(float(vals.std(ddof=1) / np.sqrt(len(vals))))
        return np.array(ses)

    out = {"n_persons": n,
           "m_summary": {"median": float(np.median(m_arr)),
                         "min": float(m_arr.min()), "max": float(m_arr.max())},
           "m_vs_anchors": {}, "results": []}

    for a in anchor_keys:
        y = col(by_pid_panss, a)
        if y is not None:
            rho, p = stats.spearmanr(m_arr, y)
            out["m_vs_anchors"][a] = {"rho": float(rho), "p": float(p)}

    med = float(np.median(m_arr))
    hi = m_arr >= med
    w = np.sqrt(m_arr)

    for ikey in index_keys:
        x = col(by_pid, ikey)
        if x is None:
            out["results"].append({"index": ikey, "note": "Metrik fehlt."})
            continue
        rho_xm, _ = stats.spearmanr(x, m_arr)
        ses = per_person_se(ikey)
        rel = None
        if ses is not None:
            var_obs = float(x.var(ddof=1))
            mean_se2 = float((ses ** 2).mean())
            rel = max(0.0, var_obs - mean_se2) / var_obs if var_obs > 0 else None
        for a in anchor_keys:
            y = col(by_pid_panss, a)
            if y is None:
                continue
            rho0, p0 = stats.spearmanr(x, y)
            rho_hi_v, p_hi = stats.spearmanr(x[hi], y[hi])
            rho_lo_v, p_lo = stats.spearmanr(x[~hi], y[~hi])
            b_int, p_int = interaction_beta(x, y, m_arr)
            row = {
                "index": ikey, "anchor": a, "n": n,
                "index_vs_m_rho": float(rho_xm),
                "zero_order_rho": float(rho0), "zero_order_p": float(p0),
                "split_high_m_rho": float(rho_hi_v), "split_high_m_n": int(hi.sum()),
                "split_low_m_rho": float(rho_lo_v), "split_low_m_n": int((~hi).sum()),
                "weighted_sqrt_m_rho": weighted_rank_corr(x, y, w),
                "interaction_beta": b_int, "interaction_p": p_int,
            }
            if rel is not None and rel > 0.1:
                row["mean_reliability"] = float(rel)
                row["disattenuated_rho"] = float(
                    max(-1.0, min(1.0, rho0 / np.sqrt(rel))))
            out["results"].append(row)

    out["method_notes"] = (
        "Median-Split und √m-Gewichtung deskriptiv; Interaktionstest auf "
        "z-standardisierten Rängen (low power bei n<60). Disattenuation nach "
        "klassischer Messfehlertheorie (Anker-Reliabilität = 1 angenommen): "
        "Sensitivitätswert für die Diskussion, kein Hauptergebnis. Lesart: "
        "split_high_m_rho > split_low_m_rho und weighted > zero_order stützen "
        "die Attenuations-These (echtes Signal, durch geringe Sprachmenge "
        "verrauscht); umgekehrtes Muster spricht für Artefakt."
    )
    return out


def panss_correlations(combined: list[dict], panss_rows: list[dict] | None) -> dict:
    """Explorative Konstrukt-Validierungs-Korrelationen IPASE-Subindizes ↔ PANSS.

    Implementiert M.5.5: bis zu 25 Tests (5 Faktor-Sub-Indizes × 5 Anker),
    FDR-Korrektur nach Benjamini-Hochberg, alpha = .05.

    Wenn panss_rows None ist, wird ein Demo-Lauf mit zufälligen Werten (seed=42)
    gemacht.
    """
    by_pid = {r["person_id"]: r for r in combined}
    # M.5.5: 5 IPASE-Faktor-Sub-Indizes × 5 klinische Anker = 25 Tests
    # Anker laut M.5.5: PANSS-P2, SAPS-Denkstörung, SANS, PANSS-N, CDSS
    anchor_keys = ["panss_p2", "saps_thought_disorder", "sans_total",
                   "panss_negative", "cdss_total"]
    is_demo = False
    if panss_rows is None:
        is_demo = True
        rng = np.random.default_rng(42)
        panss_rows = []
        for pid in by_pid:
            panss_rows.append({
                "person_id": pid,
                "panss_p2": int(rng.integers(1, 8)),
                "saps_thought_disorder": int(rng.integers(0, 26)),
                "sans_total": int(rng.integers(0, 126)),
                "panss_negative": int(rng.integers(7, 49)),
                "cdss_total": int(rng.integers(0, 28)),
            })
    panss_by_pid = {r["person_id"]: r for r in panss_rows}

    # Nur echte Fälle (Nr_XXX) in klinische Korrelationen.
    common = sorted(p for p in (set(by_pid) & set(panss_by_pid)) if is_real_case(p))
    if len(common) < 3:
        return {
            "is_demo": is_demo,
            "n_persons": len(common),
            "note": "Zu wenige Personen (n<3). In der echten Studie n=30-40 angestrebt.",
            "panss_demo_values": panss_by_pid if is_demo else None,
            "correlations": [],
        }

    # Nur die 5 Faktor-Sub-Indizes (25 Tests), ohne ipase_mean.
    # M.5.5 spezifiziert die 5x5-Matrix der Sub-Indizes gegen die Anker.
    metric_keys = [
        ("ipase_f1_cognition", "f1_cognition"),
        ("ipase_f2_self", "f2_self-awareness"),
        ("ipase_f3_consc", "f3_consciousness"),
        ("ipase_f4_soma", "f4_somatization"),
        ("ipase_f5_demarcation", "f5_demarcation"),
    ]
    correlations = []
    p_values = []
    for ipase_label, ipase_col in metric_keys:
        if ipase_col not in by_pid[common[0]]:
            continue
        x = np.array([float(by_pid[p][ipase_col]) for p in common])
        for pk in anchor_keys:
            if pk not in panss_by_pid[common[0]]:
                continue
            # Leere Werte (fehlende PANSS-Daten) herausfiltern
            valid = [p for p in common if panss_by_pid[p].get(pk, "").strip() != ""]
            if len(valid) < 5:
                continue
            x_v = np.array([float(by_pid[p][ipase_col]) for p in valid])
            y = np.array([float(panss_by_pid[p][pk]) for p in valid])
            try:
                r, p = stats.spearmanr(x_v, y)
                correlations.append({
                    "ipase_metric": ipase_label,
                    "anchor": pk,
                    "spearman_r": float(r),
                    "spearman_p_uncorrected": float(p),
                    "n": len(valid),
                })
                p_values.append(float(p))
            except Exception as e:
                correlations.append({"ipase_metric": ipase_label, "anchor": pk,
                                     "error": str(e)})

    # Benjamini-Hochberg FDR über alle Tests.
    if p_values:
        bh_p, bh_rejected = benjamini_hochberg(p_values, alpha=0.05)
        for c, p_adj, rej in zip([c for c in correlations if "spearman_p_uncorrected" in c],
                                  bh_p, bh_rejected):
            c["bh_adjusted_p"] = float(p_adj)
            c["significant_fdr_05"] = bool(rej)

    return {
        "is_demo": is_demo,
        "n_persons": len(common),
        "n_tests": len([c for c in correlations if "spearman_p_uncorrected" in c]),
        "panss_source": "Demo (seed=42)" if is_demo else "data/panss_scores.csv",
        "method_notes": (
            "Spearman ρ für jede Kombination aus IPASE-Faktor-Sub-Index und klinischem "
            f"Anker (5 × {len(anchor_keys)} = {5 * len(anchor_keys)} Tests, vgl. M.5.5). "
            "Anker: PANSS-P2, SAPS-Denkstörung, SANS-Total, PANSS-N, CDSS-Total. "
            "Benjamini-Hochberg-FDR-Korrektur (α = .05). "
            "Explorativ, hypothesengenerierend, nicht hypothesenprüfend."
        ),
        "correlations": correlations,
    }


def combined_table(aggregates: dict, pp_csv: list[dict], features: list[dict]) -> list[dict]:
    """Eine Zeile pro Person mit allen wichtigen Metriken."""
    by_pid_pp = {r["person_id"]: r for r in pp_csv}
    by_pid_feat = {r["person_id"]: r for r in features}
    rows = []
    for pid, agg in aggregates.items():
        sub = agg["factor_subindices"]
        row = {
            "person_id": pid,
            "n_ich_saetze": agg["n_ich_saetze"],
            "mean": agg["mean_overall"],
            "median": agg["median_overall"],
            "p95": agg["p95_overall"],
            "max": agg["max_overall"],
            "f1_cognition": sub["1"]["mean"],
            "f2_self-awareness": sub["2"]["mean"],
            "f3_consciousness": sub["3"]["mean"],
            "f4_somatization": sub["4"]["mean"],
            "f5_demarcation": sub["5"]["mean"],
            # Neue Index-Varianten (03_similarity, seit 2026-07-06); bei alten
            # Aggregates-Files fehlen die Keys → leer.
            "mean_sent_max": agg.get("mean_sent_max", ""),
            "tail_rate_55": agg.get("tail_rate_55", ""),
            "specific_hit_rate": agg.get("specific_hit_rate", ""),
            "drift_share_of_hits": agg.get("drift_share_of_hits", ""),
        }
        if pid in by_pid_pp:
            row["mean_perplexity"] = float(by_pid_pp[pid]["mean_perplexity"])
            row["median_perplexity"] = float(by_pid_pp[pid]["median_perplexity"])
        if pid in by_pid_feat:
            f = by_pid_feat[pid]
            row["n_tokens_total"] = float(f.get("n_tokens_total", "nan"))
            row["freq_ich_per100"] = float(f["freq_ich_per100"])
            row["mean_coherence"] = float(f["mean_coherence"])
            row["mean_length_utterance"] = float(f["mean_length_utterance"])
            row["type_token_ratio"] = float(f["type_token_ratio"])
            row["analytic_surrogate"] = float(f["analytic_surrogate"])
            row["concreteness_surrogate"] = float(f["concreteness_surrogate"])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_report(path: Path, descriptives, emb_conv, pp_conv, factor_rows,
                 panss_result, combined, h1h2_result=None):
    lines = []
    lines.append("=" * 70)
    lines.append("STATISTIK-REPORT — IPASE-Pipeline")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Personen: {len([r for r in descriptives if r['person_id'] != '_global'])}")
    lines.append("")
    lines.append("--- 1. Deskriptiv (Cosine-Werte) ---")
    for r in descriptives:
        lines.append(f"  {r['person_id']:30s} n={r['n_obs']:4d}  "
                     f"M={r['mean']:.3f}  SD={r['sd']:.3f}  "
                     f"Median={r['median']:.3f}  [Min={r['min']:.3f}, Max={r['max']:.3f}]")
    lines.append("")
    lines.append("--- 2. Konvergenz mpnet vs. e5 ---")
    lines.append(f"  n_persons: {emb_conv['n_persons']}")
    if "pearson_r" in emb_conv:
        lines.append(f"  Pearson r = {emb_conv['pearson_r']:.3f} (p={emb_conv['pearson_p']:.3f})")
        lines.append(f"  Bland-Altman: Mean-Diff = {emb_conv['bland_altman_mean_diff']:.3f}, "
                     f"LoA [{emb_conv['bland_altman_loa_lower']:.3f}, "
                     f"{emb_conv['bland_altman_loa_upper']:.3f}]")
    else:
        lines.append(f"  {emb_conv.get('note', '')}")
        for r in emb_conv["per_person"]:
            lines.append(f"    {r['person_id']:30s} mpnet={r['mpnet_mean']:.3f}  "
                         f"e5={r['e5_mean']:.3f}  Diff={r['diff']:+.3f}")
    lines.append("")
    lines.append("--- 3. Konvergenz Cosine ↔ Perplexität ---")
    lines.append(f"  n_persons: {pp_conv['n_persons']}")
    if "pearson_r_cos_vs_neglogpp" in pp_conv:
        lines.append(f"  Pearson r (Cosine vs. -log PP) = "
                     f"{pp_conv['pearson_r_cos_vs_neglogpp']:.3f} (p={pp_conv['pearson_p']:.3f})")
        lines.append(f"  Spearman r (Cosine vs. PP)     = "
                     f"{pp_conv['spearman_r_cos_vs_pp']:.3f}")
    else:
        lines.append(f"  {pp_conv.get('note', '')}")
        for r in pp_conv["per_person"]:
            nl = f"{r['neg_log_pp']:.3f}" if r['neg_log_pp'] is not None else "n/a"
            lines.append(f"    {r['person_id']:30s} Cosine={r['cosine']:.3f}  "
                         f"PP={r['perplexity']:.2f}  -log(PP)={nl}")
    lines.append("")
    lines.append("--- 4. Faktor-Profile ---")
    for r in factor_rows:
        lines.append(f"  {r['person_id']:30s} höchster Faktor: {r['highest_factor']}")
        lines.append(f"  {'':30s} Faktor-Range: {r['factor_range']:.3f}  "
                     f"SD über Faktoren: {r['factor_sd']:.3f}")
    lines.append("")
    lines.append("--- 5. Primäre Hypothesentests (H1, H2) ---")
    if h1h2_result is not None:
        lines.append(f"  n_persons: {h1h2_result['n_persons']}")
        if "method_notes" in h1h2_result:
            lines.append(f"  Methode: {h1h2_result['method_notes']}")
        if h1h2_result.get("tests"):
            for t in h1h2_result["tests"]:
                if "spearman_rho" in t:
                    sig_fam = "*" if t.get("significant_at_alpha_per_family_025") else " "
                    sig_glo = "*" if t.get("significant_at_alpha_global_0125") else " "
                    lines.append(
                        f"    {t['family']} | {t['anchor_label']:36s}"
                        f"  ρ = {t['spearman_rho']:+.3f}"
                        f"  95% CI [{t['ci95_lower']:+.3f}, {t['ci95_upper']:+.3f}]"
                        f"  p = {t['spearman_p_uncorrected']:.4f}"
                        f"  [fam{sig_fam} glo{sig_glo}]"
                    )
                else:
                    lines.append(f"    {t['family']} | {t['anchor_label']:36s}  "
                                 f"{t.get('note', '')}")
            lines.append("    Legende: fam* = sig. bei α/2 = .025 (M.5.3-konform); "
                         "glo* = sig. bei α/4 = .0125 (global Bonferroni)")
        else:
            lines.append(f"  {h1h2_result.get('note', '')}")
    else:
        lines.append("  (keine PANSS-Daten geladen)")
    lines.append("")
    lines.append("--- 6. Explorative Konstrukt-Validierung (M.5.5) ---")
    if panss_result.get("is_demo"):
        lines.append("  (DEMO-Lauf mit zufälligen PANSS-Werten, seed=42)")
    lines.append(f"  n_persons: {panss_result['n_persons']}")
    if panss_result.get("method_notes"):
        lines.append(f"  Methode: {panss_result['method_notes']}")
    if panss_result.get("correlations"):
        for c in panss_result["correlations"]:
            if "spearman_r" in c:
                fdr_p = c.get("bh_adjusted_p")
                fdr_marker = "*" if c.get("significant_fdr_05") else " "
                fdr_str = f"  BH-p = {fdr_p:.4f} [{fdr_marker}]" if fdr_p is not None else ""
                lines.append(f"    {c['ipase_metric']:25s} ↔ {c['anchor']:24s} "
                             f"r = {c['spearman_r']:+.3f}  "
                             f"p = {c['spearman_p_uncorrected']:.4f}"
                             f"{fdr_str}")
        lines.append("    Legende: BH-p = Benjamini-Hochberg-FDR-adjustierter p-Wert; "
                     "[*] sig. bei FDR-α = .05")
    else:
        lines.append(f"  {panss_result.get('note', '')}")
    lines.append("")
    lines.append("=" * 70)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Statistik über Pipeline-Outputs")
    parser.add_argument("--results", default="results")
    parser.add_argument("--panss", default=None,
                        help="Optional: CSV mit panss_positive,panss_negative,panss_general")
    args = parser.parse_args()

    results_dir = Path(args.results)
    print(f"Lese Outputs aus {results_dir}/ …")

    sim_matrices = load_similarity_matrices(results_dir)
    aggregates = load_aggregates(results_dir)
    mpnet_csv = load_overview(results_dir / "_overview_similarity.csv")
    e5_csv = load_overview(results_dir / "_overview_robustness_embedding.csv")
    pp_csv = load_overview(results_dir / "_overview_robustness_perplexity.csv")
    feat_csv = load_overview(results_dir / "_overview_features.csv")

    print(f"  {len(aggregates)} Personen mit Aggregaten")
    print(f"  {len(sim_matrices)} Similarity-Matrizen")
    print(f"  mpnet: {len(mpnet_csv)}, e5: {len(e5_csv)}, PP: {len(pp_csv)}, Feat: {len(feat_csv)}")

    print("\n1) Deskriptiv …")
    descriptives = descriptive_stats(sim_matrices)
    write_csv(results_dir / "_stats_descriptives.csv", descriptives)

    print("2) Konvergenz mpnet vs. e5 …")
    emb_conv = embedding_convergence(mpnet_csv, e5_csv)
    write_csv(results_dir / "_stats_convergence_embedding.csv",
              emb_conv.get("per_person", []))
    (results_dir / "_stats_convergence_embedding.json").write_text(
        json.dumps(emb_conv, ensure_ascii=False, indent=2), encoding="utf-8")

    print("3) Konvergenz Cosine vs. Perplexität …")
    pp_conv = perplexity_convergence(mpnet_csv, pp_csv)
    write_csv(results_dir / "_stats_convergence_perplexity.csv",
              pp_conv.get("per_person", []))
    (results_dir / "_stats_convergence_perplexity.json").write_text(
        json.dumps(pp_conv, ensure_ascii=False, indent=2), encoding="utf-8")

    print("4) Faktor-Profile …")
    factor_rows = factor_profile_stats(aggregates)
    write_csv(results_dir / "_stats_factor_profiles.csv", factor_rows)

    print("5) Kombinierte Tabelle …")
    combined = combined_table(aggregates, pp_csv, feat_csv)
    write_csv(results_dir / "_stats_combined_table.csv", combined)

    print("6) PANSS-Daten laden …")
    panss_rows = None
    if args.panss and Path(args.panss).exists():
        panss_rows = load_overview(Path(args.panss))
        print(f"   PANSS aus {args.panss}: {len(panss_rows)} Personen")
    else:
        print("   Kein PANSS-File angegeben → Demo-Lauf mit seed=42")

    print("7) Primäre Hypothesentests H1/H2 (Bootstrap-CI + Bonferroni) …")
    # Demo-PANSS-Spalten für Pipeline-Test ohne echte klinische Daten.
    # Alle in M.5.3 und M.5.5 spezifizierten Anker werden erzeugt.
    anchors_are_demo = panss_rows is None
    if panss_rows is None:
        rng = np.random.default_rng(42)
        panss_rows = []
        by_pid_combined = {r["person_id"]: r for r in combined}
        for pid in by_pid_combined:
            panss_rows.append({
                "person_id": pid,
                "panss_p2": int(rng.integers(1, 8)),
                "saps_thought_disorder": int(rng.integers(0, 26)),
                "sans_total": int(rng.integers(0, 126)),
                "panss_negative": int(rng.integers(7, 49)),
                "cdss_total": int(rng.integers(0, 28)),
            })
    h1h2_result = primary_hypothesis_tests(combined, panss_rows)
    # DEMO-Kennzeichnung in JEDER Zeile: Zufallsanker (seed=42) dürfen nie mit
    # echten Ergebnissen verwechselbar sein.
    h1h2_result["anchor_source"] = ("DEMO_random_seed42" if anchors_are_demo
                                    else str(args.panss))
    for t in h1h2_result.get("tests", []):
        t["anchor_source"] = h1h2_result["anchor_source"]
    write_csv(results_dir / "_stats_primary_h1h2.csv", h1h2_result.get("tests", []))
    (results_dir / "_stats_primary_h1h2.json").write_text(
        json.dumps(h1h2_result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("8a) Inkrementelle Validität (Semipartials auf Rängen) …")
    inc_result = incremental_validity(combined, panss_rows)
    inc_result["anchor_source"] = h1h2_result["anchor_source"]
    for r in inc_result.get("results", []):
        r["anchor_source"] = h1h2_result["anchor_source"]
    write_csv(results_dir / "_stats_incremental_validity.csv",
              inc_result.get("results", []))
    (results_dir / "_stats_incremental_validity.json").write_text(
        json.dumps(inc_result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("8b) m-Moderator-Analyse (Konfund Messpräzision × SANS) …")
    mod_result = m_moderator_analysis(combined, panss_rows, sim_matrices)
    mod_result["anchor_source"] = h1h2_result["anchor_source"]
    for r in mod_result.get("results", []):
        r["anchor_source"] = h1h2_result["anchor_source"]
    write_csv(results_dir / "_stats_m_moderator.csv", mod_result.get("results", []))
    (results_dir / "_stats_m_moderator.json").write_text(
        json.dumps(mod_result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("8) Explorative Konstrukt-Validierung (FDR-korrigiert) …")
    panss_result = panss_correlations(combined, panss_rows)
    write_csv(results_dir / "_stats_panss_correlations.csv",
              panss_result.get("correlations", []))
    (results_dir / "_stats_panss_correlations.json").write_text(
        json.dumps(panss_result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("9) Report …")
    write_report(results_dir / "_stats_report.txt",
                 descriptives, emb_conv, pp_conv, factor_rows, panss_result, combined,
                 h1h2_result=h1h2_result)

    print(f"\nFertig. Report: {results_dir / '_stats_report.txt'}")


if __name__ == "__main__":
    main()
