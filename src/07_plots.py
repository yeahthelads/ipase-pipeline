"""APA-7-konforme Visualisierungen für die IPASE-Pipeline.

Erzeugt Abbildungen nach APA-7 Standards:
  - Sans-serif Font (DejaVu Sans / Arial-Familie), 11pt im Plot
  - Keine In-Plot-Titel (Titel kommen als Figure-Captions in der Arbeit)
  - Achsenbeschriftung in sentence case, Einheiten in Klammern
  - Weißer Hintergrund, keine Frame-Box (top und right spines entfernt)
  - Tick marks außen
  - Color-blind-safe: Farben + Marker-Formen + Linienstile redundant kodiert
  - Speicherung als 300 DPI PNG und PDF
  - Separate captions.md mit APA-7-konformen Figure-Captions

Erzeugte Abbildungen:
  1) Heatmap: n_ich_saetze × 57 IPASE-Items (pro Person, viridis grayscale-safe)
  2) Faktor-Profil als Polardiagramm (alle Personen überlagert)
  3) Konvergenz mpnet ↔ e5 (Scatter + Bland-Altman)
  4) Konvergenz Cosine ↔ Perplexität (Scatter)
  5) Top-Match-Verteilung auf Faktoren (gestapeltes Balkendiagramm)
  6) Personen-Metriken-Übersicht

Verwendung:
    python src/07_plots.py
    python src/07_plots.py --formats png pdf       # beide Formate (default)
    python src/07_plots.py --formats png            # nur PNG
"""
import argparse
import json
import csv
from pathlib import Path
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


FACTOR_NAMES = {
    "1": "F1 Cognition",
    "2": "F2 Self-awareness",
    "3": "F3 Consciousness",
    "4": "F4 Somatization",
    "5": "F5 Demarcation",
}

# Color-blind-safe Palette (Wong 2011, "Points of view: Color blindness")
CB_PALETTE = [
    "#000000",    # Schwarz
    "#0072B2",    # Blau
    "#D55E00",    # Vermillion
    "#009E73",    # Bluish green
    "#CC79A7",    # Reddish purple
    "#56B4E9",    # Sky blue
    "#E69F00",    # Orange
    "#F0E442",    # Gelb
]
# Marker-Shapes redundant zu Farben
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
# Linienstile redundant
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (3, 5, 1, 5))]


def apply_apa_style():
    """matplotlib rcParams auf APA-7 Defaults setzen."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "lines.linewidth": 1.5,
        "lines.markersize": 7,
        "pdf.fonttype": 42,    # Embed TrueType, editable in PDF
        "ps.fonttype": 42,
    })


def save_figure(fig, base_path: Path, formats: list[str], dpi: int = 300):
    """Speichere Abbildung in allen gewünschten Formaten."""
    for fmt in formats:
        fig.savefig(base_path.with_suffix(f".{fmt}"), dpi=dpi, format=fmt)


def load_overview(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_aggregates(results_dir: Path) -> dict[str, dict]:
    out = {}
    for f in results_dir.glob("*.aggregates.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        out[data["person_id"]] = data
    return out


def load_similarity_matrix(person_csv: Path) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    rows = []
    sentences = []
    with open(person_csv, "r", encoding="utf-8") as cf:
        reader = csv.reader(cf)
        header = next(reader)
        item_indices = [i for i, h in enumerate(header) if h.startswith("item_")]
        item_ids = [header[i].split("_")[1] for i in item_indices]
        item_factors = [int(header[i].split("_f")[1]) for i in item_indices]
        for row in reader:
            sentences.append(row[1])
            rows.append([float(row[i]) for i in item_indices])
    return np.array(rows), sentences, item_ids, item_factors


def plot_heatmap(person_id, matrix, sentences, item_ids, item_factors,
                 base_path: Path, formats: list[str]):
    """Heatmap. Viridis (color-blind-safe + grayscale-readable)."""
    n_sent, n_items = matrix.shape
    order = sorted(range(n_items), key=lambda i: (item_factors[i], int(item_ids[i])))
    sorted_mat = matrix[:, order]
    sorted_factors = [item_factors[i] for i in order]
    sorted_ids = [item_ids[i] for i in order]

    fig_height = max(3.5, n_sent * 0.42)
    fig_width = max(13, n_items * 0.20)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(sorted_mat, aspect="auto", cmap="viridis", vmin=0, vmax=0.85)
    ax.set_xticks(range(n_items))
    ax.set_xticklabels([f"F{sorted_factors[i]}.{sorted_ids[i]}" for i in range(n_items)],
                       rotation=90, fontsize=7)
    ax.set_yticks(range(n_sent))
    ax.set_yticklabels([f"S{i+1}" for i in range(n_sent)], fontsize=9)
    ax.set_xlabel("IPASE item (factor . item id)", fontsize=11)
    ax.set_ylabel("Ich-sentence index", fontsize=11)

    prev_factor = sorted_factors[0]
    for i, f in enumerate(sorted_factors):
        if f != prev_factor:
            ax.axvline(i - 0.5, color="white", lw=1.2)
            prev_factor = f
    cbar = plt.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.85)
    cbar.outline.set_linewidth(0.5)

    # Tick marks außen
    ax.tick_params(axis="both", which="both", direction="out")
    plt.tight_layout()
    save_figure(fig, base_path, formats)
    plt.close(fig)


def plot_radar(aggregates, base_path: Path, formats: list[str]):
    """Radar-Plot der Faktor-Profile."""
    factor_ids = ["1", "2", "3", "4", "5"]
    angles = [n / 5 * 2 * math.pi for n in range(5)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for idx, (pid, agg) in enumerate(aggregates.items()):
        vals = [agg["factor_subindices"][fid]["mean"] for fid in factor_ids]
        vals += vals[:1]
        ax.plot(angles, vals,
                marker=MARKERS[idx % len(MARKERS)],
                linestyle=LINESTYLES[idx % len(LINESTYLES)],
                lw=1.6, label=pid, color=CB_PALETTE[idx % len(CB_PALETTE)],
                markersize=8)
        ax.fill(angles, vals, alpha=0.08, color=CB_PALETTE[idx % len(CB_PALETTE)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([FACTOR_NAMES[f] for f in factor_ids], fontsize=10)
    all_vals = [agg["factor_subindices"][fid]["mean"]
                for agg in aggregates.values() for fid in factor_ids]
    if all_vals:
        ymax = max(all_vals) * 1.15
        ymin = max(0, min(all_vals) * 0.80)
        ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.spines["polar"].set_linewidth(0.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10))
    plt.tight_layout()
    save_figure(fig, base_path, formats)
    plt.close(fig)


def plot_embedding_convergence(mpnet_csv, e5_csv, base_path: Path, formats: list[str]):
    """mpnet vs. e5 Scatter + Bland-Altman."""
    by_pid_mpnet = {r["person_id"]: float(r["mean"]) for r in mpnet_csv}
    by_pid_e5 = {r["person_id"]: float(r["mean_overall"]) for r in e5_csv}
    common = sorted(set(by_pid_mpnet) & set(by_pid_e5))
    if not common:
        return False
    a = np.array([by_pid_mpnet[p] for p in common])
    b = np.array([by_pid_e5[p] for p in common])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # Panel A: Scatter mit Identitätslinie
    for i, pid in enumerate(common):
        ax1.scatter(a[i], b[i], s=70,
                    marker=MARKERS[i % len(MARKERS)],
                    color=CB_PALETTE[i % len(CB_PALETTE)],
                    edgecolor="black", linewidth=0.6, label=pid)
    lo = min(a.min(), b.min()) - 0.05
    hi = max(a.max(), b.max()) + 0.05
    ax1.plot([lo, hi], [lo, hi], color="black", lw=0.6,
             linestyle="--", label="Identity line")
    ax1.set_xlabel("Mean cosine, mpnet")
    ax1.set_ylabel("Mean cosine, e5")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect("equal")
    ax1.legend(loc="lower right")
    ax1.text(0.02, 0.97, "(a)", transform=ax1.transAxes,
             fontsize=12, fontweight="bold", va="top")

    # Panel B: Bland-Altman
    if len(common) >= 2:
        mean_val = (a + b) / 2
        diff = b - a
        m = diff.mean()
        sd = diff.std(ddof=1) if len(common) > 1 else 0
        for i, pid in enumerate(common):
            ax2.scatter(mean_val[i], diff[i], s=70,
                        marker=MARKERS[i % len(MARKERS)],
                        color=CB_PALETTE[i % len(CB_PALETTE)],
                        edgecolor="black", linewidth=0.6)
        ax2.axhline(m, color="black", lw=0.8)
        ax2.axhline(m + 1.96 * sd, color="grey", lw=0.6, linestyle="--")
        ax2.axhline(m - 1.96 * sd, color="grey", lw=0.6, linestyle="--")
        ax2.text(0.98, m, f"M = {m:+.3f}",
                 transform=ax2.get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=8)
        ax2.text(0.98, m + 1.96 * sd, "+1.96 SD",
                 transform=ax2.get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=8, color="grey")
        ax2.text(0.98, m - 1.96 * sd, "−1.96 SD",
                 transform=ax2.get_yaxis_transform(),
                 ha="right", va="top", fontsize=8, color="grey")
        ax2.set_xlabel("Mean of mpnet and e5 cosine")
        ax2.set_ylabel("Difference (e5 minus mpnet)")
        ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes,
                 fontsize=12, fontweight="bold", va="top")
    else:
        ax2.text(0.5, 0.5, "n < 2: insufficient for Bland-Altman",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10)
        ax2.set_axis_off()

    plt.tight_layout()
    save_figure(fig, base_path, formats)
    plt.close(fig)
    return True


def plot_cosine_perplexity(mpnet_csv, pp_csv, base_path: Path, formats: list[str]):
    """Cosine ↔ Perplexität."""
    by_pid_cos = {r["person_id"]: float(r["mean"]) for r in mpnet_csv}
    by_pid_pp = {r["person_id"]: float(r["mean_perplexity"]) for r in pp_csv}
    common = sorted(set(by_pid_cos) & set(by_pid_pp))
    if not common:
        return False
    cos = np.array([by_pid_cos[p] for p in common])
    pp = np.array([by_pid_pp[p] for p in common])

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, pid in enumerate(common):
        ax.scatter(pp[i], cos[i], s=90,
                   marker=MARKERS[i % len(MARKERS)],
                   color=CB_PALETTE[i % len(CB_PALETTE)],
                   edgecolor="black", linewidth=0.6, label=pid)
    ax.set_xscale("log")
    ax.set_xlabel("Mean conditional perplexity, XGLM (log scale)")
    ax.set_ylabel("Mean cosine similarity, mpnet")
    ax.legend(loc="best")
    plt.tight_layout()
    save_figure(fig, base_path, formats)
    plt.close(fig)
    return True


def plot_top_match_factors(results_dir, base_path: Path, formats: list[str]):
    """Top-Match-Verteilung auf Faktoren."""
    files = sorted(results_dir.glob("*.top_matches.json"))
    if not files:
        return False
    factor_counts_per_person = {}
    factor_name_to_id = {
        "Cognition": "1",
        "Self-Awareness": "2",
        "Consciousness": "3",
        "Somatization": "4",
        "Demarcation": "5",
    }
    for f in files:
        pid = f.stem.replace(".top_matches", "")
        matches = json.loads(f.read_text(encoding="utf-8"))
        counts = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        for m in matches:
            fname = m["best_match_factor"]
            for prefix, fid in factor_name_to_id.items():
                if fname.startswith(prefix):
                    counts[fid] += 1
                    break
        factor_counts_per_person[pid] = counts

    persons = list(factor_counts_per_person.keys())
    factor_ids = ["1", "2", "3", "4", "5"]
    n_persons = len(persons)
    bar_width = 0.8 / n_persons

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(5)

    # Hatch-Patterns für Print-/Grayscale-Lesbarkeit
    hatches = ["", "//", "..", "xx", "\\\\", "++", "oo", "**"]

    for i, pid in enumerate(persons):
        counts = [factor_counts_per_person[pid][fid] for fid in factor_ids]
        ax.bar(x + i * bar_width, counts, bar_width, label=pid,
               color=CB_PALETTE[i % len(CB_PALETTE)],
               edgecolor="black", linewidth=0.5,
               hatch=hatches[i % len(hatches)])
    ax.set_xticks(x + bar_width * (n_persons - 1) / 2)
    ax.set_xticklabels([FACTOR_NAMES[fid] for fid in factor_ids],
                       rotation=20, ha="right")
    ax.set_ylabel("Number of Ich-sentences with top match on this factor")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, base_path, formats)
    plt.close(fig)
    return True


def plot_person_metrics_bar(combined_csv, base_path: Path, formats: list[str]):
    """Vier zentrale Metriken pro Person als kleines Multi-Panel."""
    rows = load_overview(combined_csv)
    if not rows:
        return False
    persons = [r["person_id"] for r in rows]
    metrics = [
        ("mean", "Mean cosine, mpnet", False),
        ("mean_perplexity", "Mean perplexity, XGLM (log)", True),
        ("freq_ich_per100", "First-person singular per 100 tokens", False),
        ("mean_coherence", "Mean coherence, consecutive sentences", False),
    ]
    available = [(k, label, logy) for k, label, logy in metrics if k in rows[0]]
    n = len(available)
    if n == 0:
        return False

    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.2))
    if n == 1:
        axes = [axes]
    panel_letters = ["(a)", "(b)", "(c)", "(d)"]
    for ax_idx, (ax, (key, label, logy)) in enumerate(zip(axes, available)):
        vals = [float(r[key]) if r.get(key) not in (None, "", "nan") else 0 for r in rows]
        bars = ax.bar(range(len(persons)), vals,
                      color=[CB_PALETTE[i % len(CB_PALETTE)] for i in range(len(persons))],
                      edgecolor="black", linewidth=0.5)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(range(len(persons)))
        ax.set_xticklabels(persons, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=10)
        ax.text(0.02, 0.97, panel_letters[ax_idx], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
    plt.tight_layout()
    save_figure(fig, base_path, formats)
    plt.close(fig)
    return True


def write_captions(out_dir: Path, aggregates_count: int):
    """APA-7 Figure-Captions als markdown, zum Copy-Paste in die Arbeit."""
    n = aggregates_count
    captions = [
        ("Figure 1",
         "IPASE-Cosine-Heatmap für eine einzelne Person.",
         f"Note. Heatmap der Cosine-Ähnlichkeit zwischen den n Ich-Sätzen einer Person "
         f"(Ordinate) und allen 57 IPASE-Items (Abszisse). Die Items werden in zwei "
         f"Stufen sortiert: erstens nach Faktor (F1 Cognition, F2 Self-awareness, "
         f"F3 Consciousness, F4 Somatization, F5 Demarcation), zweitens innerhalb "
         f"jedes Faktors aufsteigend nach Item-ID. Da die 57 Item-IDs in der IPASE-"
         f"Originalfassung nicht blockweise nach Faktor vergeben wurden, sondern "
         f"durchmischt (z.B. enthält F1 die IDs 1, 12, 20, 28, 37, 49, 56), springen "
         f"die IDs zwischen den Faktoren und am Ende steht F5.45, nicht F1.57 oder "
         f"F5.57. Diese visuell ungleichmäßige ID-Folge ist kein Datenverlust: alle "
         f"57 Items sind in der Heatmap dargestellt. Weiße vertikale Linien markieren "
         f"die Übergänge zwischen den Faktoren. Embedding: paraphrase-multilingual-"
         f"mpnet-base-v2. Farbskala: viridis (color-blind-sicher und grayscale-lesbar)."),
        ("Figure 2",
         "IPASE-Faktor-Profile aller Personen im Polardiagramm.",
         f"Note. Mean Cosine pro IPASE-Faktor (F1 Cognition, F2 Self-awareness, "
         f"F3 Consciousness, F4 Somatization, F5 Demarcation) für jede Person (N = {n}). "
         f"Marker- und Linienstil sind redundant zur Farbe kodiert. "
         f"Embedding: paraphrase-multilingual-mpnet-base-v2."),
        ("Figure 3",
         "Konvergenz der Embedding-Modelle mpnet und multilingual-e5-base.",
         f"Note. Panel (a) Scatter der Mean-Cosine-Werte pro Person für beide Modelle "
         f"mit Identitätslinie. Panel (b) Bland-Altman-Plot: Differenz (e5 minus mpnet) "
         f"gegen Mittelwert beider Modelle. Durchgezogene Linie: mittlere Differenz; "
         f"gestrichelte Linien: ±1.96 SD (Limits of Agreement)."),
        ("Figure 4",
         "Konvergenz semantischer Ähnlichkeit und probabilistischer Plausibilität.",
         f"Note. Mean cosine similarity (mpnet) gegen mean conditional perplexity "
         f"(XGLM-564M) pro Person. Bedingung: P(Ich-Satz | IPASE-Kontext). "
         f"Erwartete Konvergenz: niedrige Perplexität ↔ hohe Cosine. "
         f"Abszisse log-skaliert."),
        ("Figure 5",
         "Top-Match-Verteilung der Ich-Sätze auf die fünf IPASE-Faktoren.",
         f"Note. Pro Ich-Satz wurde das Item mit höchster Cosine-Ähnlichkeit "
         f"(Top-Match) ermittelt. Balken zeigen, wie viele Ich-Sätze pro Person "
         f"auf jeden der fünf Faktoren ihren Top-Match haben. "
         f"Hatch-Patterns sind redundant zur Farbe kodiert."),
        ("Figure 6",
         "Übersicht der Pipeline-Hauptmetriken pro Person.",
         f"Note. Panel (a) Mean cosine similarity (mpnet). "
         f"Panel (b) Mean conditional perplexity (XGLM, log-skaliert). "
         f"Panel (c) Frequenz der Pronomen der ersten Person Singular pro 100 Tokens. "
         f"Panel (d) Mean coherence zwischen konsekutiven Sätzen (mpnet)."),
    ]
    lines = ["# APA-7 Figure Captions\n",
             "Zum Copy-Paste in die Masterarbeit. Pro Figure ein Eintrag mit:\n",
             "- **Figure X** (bold)\n",
             "- *Title* (italics)\n",
             "- *Note.* gefolgt vom Note-Text (italics für „Note.\")\n\n",
             "---\n"]
    for num, title, note in captions:
        lines.append(f"\n## {num}\n\n")
        lines.append(f"**{num}**\n\n")
        lines.append(f"*{title}*\n\n")
        lines.append(f"*Note.* {note}\n")
        lines.append("\n---\n")
    (out_dir / "captions.md").write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="APA-7 Plots für IPASE-Pipeline")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/plots")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"],
                        help="Output-Formate (png, pdf)")
    parser.add_argument("--skip-heatmap", action="store_true")
    parser.add_argument("--skip-radar", action="store_true")
    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument("--skip-topmatch", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    args = parser.parse_args()

    apply_apa_style()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregates = load_aggregates(results_dir)
    mpnet_csv = load_overview(results_dir / "_overview_similarity.csv")
    e5_csv = load_overview(results_dir / "_overview_robustness_embedding.csv")
    pp_csv = load_overview(results_dir / "_overview_robustness_perplexity.csv")

    print(f"Pipeline-Outputs: {len(aggregates)} Personen. Formate: {args.formats}")

    if not args.skip_heatmap:
        print("\n1) Heatmaps pro Person …")
        for csv_file in sorted(results_dir.glob("*.similarity_matrix.csv")):
            pid = csv_file.stem.replace(".similarity_matrix", "")
            matrix, sentences, item_ids, item_factors = load_similarity_matrix(csv_file)
            safe_pid = pid.replace(" ", "_").replace("/", "_")
            base_path = out_dir / f"fig01_heatmap_{safe_pid}"
            plot_heatmap(pid, matrix, sentences, item_ids, item_factors,
                         base_path, args.formats)
            print(f"   → {base_path}.[{'/'.join(args.formats)}]")

    if not args.skip_radar and aggregates:
        print("\n2) Faktor-Profil (Radar) …")
        base_path = out_dir / "fig02_radar_factor_profiles"
        plot_radar(aggregates, base_path, args.formats)
        print(f"   → {base_path}.[{'/'.join(args.formats)}]")

    if not args.skip_convergence:
        print("\n3) Konvergenz mpnet ↔ e5 …")
        base_path = out_dir / "fig03_convergence_embedding"
        ok = plot_embedding_convergence(mpnet_csv, e5_csv, base_path, args.formats)
        if ok:
            print(f"   → {base_path}.[{'/'.join(args.formats)}]")

        print("\n4) Konvergenz Cosine ↔ Perplexität …")
        base_path = out_dir / "fig04_convergence_cosine_perplexity"
        ok = plot_cosine_perplexity(mpnet_csv, pp_csv, base_path, args.formats)
        if ok:
            print(f"   → {base_path}.[{'/'.join(args.formats)}]")

    if not args.skip_topmatch:
        print("\n5) Top-Match-Verteilung …")
        base_path = out_dir / "fig05_topmatch_factor_distribution"
        ok = plot_top_match_factors(results_dir, base_path, args.formats)
        if ok:
            print(f"   → {base_path}.[{'/'.join(args.formats)}]")

    if not args.skip_metrics:
        print("\n6) Personen-Metriken-Übersicht …")
        base_path = out_dir / "fig06_person_metrics_overview"
        ok = plot_person_metrics_bar(results_dir / "_stats_combined_table.csv",
                                     base_path, args.formats)
        if ok:
            print(f"   → {base_path}.[{'/'.join(args.formats)}]")

    print("\n7) APA-7-Captions …")
    write_captions(out_dir, len(aggregates))
    print(f"   → {out_dir / 'captions.md'}")

    print(f"\nAlle Outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
