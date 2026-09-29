#!/usr/bin/env python3
"""Graphbasierte Sprach-Connectedness nach Mota et al. (2012, 2014).

Ergaenzung zu 05_features.py. Berechnet pro Person die Konnektivitaetsmasse des
Wortgraphen ueber ein gleitendes Fenster fester Laenge (Default 30 Woerter,
Schrittweite 1) und mittelt ueber alle Fenster.

Die feste Fensterlaenge ist der Kern des Verfahrens und nicht optional: LCC, LSC
und ATD wachsen monoton mit der Textlaenge. Ein Graph ueber das Gesamttranskript
waere zwischen Faellen unterschiedlicher Laenge nicht vergleichbar. Die
Gesamttranskript-Werte werden trotzdem als *_full mitgeschrieben, ausdruecklich
als laengenkonfundierte Vergleichsspalten.

Graphdefinition (Mota et al. 2012, PLoS ONE 7:e34928):
  Knoten = Wort, gerichtete Kante = aufeinanderfolgendes Wortpaar,
  Mehrfachkanten bleiben erhalten.

Masse pro Fenster:
  N     Knotenzahl (Types im Fenster)
  E     Kantenzahl inkl. Mehrfachkanten
  PE    parallele Kanten (E minus eindeutige Kanten) = wiederholte Bigramme
  RE    Zahl der Kanten, die mindestens einmal wiederholt auftreten
  ATD   mittlerer Gesamtgrad = 2E/N
  LCC   Knoten in der groessten (schwach) zusammenhaengenden Komponente
  LSC   Knoten in der groessten stark zusammenhaengenden Komponente
  L1    Schleifen der Laenge 1 (Wortwiederholung direkt hintereinander)
  L2    Schleifen der Laenge 2 (Wortpaar in beiden Richtungen)
  L3    gerichtete Dreiecke
  density, ASP (mittlere kuerzeste Pfadlaenge in der LCC), CC (Clustering)

ASP und CC kosten etwa das Neunfache der uebrigen Masse. Fuer schnelle
Durchlaeufe --no-extended setzen; sie sind bei Mota Sekundaermasse.

Wichtig fuer die Auswertung: n_words unbedingt als Kovariate mitfuehren.
Alogie verkuerzt das Transkript, und kurze Transkripte liefern weniger Fenster.
Ohne Kontrolle von n_words ist jeder Zusammenhang mit SANS oder PANSS-N
zweideutig.

Liest:
  - data/transcripts/*.txt   (Patient-only-Transkripte aus 01b_diarize_transcribe.py)

Schreibt:
  - results/_overview_graph.csv           Personenebene, eine Zeile pro Fall
  - results/_graph_windows/<id>.csv       Fensterebene, nur mit --save-windows

Verwendung:
    python src/09_graph.py --jobs 8
    python src/09_graph.py --jobs 8 --lemma --out results_lemma
    python src/09_graph.py --jobs 8 --no-extended          # schneller Vorlauf
    python src/09_graph.py --tokenizer regex               # ganz ohne spaCy
"""
import argparse
import csv
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import networkx as nx

MEASURES = ["N", "E", "PE", "RE", "ATD", "LCC", "LSC",
            "L1", "L2", "L3", "density", "ASP", "CC"]

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

_NLP = None
_CFG = None


# ---------------------------------------------------------------- Tokenisierung

def load_nlp(tokenizer, spacy_model, use_lemma):
    """Nur so viel spaCy laden wie noetig.

    Wortformen brauchen ausschliesslich den Tokenizer; spacy.blank("de") nutzt
    dieselben Tokenisierungsregeln wie de_core_news_lg, laedt aber kein
    statistisches Modell. Lemmata brauchen Tagger und AttributeRuler, aber
    weder Parser noch NER.
    """
    if tokenizer == "regex" and not use_lemma:
        return None
    import spacy
    if use_lemma:
        nlp = spacy.load(spacy_model, disable=["parser", "ner"])
    else:
        nlp = spacy.blank("de")
    nlp.max_length = 20_000_000
    return nlp


def tokenize(text, nlp, use_lemma=False, keep_stopwords=True):
    if nlp is None:
        words = [w.lower() for w in WORD_RE.findall(text)]
        return words
    out = []
    for t in nlp(text):
        if t.is_space or t.is_punct or not t.is_alpha:
            continue
        if not keep_stopwords and t.is_stop:
            continue
        w = (t.lemma_ if use_lemma else t.text).lower().strip()
        if w:
            out.append(w)
    return out


# ------------------------------------------------------------------- Graphmasse

def window_measures(words, extended=True):
    """Mota-Masse fuer eine Wortfolge (ein Fenster oder ein ganzes Transkript)."""
    n_w = len(words)
    if n_w < 2:
        return None

    edges = [(words[i], words[i + 1]) for i in range(n_w - 1)]
    ec = Counter(edges)
    E = len(edges)
    PE = E - len(ec)
    RE = sum(1 for c in ec.values() if c > 1)

    nodes = set(words)
    N = len(nodes)

    L1 = sum(1 for (u, v) in ec if u == v)
    simple = {(u, v) for (u, v) in ec if u != v}
    L2 = sum(1 for (u, v) in simple if u < v and (v, u) in simple)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(simple)

    succ = {u: set(G.successors(u)) for u in G.nodes}
    tri = 0
    for u in G.nodes:
        for v in succ[u]:
            for w in succ[v]:
                if w != u and w != v and u in succ[w]:
                    tri += 1
    L3 = tri // 3

    LSC = max((len(c) for c in nx.strongly_connected_components(G)), default=0)
    U = G.to_undirected()
    comps = list(nx.connected_components(U)) if N else []
    LCC = max((len(c) for c in comps), default=0)

    out = {
        "N": float(N), "E": float(E), "PE": float(PE), "RE": float(RE),
        "ATD": 2.0 * E / N if N else float("nan"),
        "LCC": float(LCC), "LSC": float(LSC),
        "L1": float(L1), "L2": float(L2), "L3": float(L3),
        "density": nx.density(G) if N > 1 else float("nan"),
    }

    if extended and LCC > 1:
        sub = U.subgraph(max(comps, key=len))
        out["ASP"] = nx.average_shortest_path_length(sub)
        out["CC"] = nx.average_clustering(sub)
    else:
        out["ASP"] = float("nan")
        out["CC"] = float("nan")

    return out


def sliding(words, size, step=1):
    for i in range(0, max(0, len(words) - size + 1), step):
        yield words[i:i + size]


# ------------------------------------------------------- Verarbeitung pro Fall

def process_file(path):
    """Einen Fall verarbeiten. Nutzt die Prozess-globalen _NLP und _CFG."""
    cfg = _CFG
    pid = Path(path).stem
    text = Path(path).read_text(encoding="utf-8").strip()
    words = tokenize(text, _NLP, use_lemma=cfg["lemma"],
                     keep_stopwords=not cfg["no_stopwords"])

    row = {
        "person_id": pid,
        "n_words": len(words),
        "n_types": len(set(words)),
        "window_size": cfg["window"],
        "step": cfg["step"],
        "tokenizer": "lemma" if cfg["lemma"] else cfg["tokenizer"],
        "stopwords": "removed" if cfg["no_stopwords"] else "kept",
    }

    wins = [w for w in (window_measures(x, extended=cfg["extended"])
                        for x in sliding(words, cfg["window"], cfg["step"]))
            if w is not None]
    row["n_windows"] = len(wins)

    for m in MEASURES:
        if wins:
            vals = np.array([w[m] for w in wins], dtype=float)
            row[f"{m}_mean"] = float(np.nanmean(vals))
            row[f"{m}_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        else:
            row[f"{m}_mean"] = float("nan")
            row[f"{m}_sd"] = float("nan")

    full = window_measures(words, extended=cfg["extended"])
    for m in MEASURES:
        row[f"{m}_full"] = float(full[m]) if full else float("nan")

    windows_out = None
    if cfg["save_windows"] and wins:
        windows_out = (pid, wins)
    return row, windows_out


def init_worker(cfg):
    global _NLP, _CFG
    _CFG = cfg
    _NLP = load_nlp(cfg["tokenizer"], cfg["spacy_model"], cfg["lemma"])


# ------------------------------------------------------------------------ Main

def main():
    p = argparse.ArgumentParser(
        description="Graphbasierte Connectedness nach Mota et al. (2012, 2014)")
    p.add_argument("--transcripts", default="data/transcripts")
    p.add_argument("--out", default="results")
    p.add_argument("--window", type=int, default=30,
                   help="Fenstergroesse in Woertern (Mota-Default 30)")
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--tokenizer", choices=["spacy", "regex"], default="spacy")
    p.add_argument("--spacy-model", default="de_core_news_lg")
    p.add_argument("--lemma", action="store_true",
                   help="Lemmata statt Wortformen (Sensitivitaet, laedt Tagger)")
    p.add_argument("--no-stopwords", action="store_true",
                   help="Funktionswoerter entfernen (Sensitivitaet; Mota behaelt sie)")
    p.add_argument("--no-extended", action="store_true",
                   help="ASP und CC ueberspringen (etwa neunmal schneller)")
    p.add_argument("--save-windows", action="store_true",
                   help="Fensterebene je Fall zusaetzlich schreiben")
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallele Prozesse ueber Faelle (PC-75: 8 bis 16 sinnvoll)")
    args = p.parse_args()

    transcript_dir = Path(args.transcripts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "window": args.window, "step": args.step, "tokenizer": args.tokenizer,
        "spacy_model": args.spacy_model, "lemma": args.lemma,
        "no_stopwords": args.no_stopwords, "extended": not args.no_extended,
        "save_windows": args.save_windows,
    }

    txt_files = sorted(str(p) for p in transcript_dir.glob("*.txt"))
    if not txt_files:
        sys.exit(f"Keine Transkripte in {transcript_dir}")

    print(f"{len(txt_files)} Transkripte, Fenster {args.window}, "
          f"extended={cfg['extended']}, jobs={args.jobs}")
    t0 = time.perf_counter()

    results = []
    if args.jobs <= 1:
        init_worker(cfg)
        for f in txt_files:
            results.append(process_file(f))
            r = results[-1][0]
            print(f"  {r['person_id']}: {r['n_words']} Woerter, "
                  f"{r['n_windows']} Fenster, LCC={r['LCC_mean']:.2f} "
                  f"LSC={r['LSC_mean']:.2f} ATD={r['ATD_mean']:.2f}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 initializer=init_worker,
                                 initargs=(cfg,)) as ex:
            for res in ex.map(process_file, txt_files):
                results.append(res)
                r = res[0]
                print(f"  {r['person_id']}: {r['n_words']} Woerter, "
                      f"{r['n_windows']} Fenster, LCC={r['LCC_mean']:.2f} "
                      f"LSC={r['LSC_mean']:.2f} ATD={r['ATD_mean']:.2f}")

    rows = [r for r, _ in results]
    rows.sort(key=lambda r: r["person_id"])

    if args.save_windows:
        wdir = out_dir / "_graph_windows"
        wdir.mkdir(parents=True, exist_ok=True)
        for _, w in results:
            if not w:
                continue
            pid, wins = w
            with (wdir / f"{pid}.csv").open("w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=["window_index"] + MEASURES)
                wr.writeheader()
                for i, win in enumerate(wins):
                    wr.writerow({"window_index": i, **win})

    out_csv = out_dir / "_overview_graph.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    dt = time.perf_counter() - t0
    total_windows = sum(r["n_windows"] for r in rows)
    print(f"\nGeschrieben: {out_csv}  ({len(rows)} Faelle, "
          f"{total_windows:,} Fenster, {dt/60:.1f} min)")

    short = [(r["person_id"], r["n_words"]) for r in rows if r["n_windows"] == 0]
    if short:
        print("\nWARNUNG: zu kurz fuer die Fensterbreite, Fenstermasse NaN:")
        for pid, n in short:
            print(f"  {pid}: {n} Woerter < {args.window}")
    print("\nHinweis: n_words in jeder Korrelation als Kovariate mitfuehren "
          "(Alogie verkuerzt das Transkript).")


if __name__ == "__main__":
    main()
