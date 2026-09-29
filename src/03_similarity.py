"""S-BERT-Ähnlichkeit zwischen „Ich"-Sätzen und IPASE-Items.

Liest:
  - data/ipase_items.json (57 IPASE-Items mit Faktor-Zuordnung)
  - data/processed/*.ich_saetze.json (Output von 02_preprocess.py)

Schreibt:
  - results/<person>.similarity_matrix.csv (n_ich_saetze × 57 Kosinus-Werte)
  - results/<person>.aggregates.json (Mean, Median, Max, P95 + Faktor-Sub-Indizes)
  - results/_overview_similarity.csv (eine Zeile pro Person mit Hauptkennzahlen)
  - results/<person>.top_matches.json (für jeden Ich-Satz: das ähnlichste IPASE-Item)

Verwendung:
    python src/03_similarity.py
    python src/03_similarity.py --model BAAI/bge-m3   # Robustheits-Check
"""
import os
# Windows/conda: doppelten OpenMP-Runtime (MKL + libiomp) tolerieren, vor torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import csv
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Atomar schreiben via .tmp + os.replace() — kein korruptes JSON bei Kill."""
    import os as _os
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding=encoding)
    _os.replace(tmp, path)


def cosine_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between two sets of embeddings (rows × rows)."""
    a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
    b = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)
    return a @ b.T


def aggregate(sim_matrix: np.ndarray, items: list[dict],
              sim_control: np.ndarray | None = None) -> dict:
    """Berechnet Personen-Aggregate aus der n_sents × 57 Ähnlichkeitsmatrix.

    Neben dem Mittelwert über die Gesamtmatrix (Pilotbefund: flach über Personen,
    konvergiert gegen die Baseline des Embedding-Raums ~.29) werden Aggregate
    über das Satz-Maximum berechnet: Für jeden Ich-Satz zählt nur das ähnlichste
    IPASE-Item (ein Satz drückt höchstens EINE spezifische Anomalie aus, nicht
    alle 57). Tail-Raten = Anteil der Sätze, deren bestes Item ≥ Schwelle liegt.
    """
    flat = sim_matrix.flatten()
    per_sent_max = sim_matrix.max(axis=1)  # bestes Item pro Ich-Satz
    m = sim_matrix.shape[0]
    out = {
        "n_ich_saetze": int(m),
        "n_ipase_items": int(sim_matrix.shape[1]),
        "mean_overall": float(flat.mean()),
        "median_overall": float(np.median(flat)),
        "max_overall": float(flat.max()),
        "p95_overall": float(np.percentile(flat, 95)),
        # Satz-Max-basierte Aggregate (weniger Verdünnung durch 57-Item-Mittelung)
        "mean_sent_max": float(per_sent_max.mean()),
        "median_sent_max": float(np.median(per_sent_max)),
        "p90_sent_max": float(np.percentile(per_sent_max, 90)),
        # Tail-Raten: Anteil Ich-Sätze mit bestem Item ≥ τ (in %)
        "tail_rate_50": float((per_sent_max >= 0.50).mean() * 100),
        "tail_rate_55": float((per_sent_max >= 0.55).mean() * 100),
        "tail_rate_60": float((per_sent_max >= 0.60).mean() * 100),
    }
    # Kontrast-Baseline (P2): Δ = bestes IPASE-Item − bestes Kontroll-Item pro Satz.
    # Thematische Scheintreffer (referenzielle Drift) matchen die Kontrolle ähnlich
    # hoch und fallen aus Δ heraus; echte Anomalie-Nähe bleibt positiv.
    if sim_control is not None and sim_control.shape[0] == m:
        per_sent_ctrl_max = sim_control.max(axis=1)
        delta = per_sent_max - per_sent_ctrl_max
        high = per_sent_max >= 0.50
        out.update({
            "n_control_items": int(sim_control.shape[1]),
            "mean_delta_sent_max": float(delta.mean()),
            "median_delta_sent_max": float(np.median(delta)),
            "p90_delta_sent_max": float(np.percentile(delta, 90)),
            # spezifische Treffer: hoch UND deutlich über der Topic-Kontrolle
            "specific_hit_rate": float(((per_sent_max >= 0.50) & (delta >= 0.10)).mean() * 100),
            # Drift-Diagnostik: Anteil der High-Cosine-Treffer, die die Kontrolle
            # genauso gut erklärt (Δ < .05) — reine Themen-Artefakte
            "drift_share_of_hits": (
                float((delta[high] < 0.05).mean() * 100) if high.any() else float("nan")
            ),
        })
    # Faktor-Sub-Indizes (Mean-Ähnlichkeit zu den Items des jeweiligen Faktors)
    factor_subindices = {}
    for fid in (1, 2, 3, 4, 5):
        cols = [i for i, it in enumerate(items) if it["factor"] == fid]
        sub = sim_matrix[:, cols]
        factor_subindices[str(fid)] = {
            "mean": float(sub.mean()),
            "median": float(np.median(sub)),
            "n_items": len(cols),
        }
    out["factor_subindices"] = factor_subindices
    return out


def main():
    parser = argparse.ArgumentParser(description="S-BERT-Ähnlichkeit zu IPASE-Items")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"sentence-transformers Modell (default: {DEFAULT_MODEL})")
    parser.add_argument("--ipase", default="data/ipase_items.json",
                        help="Pfad zu ipase_items.json")
    parser.add_argument("--controls", default="data/control_items.json",
                        help="Pfad zu control_items.json (Kontrast-Baseline, P2); "
                             "fehlt die Datei, läuft alles ohne Kontrolle weiter")
    parser.add_argument("--processed", default="data/processed",
                        help="Ordner mit *.ich_saetze.json")
    parser.add_argument("--out", default="results",
                        help="Output-Ordner")
    args = parser.parse_args()

    ipase_path = Path(args.ipase)
    processed_dir = Path(args.processed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # IPASE-Items laden
    ipase = json.loads(ipase_path.read_text(encoding="utf-8"))
    items = sorted(ipase["items"], key=lambda x: x["id"])
    factors = ipase["factors"]
    item_texts = [it["text"] for it in items]
    print(f"IPASE-Items: {len(items)} (über {len(factors)} Faktoren)")

    # Modell laden
    print(f"Lade S-BERT-Modell: {args.model}")
    model = SentenceTransformer(args.model)
    print(f"Modell geladen (Embedding-Dim: {model.get_sentence_embedding_dimension()})")

    # IPASE-Items embedden (einmal)
    print("Embedde IPASE-Items...")
    item_emb = model.encode(item_texts, convert_to_numpy=True,
                            show_progress_bar=False, normalize_embeddings=False)
    print(f"  Shape: {item_emb.shape}")

    # Kontroll-Items (Kontrast-Baseline P2), optional
    control_emb = None
    control_items = None
    controls_path = Path(args.controls)
    if controls_path.exists():
        controls = json.loads(controls_path.read_text(encoding="utf-8"))
        control_items = controls["items"]
        print(f"Kontroll-Items: {len(control_items)} (Kontrast-Baseline aktiv)")
        control_emb = model.encode([c["text"] for c in control_items],
                                   convert_to_numpy=True,
                                   show_progress_bar=False,
                                   normalize_embeddings=False)
    else:
        print(f"Keine Kontroll-Items unter {controls_path} — Kontrast-Baseline übersprungen.")

    # Personen-Files finden
    files = sorted(processed_dir.glob("*.ich_saetze.json"))
    if not files:
        print(f"\nKeine *.ich_saetze.json in {processed_dir}/. Erst 02_preprocess.py laufen lassen.")
        return

    overview_rows = []

    for f in files:
        person_id = f.stem.replace(".ich_saetze", "")
        print(f"\n→ {person_id}")
        data = json.loads(f.read_text(encoding="utf-8"))
        ich_saetze = data["ich_saetze"]
        if not ich_saetze:
            print("  Keine Ich-Sätze, überspringe.")
            continue
        sentences = [s["text"] for s in ich_saetze]

        # Sätze embedden
        sent_emb = model.encode(sentences, convert_to_numpy=True,
                                show_progress_bar=False, normalize_embeddings=False)
        sim = cosine_matrix(sent_emb, item_emb)  # n_sents × 57
        print(f"  Matrix: {sim.shape}, Mean cosine: {sim.mean():.4f}")

        sim_ctrl = None
        if control_emb is not None:
            sim_ctrl = cosine_matrix(sent_emb, control_emb)  # n_sents × n_controls

        # Save matrix as CSV
        matrix_path = out_dir / f"{person_id}.similarity_matrix.csv"
        with open(matrix_path, "w", encoding="utf-8", newline="") as cf:
            w = csv.writer(cf)
            header = ["satz_idx", "satz_text"] + [f"item_{it['id']}_f{it['factor']}" for it in items]
            w.writerow(header)
            for i, s in enumerate(ich_saetze):
                row = [s["idx"], s["text"]] + [f"{v:.4f}" for v in sim[i]]
                w.writerow(row)

        # Aggregates
        agg = aggregate(sim, items, sim_control=sim_ctrl)
        agg["person_id"] = person_id
        agg["model"] = args.model
        agg_path = out_dir / f"{person_id}.aggregates.json"
        _atomic_write_text(agg_path, json.dumps(agg, ensure_ascii=False, indent=2))

        # Top matches: für jeden Ich-Satz, was ist das ähnlichste IPASE-Item?
        top_matches = []
        for i, s in enumerate(ich_saetze):
            best_idx = int(sim[i].argmax())
            best_item = items[best_idx]
            entry = {
                "satz": s["text"],
                "best_match_item_id": best_item["id"],
                "best_match_item_text": best_item["text"],
                "best_match_factor": factors[str(best_item["factor"])],
                "best_match_score": float(sim[i, best_idx]),
            }
            if sim_ctrl is not None:
                ctrl_idx = int(sim_ctrl[i].argmax())
                ctrl_score = float(sim_ctrl[i, ctrl_idx])
                entry.update({
                    "best_control_topic": control_items[ctrl_idx]["topic"],
                    "best_control_score": ctrl_score,
                    "delta": float(sim[i, best_idx]) - ctrl_score,
                })
            top_matches.append(entry)
        top_path = out_dir / f"{person_id}.top_matches.json"
        _atomic_write_text(top_path, json.dumps(top_matches, ensure_ascii=False, indent=2))

        # Console-Report
        print(f"  IPASE-Similarity-Index (Mean): {agg['mean_overall']:.4f}")
        for fid in (1, 2, 3, 4, 5):
            sub = agg["factor_subindices"][str(fid)]
            print(f"    Faktor {fid} ({factors[str(fid)]}): {sub['mean']:.4f}")

        # Overview row
        row = {
            "person_id": person_id,
            "n_ich_saetze": agg["n_ich_saetze"],
            "mean": agg["mean_overall"],
            "median": agg["median_overall"],
            "p95": agg["p95_overall"],
            "max": agg["max_overall"],
            "mean_sent_max": agg["mean_sent_max"],
            "p90_sent_max": agg["p90_sent_max"],
            "tail_rate_50": agg["tail_rate_50"],
            "tail_rate_55": agg["tail_rate_55"],
            "tail_rate_60": agg["tail_rate_60"],
            "mean_delta_sent_max": agg.get("mean_delta_sent_max", ""),
            "p90_delta_sent_max": agg.get("p90_delta_sent_max", ""),
            "specific_hit_rate": agg.get("specific_hit_rate", ""),
            "drift_share_of_hits": agg.get("drift_share_of_hits", ""),
        }
        for fid in (1, 2, 3, 4, 5):
            row[f"factor_{fid}_mean"] = agg["factor_subindices"][str(fid)]["mean"]
        overview_rows.append(row)

    # Aggregate overview CSV
    overview_path = out_dir / "_overview_similarity.csv"
    if overview_rows:
        with open(overview_path, "w", encoding="utf-8", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=list(overview_rows[0].keys()))
            w.writeheader()
            w.writerows(overview_rows)
        print(f"\nÜbersicht: {overview_path}")

    print("\nFertig.")


if __name__ == "__main__":
    main()
