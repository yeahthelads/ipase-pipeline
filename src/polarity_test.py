"""Polaritäts-Test: löst S-BERT-Cosine oder LeoLM-Perplexität polare Inversionen auf?

M.4.6 offene Frage: Thematische Nähe vs. polare Korrektheit — erkennen unsere Maße
den Unterschied zwischen einem Satz, der eine anomale Erfahrung bejaht (kongruent),
und einem, der dieselbe Erfahrung verneint (polar)?

Verwendung:
    python src/polarity_test.py
    python src/polarity_test.py --skip-perplexity   # nur S-BERT
    python src/polarity_test.py --gguf models/leo-hessianai-7b.Q4_K_M.gguf
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# CUDA-DLLs für llama-cpp-python (muss vor llama_cpp-Import stehen)
try:
    import torch as _torch
    _tl = os.path.join(os.path.dirname(_torch.__file__), "lib")
    if os.path.isdir(_tl):
        os.add_dll_directory(_tl)
except Exception:
    pass

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

SBERT_MODEL       = "paraphrase-multilingual-mpnet-base-v2"
LEOLM_GGUF_DEFAULT = "models/leo-hessianai-7b.Q4_K_M.gguf"

# ── Satzpaare ────────────────────────────────────────────────────────────────
# Je Paar: ein kongruenter Satz (bejaht die anomale Erfahrung) und
# ein polarer Satz (verneint dieselbe Erfahrung).
# Zugeordnet zu einer IPASE-Dimension (für Lesbarkeit).
PAIRS = [
    # Vom User spezifiziert
    {
        "dimension": "Selbstentfremdung",
        "kongruent": "In letzter Zeit fühle ich mich fremd in mir selbst, wie eine andere Person.",
        "polar":     "In letzter Zeit fühle ich mich so vertraut und zuhause in mir selbst wie nie.",
    },
    {
        "dimension": "Selbst-Kontakt",
        "kongruent": "Ich habe den Kontakt zu mir selbst völlig verloren.",
        "polar":     "Ich habe endlich wieder vollen Kontakt zu mir selbst.",
    },
    {
        "dimension": "Grenze zur Welt",
        "kongruent": "Die Grenze zwischen mir und der Welt hat sich aufgelöst.",
        "polar":     "Die Grenze zwischen mir und der Welt ist so klar und fest wie immer.",
    },
    {
        "dimension": "Gedanken-Herkunft",
        "kongruent": "Meine Gedanken werden von jemand anderem gemacht, nicht von mir.",
        "polar":     "Meine Gedanken sind ganz meine eigenen, niemand mischt sich ein.",
    },
    {
        "dimension": "Affektabflachung",
        "kongruent": "Ich fühle viel weniger stark als früher, alles ist abgeflacht.",
        "polar":     "Ich fühle wieder so stark und lebendig wie früher.",
    },
    {
        "dimension": "Urheberschaft",
        "kongruent": "Wenn ich etwas tue, ist es, als würde nicht wirklich ich es tun.",
        "polar":     "Wenn ich etwas tue, bin ganz klar ich selbst es, der handelt.",
    },
    # Ergänzte Paare
    {
        "dimension": "Körpergrenzen",
        "kongruent": "Manchmal spüre ich nicht mehr, wo mein Körper aufhört und die Welt anfängt.",
        "polar":     "Ich spüre meinen Körper klar und deutlich als meinen eigenen, abgegrenzt von der Welt.",
    },
    {
        "dimension": "Zeiterleben",
        "kongruent": "Die Zeit steht für mich still, ich komme aus dem Moment nicht heraus.",
        "polar":     "Die Zeit fließt für mich ganz normal, ich erlebe Vergangenheit und Zukunft wie gewohnt.",
    },
    {
        "dimension": "Hyperreflexivität",
        "kongruent": "Ich beobachte mich selbst ständig von außen, als wäre ich ein Fremder für mich.",
        "polar":     "Ich lebe ganz selbstverständlich in mir, ohne mich ständig von außen zu beobachten.",
    },
    {
        "dimension": "Gedankenlautwerden",
        "kongruent": "Manchmal höre ich meine eigenen Gedanken laut, als kämen sie von außen.",
        "polar":     "Meine Gedanken bleiben für mich innerlich und still, ganz wie immer.",
    },
    {
        "dimension": "Derealisation",
        "kongruent": "Die Dinge um mich herum wirken unwirklich, wie hinter einer Glasscheibe.",
        "polar":     "Die Dinge um mich herum wirken lebendig und wirklich, ganz wie immer.",
    },
    {
        "dimension": "Ich-Auflösung",
        "kongruent": "Manchmal habe ich das Gefühl, dass mein Ich sich auflöst und verschwindet.",
        "polar":     "Mein Ich ist fest und stabil, ich bin mir meiner selbst immer klar bewusst.",
    },
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def run_sbert(ipase_items: list[dict], pairs: list[dict]) -> list[dict]:
    print(f"\n[S-BERT] Lade {SBERT_MODEL} ...")
    model = SentenceTransformer(SBERT_MODEL)
    item_texts = [it["text"] for it in ipase_items]
    item_emb   = model.encode(item_texts, convert_to_numpy=True, show_progress_bar=False)
    # L2-Normalisierung für schnelle Dot-Product-Cosine
    item_emb  /= np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-10

    results = []
    for p in pairs:
        emb_k = model.encode(p["kongruent"], convert_to_numpy=True)
        emb_p = model.encode(p["polar"],     convert_to_numpy=True)
        emb_k /= np.linalg.norm(emb_k) + 1e-10
        emb_p /= np.linalg.norm(emb_p) + 1e-10

        # Best-match (max über alle Items) — analog zu top_matches in 03_similarity.py
        sims_k = emb_k @ item_emb.T
        sims_p = emb_p @ item_emb.T
        best_k = float(sims_k.max())
        best_p = float(sims_p.max())
        mean_k = float(sims_k.mean())
        mean_p = float(sims_p.mean())

        results.append({
            "dimension":       p["dimension"],
            "cos_max_k":       best_k,
            "cos_max_p":       best_p,
            "cos_mean_k":      mean_k,
            "cos_mean_p":      mean_p,
            "delta_cos_max":   best_k - best_p,
            "delta_cos_mean":  mean_k - mean_p,
        })
        print(f"  {p['dimension']}: max-cos k={best_k:.4f}  p={best_p:.4f}  Δ={best_k-best_p:+.4f}")
    return results


def _mean_logprob(llm, context: str, target: str, ctx_tokens: int) -> float:
    full = context + " " + target
    try:
        out = llm.create_completion(full, max_tokens=1, echo=True,
                                     logprobs=True, temperature=0.0)
        lps = out["choices"][0]["logprobs"]["token_logprobs"]
        tgt = [lp for lp in lps[ctx_tokens:] if lp is not None]
        return float(np.mean(tgt)) if tgt else float("nan")
    except Exception as e:
        print(f"    [warn] {e}")
        return float("nan")


def run_leolm(gguf_path: str, ipase_items: list[dict],
              pairs: list[dict], n_gpu_layers: int = -1) -> list[dict]:
    print(f"\n[LeoLM] Lade {gguf_path} (n_gpu_layers={n_gpu_layers}) ...")
    try:
        from llama_cpp import Llama
    except ImportError:
        print("  llama-cpp-python nicht installiert — überspringe Perplexität.")
        return []

    gguf = Path(gguf_path)
    if not gguf.exists():
        print(f"  GGUF nicht gefunden: {gguf} — überspringe Perplexität.")
        return []

    llm = Llama(model_path=str(gguf), n_ctx=2048,
                n_gpu_layers=n_gpu_layers, verbose=False, logits_all=True)
    context       = " ".join(it["text"] for it in ipase_items)
    ctx_tokens    = len(llm.tokenize(context.encode("utf-8")))
    print(f"  Kontext: {ctx_tokens} Tokens")

    results = []
    for p in pairs:
        lp_k = _mean_logprob(llm, context, p["kongruent"], ctx_tokens)
        lp_p = _mean_logprob(llm, context, p["polar"],     ctx_tokens)
        ppl_k = math.exp(-lp_k) if not math.isnan(lp_k) else float("nan")
        ppl_p = math.exp(-lp_p) if not math.isnan(lp_p) else float("nan")
        delta = ppl_p - ppl_k  # positiv = polar schwerer (gut)
        results.append({
            "dimension": p["dimension"],
            "ppl_k":     ppl_k,
            "ppl_p":     ppl_p,
            "delta_ppl": delta,
        })
        print(f"  {p['dimension']}: PPL k={ppl_k:.2f}  p={ppl_p:.2f}  Δ={delta:+.2f}")
    return results


def print_table(sbert_rows: list[dict], leolm_rows: list[dict]) -> None:
    leolm_by_dim = {r["dimension"]: r for r in leolm_rows}

    sep  = "─" * 110
    hdr  = (f"{'Dimension':<22} │ {'cos_k':>7} {'cos_p':>7} {'Δcos':>7} │ "
            f"{'ppl_k':>7} {'ppl_p':>7} {'Δppl':>7} │ Fazit")
    print(f"\n{sep}")
    print(hdr)
    print(sep)

    n_cos_ok  = 0   # kongruent hat höheren Cosine
    n_ppl_ok  = 0   # polar hat höhere PPL (= kongruent leichter)
    n_both_ok = 0

    for r in sbert_rows:
        dim  = r["dimension"]
        ck   = r["cos_mean_k"]
        cp   = r["cos_mean_p"]
        dc   = r["delta_cos_mean"]
        lr   = leolm_by_dim.get(dim, {})
        pk   = lr.get("ppl_k",   float("nan"))
        pp   = lr.get("ppl_p",   float("nan"))
        dp   = lr.get("delta_ppl", float("nan"))

        cos_ok  = dc > 0.005
        ppl_ok  = (not math.isnan(dp)) and dp > 1.0
        n_cos_ok  += int(cos_ok)
        n_ppl_ok  += int(ppl_ok)
        n_both_ok += int(cos_ok and ppl_ok)

        if cos_ok and ppl_ok:
            verdict = "BEIDE trennen"
        elif cos_ok:
            verdict = "nur Cosine"
        elif ppl_ok:
            verdict = "nur PPL"
        else:
            verdict = "keines trennt"

        ppl_k_s = f"{pk:7.2f}" if not math.isnan(pk) else "    n/a"
        ppl_p_s = f"{pp:7.2f}" if not math.isnan(pp) else "    n/a"
        dp_s    = f"{dp:+7.2f}" if not math.isnan(dp) else "    n/a"

        print(f"{dim:<22} │ {ck:7.4f} {cp:7.4f} {dc:+7.4f} │ "
              f"{ppl_k_s} {ppl_p_s} {dp_s} │ {verdict}")

    n = len(sbert_rows)
    print(sep)
    print(f"{'GESAMT':<22} │ {'':>7} {'':>7} {f'{n_cos_ok}/{n}':>7} │ "
          f"{'':>7} {'':>7} {f'{n_ppl_ok}/{n}':>7} │ beide: {n_both_ok}/{n}")
    print(sep)

    print("\nFazit:")
    print(f"  S-BERT-Cosine trennt Polarität in {n_cos_ok}/{n} Paaren (Δ > 0.005).")
    print(f"  LeoLM-PPL   trennt Polarität in {n_ppl_ok}/{n} Paaren (Δ > 1.0 PPL-Punkte).")
    if n_both_ok == n:
        print("  → Beide Maße lösen polare Inversionen konsistent auf.")
    elif n_cos_ok > n_ppl_ok:
        print("  → S-BERT-Cosine ist robuster gegenüber polarer Inversion als LeoLM-PPL.")
        print("    Cosine misst thematische Nähe UND wertet Polarität korrekt ab.")
        print("    PPL zeigt stärkeren thematischen Bias (ähnlich leicht für beide Polaritäten).")
    elif n_ppl_ok > n_cos_ok:
        print("  → LeoLM-PPL trennt besser: polarer Satz hat deutlich höhere Perplexität.")
        print("    Cosine leidet stärker unter thematischem Bias.")
    else:
        print("  → Gemischtes Bild — beide Maße haben ähnliche Stärken bei Polaritätstrennung.")


def main():
    parser = argparse.ArgumentParser(description="Polaritäts-Test für IPASE-Maße")
    parser.add_argument("--ipase",          default="data/ipase_items.json")
    parser.add_argument("--gguf",           default=LEOLM_GGUF_DEFAULT)
    parser.add_argument("--n-gpu-layers",   type=int, default=-1)
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--out",            default=None,
                        help="Optionale JSON-Ausgabedatei für Rohergebnisse")
    args = parser.parse_args()

    ipase = json.loads(Path(args.ipase).read_text(encoding="utf-8"))
    items = sorted(ipase["items"], key=lambda x: x["id"])
    print(f"IPASE-Items: {len(items)}")

    sbert_rows = run_sbert(items, PAIRS)

    leolm_rows = []
    if not args.skip_perplexity:
        leolm_rows = run_leolm(args.gguf, items, PAIRS, n_gpu_layers=args.n_gpu_layers)

    print_table(sbert_rows, leolm_rows)

    if args.out:
        out = {"sbert": sbert_rows, "leolm": leolm_rows}
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"\nRohergebnisse: {args.out}")


if __name__ == "__main__":
    main()
