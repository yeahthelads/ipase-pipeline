"""Robustheits-Companion und Perplexitäts-Triangulation.

Zwei unabhängige methodologische Vergleichsachsen zum S-BERT-IPASE-Index aus 03_similarity.py:

A) Zweites Embedding-Modell (multilingual-e5-base) → alternative Cosine-Werte.
   Konvergente Cosine zwischen mpnet und e5 → Robustheit gegen modell-spezifische Artefakte.

B) Perplexitäts-IPASE-Alignment mit LeoLM-7B-GGUF (Deutsch-spezialisiertes LLM, M.4.6).
   Niedrige bedingte Perplexität P(„Ich"-Satz | IPASE-Items) = gutes probabilistisches Alignment.
   Hohe Perplexität trotz hoher Cosine = thematische, nicht polare Nähe.
   Modell: LeoLM/leo-hessianai-7b Q4_K_M via llama-cpp-python (CUDA-Offloading).

Verwendung:
    python src/04_robustness.py
    python src/04_robustness.py --skip-perplexity      # nur e5-Embedding
    python src/04_robustness.py --skip-embedding       # nur Perplexität
    python src/04_robustness.py --gguf models/leo-hessianai-7b.Q4_K_M.gguf
"""
import os
# Windows/conda: doppelten OpenMP-Runtime (MKL + libiomp) tolerieren, vor torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Windows: CUDA-DLLs aus torch/lib für llama-cpp-python (ggml-cuda.dll) verfügbar machen.
# Muss VOR dem ersten llama_cpp-Import stehen (DLLs werden beim Modul-Import geladen).
try:
    import torch as _torch
    _tl = os.path.join(os.path.dirname(_torch.__file__), "lib")
    if os.path.isdir(_tl):
        os.add_dll_directory(_tl)
except Exception:
    pass

import argparse
import json
import csv
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ALT_EMBEDDING    = "intfloat/multilingual-e5-base"
LEOLM_GGUF_DEFAULT = "models/leo-hessianai-7b.Q4_K_M.gguf"


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Atomar schreiben via .tmp + os.replace() — kein korruptes JSON bei Kill."""
    import os as _os
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding=encoding)
    _os.replace(tmp, path)


def cosine_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
    b = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)
    return a @ b.T


def run_embedding_check(model_name, ipase_items, processed_dir, out_dir):
    print(f"\n=== A) Alternativ-Embedding: {model_name} ===")
    item_texts = [it["text"] for it in ipase_items]
    item_input = ["passage: " + t for t in item_texts] if "e5" in model_name else item_texts

    model = SentenceTransformer(model_name)
    print(f"Modell geladen (Dim {model.get_sentence_embedding_dimension()})")
    item_emb = model.encode(item_input, convert_to_numpy=True, show_progress_bar=False)

    files = sorted(processed_dir.glob("*.ich_saetze.json"))
    results = []
    for f in files:
        person_id = f.stem.replace(".ich_saetze", "")
        data = json.loads(f.read_text(encoding="utf-8"))
        ich_saetze = data["ich_saetze"]
        if not ich_saetze:
            continue
        sentences = [s["text"] for s in ich_saetze]
        sent_input = ["passage: " + t for t in sentences] if "e5" in model_name else sentences
        sent_emb = model.encode(sent_input, convert_to_numpy=True, show_progress_bar=False)
        sim = cosine_matrix(sent_emb, item_emb)

        mean_overall = float(sim.mean())
        factor_subindices = {}
        for fid in (1, 2, 3, 4, 5):
            cols = [i for i, it in enumerate(ipase_items) if it["factor"] == fid]
            factor_subindices[str(fid)] = float(sim[:, cols].mean())

        print(f"  {person_id}: Mean cosine (e5) = {mean_overall:.4f}")
        results.append({
            "person_id": person_id,
            "model": model_name,
            "mean_overall": mean_overall,
            **{f"factor_{fid}_mean": v for fid, v in factor_subindices.items()},
        })

    csv_path = out_dir / "_overview_robustness_embedding.csv"
    if results:
        with open(csv_path, "w", encoding="utf-8", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"  → {csv_path}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# LeoLM-Perplexität via llama-cpp-python
# ──────────────────────────────────────────────────────────────────────────────

def _sentence_log_prob_leolm(llm, context_text: str, target_text: str,
                              ctx_tokens: list) -> tuple[float, int]:
    """Mean log P(target | context) via llama-cpp echo+logprobs API.

    Gibt (mean_log_prob, n_scored_tokens) zurück.
    ctx_tokens: vorberechnete Token-IDs des Kontexts (einmal pro Lauf).

    Ziel-Grenze: BPE-Tokenizer können die Naht Kontext+" "+Ziel anders
    segmentieren als den Kontext allein (Merge über die Naht). Die Grenze wird
    deshalb als Länge des gemeinsamen Token-Präfixes von Kontext- und
    Volltext-Tokenisierung bestimmt, nicht als fixe Kontext-Tokenzahl.
    """
    full_text = context_text + " " + target_text
    try:
        full_tokens = llm.tokenize(full_text.encode("utf-8"))
        boundary = 0
        for a, b in zip(ctx_tokens, full_tokens):
            if a != b:
                break
            boundary += 1
        out = llm.create_completion(
            full_text,
            max_tokens=1,
            echo=True,
            logprobs=True,
            temperature=0.0,
        )
        token_logprobs = out["choices"][0]["logprobs"]["token_logprobs"]
        # token_logprobs[0] ist immer None (kein Vorgänger für erstes Token).
        # Ziel-Tokens beginnen ab der gemeinsamen Präfix-Grenze.
        tgt_lps = [lp for lp in token_logprobs[boundary:] if lp is not None]
        if not tgt_lps:
            return float("nan"), 0
        return float(np.mean(tgt_lps)), len(tgt_lps)
    except Exception:
        return float("nan"), 0


def _sentence_log_prob_leolm_baseline(llm, target_text: str) -> tuple[float, int]:
    """Mean log P(target | ∅) für PMI-Baseline (ohne IPASE-Kontext)."""
    try:
        out = llm.create_completion(
            target_text,
            max_tokens=1,
            echo=True,
            logprobs=True,
            temperature=0.0,
        )
        token_logprobs = out["choices"][0]["logprobs"]["token_logprobs"]
        # Erstes Token hat None → ab Index 1
        lps = [lp for lp in token_logprobs[1:] if lp is not None]
        if not lps:
            return float("nan"), 0
        return float(np.mean(lps)), len(lps)
    except Exception:
        return float("nan"), 0


def run_perplexity_check(gguf_path: str, ipase_items, processed_dir, out_dir,
                         n_gpu_layers: int = -1, compute_pmi: bool = False):
    """Konditionale Perplexität P(Ich-Satz | IPASE-Kontext) via LeoLM-GGUF.

    PMI(s; C) = log P(s | C) - log P(s | ∅)  — optional via --pmi.
    M.4.6: LeoLM ist Deutsch-spezialisiert (LLaMA-2 Fine-Tune), daher besser
    geeignet als multilinguales XGLM für deutsche klinische Sprache.
    """
    print(f"\n=== B) Perplexitäts-Alignment: {gguf_path} ===")

    try:
        from llama_cpp import Llama
    except ImportError:
        import sys
        sys.exit(
            "llama-cpp-python nicht installiert. Bitte ausfuehren:\n"
            "  pip install llama-cpp-python "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124\n"
            "oder (CPU-Fallback): pip install llama-cpp-python"
        )

    gguf = Path(gguf_path)
    if not gguf.exists():
        import sys
        sys.exit(
            f"GGUF-Datei nicht gefunden: {gguf}\n"
            "Bitte zuerst LeoLM-GGUF herunterladen:\n"
            "  python -c \"from huggingface_hub import hf_hub_download; "
            "hf_hub_download('TheBloke/leo-hessianai-7B-GGUF', "
            "'leo-hessianai-7b.Q4_K_M.gguf', local_dir='models')\""
        )

    print(f"  Lade GGUF (n_gpu_layers={n_gpu_layers})...")
    llm = Llama(
        model_path=str(gguf),
        n_ctx=2048,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
        logits_all=True,  # Required for logprobs in create_completion
    )

    context = " ".join(it["text"] for it in ipase_items)
    # Kontext-Tokens einmalig bestimmen (IDs, für Präfix-Grenzabgleich pro Satz)
    ctx_tokens = llm.tokenize(context.encode("utf-8"))
    print(f"  Kontext: {len(ctx_tokens)} Tokens")
    if compute_pmi:
        print("  PMI-Modus: berechne zusaetzlich Baseline log P(s | ∅) pro Satz.")

    files = sorted(processed_dir.glob("*.ich_saetze.json"))
    results = []
    for f in files:
        person_id = f.stem.replace(".ich_saetze", "")
        data = json.loads(f.read_text(encoding="utf-8"))
        ich_saetze = data["ich_saetze"]
        if not ich_saetze:
            continue

        per_sentence = []
        for idx, s in enumerate(ich_saetze):
            target_text = s["text"]
            mean_lp_ctx, n_tok = _sentence_log_prob_leolm(
                llm, context, target_text, ctx_tokens
            )
            perplexity = (float(np.exp(-mean_lp_ctx))
                          if not np.isnan(mean_lp_ctx) else float("nan"))
            row = {
                "satz": target_text,
                "n_tokens": n_tok,
                "mean_log_prob_ctx": mean_lp_ctx,
                "perplexity": perplexity,
            }
            if compute_pmi:
                mean_lp_base, _ = _sentence_log_prob_leolm_baseline(llm, target_text)
                if not (np.isnan(mean_lp_ctx) or np.isnan(mean_lp_base)):
                    row["mean_log_prob_baseline"] = mean_lp_base
                    row["pmi"] = float(mean_lp_ctx - mean_lp_base)
                else:
                    row["mean_log_prob_baseline"] = mean_lp_base
                    row["pmi"] = float("nan")
            per_sentence.append(row)

            # Fortschrittsanzeige: alle 10 Sätze eine Zeile
            if (idx + 1) % 10 == 0:
                done = idx + 1
                total = len(ich_saetze)
                print(f"    {person_id}: {done}/{total} Sätze ...", flush=True)

        valid_pp = [p["perplexity"] for p in per_sentence
                    if not np.isnan(p["perplexity"])]
        mean_pp   = float(np.mean(valid_pp))   if valid_pp else float("nan")
        median_pp = float(np.median(valid_pp)) if valid_pp else float("nan")
        print(f"  {person_id}: Mean PPL = {mean_pp:.2f}, Median = {median_pp:.2f}")

        person_aggregates = {
            "person_id":        person_id,
            "model":            gguf_path,
            "mean_perplexity":  mean_pp,
            "median_perplexity": median_pp,
            "per_sentence":     per_sentence,
        }
        overview_row = {
            "person_id":        person_id,
            "model":            gguf_path,
            "mean_perplexity":  mean_pp,
            "median_perplexity": median_pp,
        }
        if compute_pmi:
            pmi_vals = [p["pmi"] for p in per_sentence
                        if p.get("pmi") is not None and not np.isnan(p["pmi"])]
            if pmi_vals:
                mean_pmi   = float(np.mean(pmi_vals))
                median_pmi = float(np.median(pmi_vals))
                person_aggregates["mean_pmi"]   = mean_pmi
                person_aggregates["median_pmi"] = median_pmi
                overview_row["mean_pmi"]        = mean_pmi
                overview_row["median_pmi"]      = median_pmi
                print(f"  {person_id}: Mean PMI = {mean_pmi:+.4f}")

        out_path = out_dir / f"{person_id}.perplexity.json"
        _atomic_write_text(out_path, json.dumps(person_aggregates, ensure_ascii=False, indent=2))
        results.append(overview_row)

    csv_path = out_dir / "_overview_robustness_perplexity.csv"
    if results:
        keys: list[str] = []
        for r in results:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(csv_path, "w", encoding="utf-8", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"  → {csv_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Robustheits-Companion: e5-Embedding + LeoLM-Perplexität"
    )
    parser.add_argument("--ipase",         default="data/ipase_items.json")
    parser.add_argument("--processed",     default="data/processed")
    parser.add_argument("--out",           default="results")
    parser.add_argument("--alt-embedding", default=ALT_EMBEDDING)
    parser.add_argument("--gguf",          default=LEOLM_GGUF_DEFAULT,
                        help="Pfad zur LeoLM-GGUF-Datei (default: %(default)s)")
    parser.add_argument("--n-gpu-layers",  type=int, default=-1,
                        help="Anzahl GPU-Schichten fuer llama-cpp (-1 = alle, 0 = CPU)")
    parser.add_argument("--skip-embedding",   action="store_true")
    parser.add_argument("--skip-perplexity",  action="store_true")
    parser.add_argument("--pmi",              action="store_true",
                        help="Zusaetzlich PMI = log P(s|C) - log P(s|∅) berechnen.")
    args = parser.parse_args()

    ipase = json.loads(Path(args.ipase).read_text(encoding="utf-8"))
    items = sorted(ipase["items"], key=lambda x: x["id"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(args.processed)

    if not args.skip_embedding:
        run_embedding_check(args.alt_embedding, items, processed_dir, out_dir)
    if not args.skip_perplexity:
        run_perplexity_check(
            args.gguf, items, processed_dir, out_dir,
            n_gpu_layers=args.n_gpu_layers,
            compute_pmi=args.pmi,
        )

    print("\nFertig.")


if __name__ == "__main__":
    main()
