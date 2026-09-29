"""spaCy-Preprocessing: Tokenisierung, Satzsegmentierung, „Ich"-Satz-Extraktion.

Liest Transkripte aus data/transcripts/*.txt und produziert für jedes:
  - data/processed/<name>.sentences.json: alle Sätze mit Token- und Wortzahl
  - data/processed/<name>.ich_saetze.json: nur die Sätze, die mind. ein 1.-Pers.-Sing.-Pronomen
                                          enthalten (ich, mir, mich, mein, meiner, meines, ...)

Verwendung:
    python src/02_preprocess.py
    python src/02_preprocess.py --in data/transcripts --out data/processed
    python src/02_preprocess.py --in data/transcripts/EinDatei.txt
"""
import argparse
import json
import re
from pathlib import Path

import spacy

# Lemmas / Token-Formen, die als 1.-Pers.-Sing.-Pronomen zählen
# spaCy lemmatisiert "mir"/"mich" oft zu "ich", aber wir prüfen sicherheitshalber breiter.
ICH_LEMMAS = {"ich"}
ICH_FORMS = {"ich", "mir", "mich", "mein", "meine", "meiner", "meines",
             "meinem", "meinen", "meins"}


def is_ich_sentence(sent) -> bool:
    """True, wenn der Satz mind. ein 1.-Pers.-Sing.-Pronomen oder Possessiv enthält."""
    for tok in sent:
        text_l = tok.text.lower()
        lemma_l = tok.lemma_.lower()
        if lemma_l in ICH_LEMMAS or text_l in ICH_FORMS:
            return True
    return False


def clean_filler(text: str) -> str:
    """Optional: Entfernt typische deutsche Filler-Wörter. Aktuell NUR fürs Reporting,
    wir analysieren die Originalsätze. Die Funktion ist hier als Hook für später."""
    fillers = [r"\b(äh+|ähm+|öhm+|mh+|hm+)\b"]
    for pat in fillers:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def process_one(nlp, transcript_path: Path, out_dir: Path) -> dict:
    raw = transcript_path.read_text(encoding="utf-8").strip()
    doc = nlp(raw)

    sentences = []
    for i, sent in enumerate(doc.sents):
        token_list = [tok.text for tok in sent if not tok.is_space]
        is_ich = is_ich_sentence(sent)
        sentences.append({
            "idx": i,
            "text": sent.text.strip(),
            "n_tokens": len(token_list),
            "n_words": sum(1 for tok in sent if tok.is_alpha),
            "is_ich_satz": is_ich,
        })

    ich_saetze = [s for s in sentences if s["is_ich_satz"]]

    name = transcript_path.stem
    summary = {
        "source_file": str(transcript_path.name),
        "person_id": name,
        "n_sentences_total": len(sentences),
        "n_ich_saetze": len(ich_saetze),
        "n_tokens_total": sum(s["n_tokens"] for s in sentences),
        "n_words_total": sum(s["n_words"] for s in sentences),
        "n_words_ich_saetze": sum(s["n_words"] for s in ich_saetze),
        "share_ich_saetze": (
            len(ich_saetze) / len(sentences) if sentences else 0.0
        ),
    }

    # Output
    (out_dir / f"{name}.sentences.json").write_text(
        json.dumps({"summary": summary, "sentences": sentences},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / f"{name}.ich_saetze.json").write_text(
        json.dumps({"summary": summary, "ich_saetze": ich_saetze},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="spaCy-Preprocessing + Ich-Satz-Extraktion")
    parser.add_argument("--in", dest="in_path", default="data/transcripts",
                        help="Input: Ordner oder einzelne .txt-Datei")
    parser.add_argument("--out", dest="out_path", default="data/processed",
                        help="Output-Ordner")
    parser.add_argument("--model", default="de_core_news_lg",
                        help="spaCy-Modell (default: de_core_news_lg)")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_file():
        files = [in_path]
    else:
        files = sorted(in_path.glob("*.txt"))
    if not files:
        print(f"Keine .txt-Dateien in {in_path}")
        return

    print(f"Lade spaCy-Modell {args.model}...")
    nlp = spacy.load(args.model)
    print(f"Modell geladen. Verarbeite {len(files)} Transkript(e).\n")

    summaries = []
    for f in files:
        print(f"  → {f.name}")
        s = process_one(nlp, f, out_dir)
        summaries.append(s)
        print(f"    {s['n_sentences_total']} Sätze, {s['n_ich_saetze']} Ich-Sätze "
              f"({s['share_ich_saetze']*100:.1f}% der Sätze)")
        print(f"    {s['n_words_total']} Wörter total, {s['n_words_ich_saetze']} in Ich-Sätzen")

    # Aggregate report
    overview = out_dir / "_overview.json"
    overview.write_text(json.dumps(summaries, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nFertig. Pro-Person-Outputs in {out_dir}/, Übersicht: {overview.name}")


if __name__ == "__main__":
    main()
