"""Zusätzliche Sprachfeatures (vgl. Methodenkapitel M.4.6).

Berechnet pro Person:
  1) Pronomen-Frequenzen (1. Pers. Singular + Plural)
  2) Mittlere semantische Kohärenz konsekutiver Sätze (mpnet)
  3) POS-Komposition (Nomen, Verben, Adjektive, Adverbien, Pronomen)
  4) Type-Token-Ratio (lexikalische Vielfalt)
  5) Mean Length of Utterance (Tokens pro Satz)
  6) Konkretheit/Imageability nach Köper & Schulte im Walde (2016, LREC;
     IMS Stuttgart, ~350k dt. Lemmata), Sensitivitäts-Fallback Charbonnier &
     Wartena (2019) für nicht abgedeckte Lemmata. Nomen-Anteil-Surrogat bleibt
     als Vergleichsspalte erhalten (implementation="surrogate").
  7) LIWC-Analytic (CDI nach Pennebaker et al. 2014) aus dem deutschen
     LIWC-Lexikon nach Wolf et al. (2008), sobald Lizenz/Dictionary in
     data/lexica/ vorliegt. POS-Surrogat bleibt als Vergleichsspalte erhalten
     (implementation="surrogate"), solange kein Dictionary verfügbar ist.

Liest:
  - data/transcripts/*.txt (Whisper-Output, Vollvolumen)
  - data/processed/*.sentences.json (Output von 02_preprocess.py, segmentiert)
  - data/lexica/affective_norms.txt.gz (Köper & Schulte im Walde 2016, optional)
  - data/lexica/charbonnier_wartena_2019.tsv (Sensitivitäts-Fallback, optional)
  - data/lexica/*.dic | *.dicx (deutsches LIWC-Dictionary, optional, lizenzpflichtig)

Schreibt:
  - results/_overview_features.csv

Verwendung:
    python src/05_features.py
"""
import argparse
import gzip
import json
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer


ICH_FORMS = {"ich", "mir", "mich", "mein", "meine", "meiner", "meines",
             "meinem", "meinen", "meins"}
WIR_FORMS = {"wir", "uns", "unser", "unsere", "unseres", "unserem", "unseren"}

# Analytic-Thinking-Surrogat (LIWC-ähnlich):
# „analytisch" = mehr Artikel/Substantive/Präpositionen/Konjunktionen
# „nicht-analytisch" = mehr Pronomen/Adverbien/Interjektionen/Auxiliarverben
ANALYTIC_POS_PLUS = {"NOUN", "PROPN", "DET", "ADP", "CCONJ", "SCONJ", "NUM"}
ANALYTIC_POS_MINUS = {"PRON", "ADV", "INTJ", "AUX", "PART"}

# ─────────────────────────────────────────────────────────────────────────
# Sekundärmarker M.4.6: Konkretheit/Imageability (Köper & Schulte im Walde 2016)
# ─────────────────────────────────────────────────────────────────────────
LEXICA_DIR = Path("data/lexica")
KOEPER_LEXICON_PATH = LEXICA_DIR / "affective_norms.txt.gz"
CHARBONNIER_WARTENA_PATH = LEXICA_DIR / "charbonnier_wartena_2019.tsv"
LIWC_DICT_GLOBS = ("*.dic", "*.dicx")

# Inhaltswörter für die Konkretheits-/Imageability-Mittelung (vgl. Anweisung M.4.6)
CONCRETENESS_CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def load_koeper_lexicon(path: Path = KOEPER_LEXICON_PATH) -> dict[str, dict[str, float]] | None:
    """Lädt Köper & Schulte im Walde (2016) Affektnormen (~350k dt. Lemmata).

    Format: TSV (ggf. gzip), Spalten Word / AbstConc / Arou / IMG / Val.

    Skalenrichtung empirisch geprüft (2026-07-22) anhand bekannter Wortpaare:
    konkrete Wörter -> hohe AbstConc-Werte (Stuhl=8.01, Tisch=7.77, Auto=7.30,
    Hund=7.16, Haus=6.46), abstrakte Wörter -> niedrige Werte (Wahrheit=2.19,
    Liebe=2.30, Freiheit=2.34, Gerechtigkeit=2.50, Konzept=2.62). D.h. trotz
    des Spaltennamens "AbstConc" ist die Skala bereits so gepolt, dass hohe
    Werte = KONKRET bedeuten. Der Rohwert wird daher UNVERÄNDERT als
    Konkretheit übernommen (KEINE Umpolung).
    """
    if not path.exists():
        return None
    try:
        lex: dict[str, dict[str, float]] = {}
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            idx = {name: i for i, name in enumerate(header)}
            if "Word" not in idx or "AbstConc" not in idx or "IMG" not in idx:
                return None
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < len(header):
                    continue
                word = parts[idx["Word"]]
                try:
                    conc = float(parts[idx["AbstConc"]])
                    img = float(parts[idx["IMG"]])
                except ValueError:
                    continue
                lex[word] = {"concreteness": conc, "imageability": img}
        return lex if lex else None
    except Exception as e:
        print(f"  WARNUNG: Köper-Lexikon konnte nicht geladen werden ({e}). "
              f"Fahre mit Surrogat fort.")
        return None


def load_charbonnier_wartena(path: Path = CHARBONNIER_WARTENA_PATH) -> dict[str, float] | None:
    """Lädt Charbonnier & Wartena (2019) Konkretheits-Vorhersagen als
    Sensitivitäts-Fallback für bei Köper & Schulte im Walde fehlende Lemmata.

    Erwartetes Format: TSV, Spalten word / concreteness (keine offizielle
    öffentliche Download-Quelle bekannt -> in der Regel nicht verfügbar,
    Datei einfach nach data/lexica/charbonnier_wartena_2019.tsv legen,
    falls sie doch beschafft wird).
    """
    if not path.exists():
        return None
    try:
        lex: dict[str, float] = {}
        with open(path, "r", encoding="utf-8") as f:
            f.readline()  # Header überspringen
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                try:
                    lex[parts[0]] = float(parts[1])
                except ValueError:
                    continue
        return lex if lex else None
    except Exception as e:
        print(f"  WARNUNG: Charbonnier & Wartena Fallback konnte nicht geladen werden ({e}).")
        return None


def compute_concreteness_real(doc, koeper_lex: dict | None,
                               cw_fallback_lex: dict | None = None) -> dict:
    """Echte Konkretheit/Imageability nach Köper & Schulte im Walde (2016).

    Mittelwert über Inhaltswörter (NOUN, VERB, ADJ, ADV) via Lemma-Lookup.
    Bei fehlendem Lexikon (Download nicht verfügbar) wird NaN zurückgegeben
    und implementation="unavailable" gesetzt -- der Aufrufer behält in diesem
    Fall parallel das Nomen-Anteil-Surrogat (compute_concreteness_surrogate)
    als einzige nutzbare Konkretheitsspalte.
    """
    content_tokens = [t for t in doc if t.pos_ in CONCRETENESS_CONTENT_POS and t.is_alpha]
    if koeper_lex is None:
        return {
            "concreteness_koeper_mean": float("nan"),
            "imageability_koeper_mean": float("nan"),
            "coverage_concreteness": float("nan"),
            "n_content_tokens_concreteness": len(content_tokens),
            "concreteness_implementation": "unavailable",
        }
    if not content_tokens:
        return {
            "concreteness_koeper_mean": float("nan"),
            "imageability_koeper_mean": float("nan"),
            "coverage_concreteness": float("nan"),
            "n_content_tokens_concreteness": 0,
            "concreteness_implementation": "real_koeper2016",
        }

    conc_vals, img_vals = [], []
    cw_conc_vals = []  # Köper-Werte + CW-Fallback fuer nicht abgedeckte Lemmata
    n_matched = 0
    n_cw_used = 0
    for t in content_tokens:
        lemma = t.lemma_
        entry = (koeper_lex.get(lemma) or koeper_lex.get(lemma.lower())
                 or koeper_lex.get(lemma.capitalize()))
        if entry is not None:
            conc_vals.append(entry["concreteness"])
            img_vals.append(entry["imageability"])
            cw_conc_vals.append(entry["concreteness"])
            n_matched += 1
        elif cw_fallback_lex is not None:
            cw_val = cw_fallback_lex.get(lemma) or cw_fallback_lex.get(lemma.lower())
            if cw_val is not None:
                cw_conc_vals.append(cw_val)
                n_cw_used += 1

    out = {
        "concreteness_koeper_mean": float(np.mean(conc_vals)) if conc_vals else float("nan"),
        "imageability_koeper_mean": float(np.mean(img_vals)) if img_vals else float("nan"),
        "coverage_concreteness": 100.0 * n_matched / len(content_tokens),
        "n_content_tokens_concreteness": len(content_tokens),
        "concreteness_implementation": "real_koeper2016",
    }
    if cw_fallback_lex is not None:
        out["concreteness_with_cw_fallback_mean"] = (
            float(np.mean(cw_conc_vals)) if cw_conc_vals else float("nan")
        )
        out["n_cw_fallback_used"] = n_cw_used
    return out


# ─────────────────────────────────────────────────────────────────────────
# Sekundärmarker M.4.6: LIWC-Analytic (CDI, Pennebaker et al. 2014)
# ─────────────────────────────────────────────────────────────────────────

# CDI-Terme -> gängige LIWC-Kategorienamen (deutsch/englisch), Groß-/
# Kleinschreibung wird beim Abgleich ignoriert. Muss ggf. an die tatsächlichen
# Kategorienamen des deutschen LIWC (Wolf et al. 2008) angepasst werden,
# sobald das lizenzierte Dictionary vorliegt.
CDI_CATEGORY_ALIASES = {
    "article": {"article", "artikel"},
    "prep":    {"prep", "preps", "präposition", "praeposition", "adp"},
    "pronoun": {"pronoun", "ppron", "ipron", "pronomen"},
    "auxverb": {"auxverb", "hilfsverb"},
    "conj":    {"conj", "cconj", "sconj", "konjunktion"},
    "adverb":  {"adverb", "adverbien"},
    "negate":  {"negate", "negation", "verneinung"},
}


def load_liwc_dictionary(path: Path) -> dict | None:
    """Generischer Reader für das LIWC-Dictionary-Format (.dic).

    Erwartetes Format (Standard-LIWC-Textformat):
        %
        1\tfunct
        2\tpronoun
        ...
        %
        wort1\t1\t3
        wort*\t2\t5
        ...
    Zeilen zwischen dem 1. und 2. '%' definieren Kategorie-IDs -> Namen,
    danach folgen Wort-Zeilen mit ihren Kategorie-IDs. Ein '*' am Wortende
    markiert Präfix-Matching (LIWC-Wildcard-Konvention).

    Gibt {"categories": {id: name}, "words": {wort: {kategorienamen}},
    "prefixes": [(präfix, {kategorienamen}), ...]} zurück, oder None falls
    die Datei fehlt (kein Fehler -- Lizenz ggf. noch nicht vorhanden).
    """
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [l.rstrip("\n") for l in f]

    percent_positions = [i for i, l in enumerate(lines) if l.strip() == "%"]
    if len(percent_positions) < 2:
        raise ValueError(f"Kein gültiges LIWC-.dic-Format in {path} (erwarte 2x '%'-Marker)")

    cat_start, cat_end = percent_positions[0], percent_positions[1]
    categories: dict[str, str] = {}
    for line in lines[cat_start + 1:cat_end]:
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip():
            categories[parts[0].strip()] = parts[1].strip()

    exact_words: dict[str, set[str]] = {}
    prefixes: list[tuple[str, set[str]]] = []
    for line in lines[cat_end + 1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        word = parts[0].strip()
        if not word:
            continue
        cat_ids = {p.strip() for p in parts[1:] if p.strip()}
        cat_names = {categories.get(cid, cid).lower() for cid in cat_ids}
        if word.endswith("*"):
            prefixes.append((word[:-1].lower(), cat_names))
        else:
            exact_words[word.lower()] = cat_names

    return {"categories": categories, "words": exact_words, "prefixes": prefixes}


def _liwc_lookup(token_text: str, liwc: dict) -> set[str]:
    """Kategorienamen (kleingeschrieben) für ein Token, exakt oder via Präfix-Wildcard."""
    w = token_text.lower()
    if w in liwc["words"]:
        return liwc["words"][w]
    for prefix, cats in liwc["prefixes"]:
        if w.startswith(prefix):
            return cats
    return set()


def compute_analytic_liwc(doc, liwc: dict | None) -> dict:
    """CDI-Analytic-Score (Pennebaker et al. 2014) aus LIWC-Kategorienanteilen:

        Analytic = 30 + Artikel + Präpositionen - Pronomen - Hilfsverben
                   - Konjunktionen - Adverbien - Negationen

    (Kategorienanteile in % aller Tokens). Ohne Dictionary: NaN,
    implementation="unavailable" -- Aufrufer behält das POS-Surrogat
    (compute_analytic_surrogate) als einzige nutzbare Analytic-Spalte.
    """
    if liwc is None:
        return {"analytic_liwc": float("nan"), "analytic_implementation": "unavailable"}
    tokens = [t for t in doc if not t.is_space and t.is_alpha]
    if not tokens:
        return {"analytic_liwc": float("nan"), "analytic_implementation": "liwc_de_wolf2008"}

    def cat_share(alias_key: str) -> float:
        aliases = CDI_CATEGORY_ALIASES[alias_key]
        n = sum(1 for t in tokens if _liwc_lookup(t.text, liwc) & aliases)
        return 100.0 * n / len(tokens)

    article = cat_share("article")
    prep = cat_share("prep")
    pronoun = cat_share("pronoun")
    auxverb = cat_share("auxverb")
    conj = cat_share("conj")
    adverb = cat_share("adverb")
    negate = cat_share("negate")

    analytic = 30 + article + prep - pronoun - auxverb - conj - adverb - negate
    return {"analytic_liwc": analytic, "analytic_implementation": "liwc_de_wolf2008"}


def compute_pronoun_features(tokens) -> dict:
    """Anteile der 1.-Pers.-Pronomen an Gesamt-Tokens."""
    if not tokens:
        return {"freq_ich": 0.0, "freq_wir": 0.0}
    total = len(tokens)
    n_ich = sum(1 for t in tokens if t.lower() in ICH_FORMS)
    n_wir = sum(1 for t in tokens if t.lower() in WIR_FORMS)
    return {
        "freq_ich_per100": 100.0 * n_ich / total,
        "freq_wir_per100": 100.0 * n_wir / total,
        "freq_ich_total": n_ich,
        "freq_wir_total": n_wir,
        "n_tokens_total": total,
    }


def compute_coherence(sentences, sbert_model) -> dict:
    """Mittlere Kosinus-Ähnlichkeit konsekutiver Sätze.

    min_coherence ist laengenkonfundiert: es ist das Minimum ueber n_pairs
    Ziehungen, und n_pairs streut stark zwischen Faellen (Robustheitscheck
    M.5.x, 2026-09), daher faellt es mit wachsendem n_pairs systematisch.
    min_coherence_p10 (10. Perzentil der Paar-Aehnlichkeiten) ist die
    laengenrobuste Alternative; min_coherence bleibt zum Vergleich erhalten.
    """
    texts = [s["text"] for s in sentences if s.get("text", "").strip()]
    if len(texts) < 2:
        return {"mean_coherence": float("nan"), "min_coherence": float("nan"),
                "min_coherence_p10": float("nan"), "n_pairs": 0}
    emb = sbert_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # Normalisieren
    norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sims = [float(norm[i] @ norm[i+1]) for i in range(len(norm)-1)]
    return {
        "mean_coherence": float(np.mean(sims)),
        "median_coherence": float(np.median(sims)),
        "min_coherence": float(np.min(sims)),
        "min_coherence_p10": float(np.percentile(sims, 10)),
        "n_pairs": len(sims),
    }


def compute_pos_features(doc) -> dict:
    """POS-Komposition + Type-Token-Ratio + MLU."""
    tokens = [t for t in doc if not t.is_space and t.is_alpha]
    if not tokens:
        return {}
    pos_counter = Counter(t.pos_ for t in tokens)
    total = len(tokens)
    # POS-Anteile
    out = {
        f"pos_{p.lower()}_share": 100.0 * pos_counter[p] / total
        for p in ("NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP",
                  "CCONJ", "SCONJ", "AUX", "PROPN", "PART", "INTJ")
    }

    # Type-Token-Ratio (lexikalische Vielfalt)
    lemmas = [t.lemma_.lower() for t in tokens]
    out["type_token_ratio"] = len(set(lemmas)) / len(lemmas)

    # MLU pro Satz (Mean Length of Utterance)
    sent_lengths = []
    for sent in doc.sents:
        n = sum(1 for t in sent if t.is_alpha and not t.is_space)
        if n > 0:
            sent_lengths.append(n)
    out["mean_length_utterance"] = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    out["n_sentences"] = len(sent_lengths)

    return out


def compute_concreteness_surrogate(doc) -> dict:
    """Konkretheits-Surrogat: Nomen-Anteil.

    In der echten Studie wird das durch die Konkretheits-Norm von Lahl et al. (2009)
    ersetzt: pro Wort einen kontinuierlichen Konkretheitswert (1-7 Skala), Mittelwert
    über alle Inhaltswörter.
    """
    content_tokens = [t for t in doc if t.pos_ in ("NOUN", "VERB", "ADJ", "PROPN")
                      and t.is_alpha]
    if not content_tokens:
        return {"concreteness_surrogate": float("nan")}
    n_concrete_pos = sum(1 for t in content_tokens if t.pos_ in ("NOUN", "PROPN"))
    return {
        "concreteness_surrogate": 100.0 * n_concrete_pos / len(content_tokens),
        "n_content_tokens": len(content_tokens),
    }


def compute_analytic_surrogate(doc) -> dict:
    """LIWC-Analytic-Thinking-Surrogat (vereinfacht)."""
    tokens = [t for t in doc if not t.is_space and t.is_alpha]
    if not tokens:
        return {"analytic_surrogate": float("nan")}
    plus = sum(1 for t in tokens if t.pos_ in ANALYTIC_POS_PLUS)
    minus = sum(1 for t in tokens if t.pos_ in ANALYTIC_POS_MINUS)
    total = plus + minus
    if total == 0:
        return {"analytic_surrogate": float("nan")}
    # Normalisiert auf 0-100: 100 = nur „analytische" POS, 0 = nur „nicht-analytische"
    return {"analytic_surrogate": 100.0 * plus / total}


def main():
    parser = argparse.ArgumentParser(description="Zusätzliche Sprachfeatures")
    parser.add_argument("--transcripts", default="data/transcripts")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--out", default="results")
    parser.add_argument("--spacy-model", default="de_core_news_lg")
    parser.add_argument("--sbert-model", default="paraphrase-multilingual-mpnet-base-v2")
    args = parser.parse_args()

    transcript_dir = Path(args.transcripts)
    processed_dir = Path(args.processed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lade spaCy-Modell {args.spacy_model}...")
    nlp = spacy.load(args.spacy_model)
    print(f"Lade S-BERT-Modell {args.sbert_model}...")
    sbert = SentenceTransformer(args.sbert_model)

    print(f"Lade Köper & Schulte im Walde (2016) Lexikon ({KOEPER_LEXICON_PATH})...")
    koeper_lex = load_koeper_lexicon()
    if koeper_lex is None:
        print("  Lexikon nicht verfügbar -- Konkretheit/Imageability bleiben Surrogat-only.")
    else:
        print(f"  {len(koeper_lex)} Lemmata geladen.")

    cw_lex = load_charbonnier_wartena()
    if cw_lex is None:
        print(f"  Charbonnier & Wartena (2019) Fallback nicht verfügbar ({CHARBONNIER_WARTENA_PATH}) "
              f"-- keine Sensitivitätsspalte für nicht abgedeckte Lemmata.")
    else:
        print(f"  Charbonnier & Wartena (2019) Fallback geladen: {len(cw_lex)} Lemmata.")

    liwc = None
    liwc_candidates = sorted(
        f for pattern in LIWC_DICT_GLOBS for f in LEXICA_DIR.glob(pattern)
    ) if LEXICA_DIR.exists() else []
    if liwc_candidates:
        print(f"Lade LIWC-Dictionary ({liwc_candidates[0]})...")
        try:
            liwc = load_liwc_dictionary(liwc_candidates[0])
        except Exception as e:
            print(f"  WARNUNG: LIWC-Dictionary konnte nicht geparst werden ({e}). "
                  f"Fahre mit Analytic-Surrogat fort.")
            liwc = None
    if liwc is None:
        print("  Kein deutsches LIWC-Dictionary (Wolf et al. 2008) in data/lexica/ gefunden "
              "-- Analytic bleibt Surrogat-only (Lizenz noch nicht vorhanden).")
    else:
        print(f"  LIWC-Dictionary geladen: {len(liwc['categories'])} Kategorien, "
              f"{len(liwc['words'])} Wörter, {len(liwc['prefixes'])} Präfix-Wildcards.")

    txt_files = sorted(transcript_dir.glob("*.txt"))
    rows = []
    low_coverage_cases = []
    for txt_file in txt_files:
        person_id = txt_file.stem
        print(f"\n→ {person_id}")
        raw = txt_file.read_text(encoding="utf-8").strip()
        doc = nlp(raw)
        tokens = [t.text for t in doc if not t.is_space and t.is_alpha]

        # Audio-Dauer für Wörter pro Minute: bevorzugt aus .meta.json (neue Pipeline),
        # Fallback Whisper-JSON-Segmente (alte Pipeline). NaN, wenn keins existiert.
        duration_sec = float("nan")
        meta_file = txt_file.parent / f"{person_id}.meta.json"
        whisper_json = txt_file.parent / f"{person_id}.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                duration_sec = float(meta.get("duration_sec", float("nan")))
            except Exception:
                pass
        if (np.isnan(duration_sec) or duration_sec <= 0) and whisper_json.exists():
            try:
                wdata = json.loads(whisper_json.read_text(encoding="utf-8"))
                if wdata.get("segments"):
                    duration_sec = float(wdata["segments"][-1].get("end", 0.0))
            except Exception:
                pass

        # Sentence segmentation aus processed laden (konsistent mit 02)
        sent_file = processed_dir / f"{person_id}.sentences.json"
        if sent_file.exists():
            data = json.loads(sent_file.read_text(encoding="utf-8"))
            sentences = data["sentences"]
        else:
            sentences = [{"text": s.text.strip()} for s in doc.sents]

        features = {"person_id": person_id}
        features.update(compute_pronoun_features(tokens))
        features["duration_sec"] = duration_sec
        if duration_sec > 0:
            features["words_per_minute"] = len(tokens) / duration_sec * 60.0
        else:
            features["words_per_minute"] = float("nan")
        features.update(compute_coherence(sentences, sbert))
        features.update(compute_pos_features(doc))
        features.update(compute_concreteness_surrogate(doc))
        features.update(compute_analytic_surrogate(doc))
        features.update(compute_concreteness_real(doc, koeper_lex, cw_lex))
        features.update(compute_analytic_liwc(doc, liwc))

        # Konsolen-Report
        print(f"  Tokens: {features['n_tokens_total']}, Sätze: {features['n_sentences']}")
        if not np.isnan(features['duration_sec']):
            print(f"  Audio-Dauer: {features['duration_sec']:.1f}s "
                  f"({features['words_per_minute']:.1f} WPM)")
        else:
            print("  Audio-Dauer: unbekannt (keine meta.json / Whisper-JSON gefunden)")
        print(f"  Pronomen 1.Sg.: {features['freq_ich_per100']:.2f} / 100 Tokens")
        print(f"  Pronomen 1.Pl.: {features['freq_wir_per100']:.2f} / 100 Tokens")
        print(f"  Mean coherence: {features['mean_coherence']:.4f}")
        print(f"  Type-Token-Ratio: {features['type_token_ratio']:.3f}")
        print(f"  MLU (Wörter/Satz): {features['mean_length_utterance']:.1f}")
        print(f"  Konkretheits-Surrogat: {features['concreteness_surrogate']:.1f}%")
        print(f"  Analytic-Surrogat: {features['analytic_surrogate']:.1f}/100")
        if features["concreteness_implementation"] == "real_koeper2016":
            cov = features["coverage_concreteness"]
            print(f"  Konkretheit (Köper 2016): {features['concreteness_koeper_mean']:.2f}, "
                  f"Imageability: {features['imageability_koeper_mean']:.2f}, "
                  f"Coverage: {cov:.1f}%")
            if cov < 70.0:
                print(f"  WARNUNG: Coverage < 70% für {person_id}")
                low_coverage_cases.append((person_id, cov))
        if features["analytic_implementation"] == "liwc_de_wolf2008":
            print(f"  Analytic (LIWC): {features['analytic_liwc']:.1f}")
        rows.append(features)

    if rows:
        out_path = out_dir / "_overview_features.csv"
        all_keys = sorted({k for r in rows for k in r.keys()},
                          key=lambda x: (x != "person_id", x))
        with open(out_path, "w", encoding="utf-8", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=all_keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nÜbersicht: {out_path}")

    print(f"\n--- Sekundärmarker-Status ---")
    print(f"Konkretheit/Imageability: "
          f"{'real_koeper2016 (Köper & Schulte im Walde 2016)' if koeper_lex else 'unavailable -- Surrogat only'}")
    print(f"  CW-2019-Sensitivitäts-Fallback: {'vorhanden' if cw_lex else 'nicht verfügbar'}")
    print(f"Analytic: "
          f"{'liwc_de_wolf2008' if liwc else 'unavailable -- Surrogat only (Lizenz ausstehend)'}")
    if low_coverage_cases:
        print(f"Fälle mit Konkretheits-Coverage < 70% ({len(low_coverage_cases)}):")
        for pid, cov in low_coverage_cases:
            print(f"  - {pid}: {cov:.1f}%")
    print("\nFertig.")


if __name__ == "__main__":
    main()
