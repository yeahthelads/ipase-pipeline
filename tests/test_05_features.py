# -*- coding: utf-8 -*-
"""Unit-Tests für die neuen Sekundärmarker in 05_features.py (Phase 2, M.4.6).

Getestet ohne echtes spaCy-Modell (Fake-Token-Objekte) und ohne die echten,
lizenzpflichtigen Lexika -- Köper & Schulte im Walde wird mit einer winzigen
In-Memory-Datei simuliert, das LIWC-Dictionary mit einem 5-Wort-Mini-Dictionary
(so ist der Reader bereits vor Lizenzerhalt getestet).

Ausführen:
    python -m unittest tests/test_05_features.py -v
"""
import gzip
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "src" / "05_features.py"

# Modulname beginnt mit einer Ziffer -> kein gültiger Python-Identifier,
# daher dynamisches Laden über importlib statt "import 05_features".
spec = importlib.util.spec_from_file_location("features05", MODULE_PATH)
features05 = importlib.util.module_from_spec(spec)
sys.modules["features05"] = features05
spec.loader.exec_module(features05)


class FakeToken:
    """Minimaler Ersatz für ein spaCy-Token (nur die Attribute, die die
    getesteten Funktionen tatsächlich lesen)."""

    def __init__(self, text, lemma=None, pos="X", is_alpha=True, is_space=False):
        self.text = text
        self.lemma_ = lemma if lemma is not None else text
        self.pos_ = pos
        self.is_alpha = is_alpha
        self.is_space = is_space


class TestKoeperLexicon(unittest.TestCase):
    def _write_lexicon(self, tmp_path: Path, gz: bool) -> Path:
        header = "Word\tAbstConc\tArou\tIMG\tVal\n"
        rows = [
            "Stuhl\t8.01\t2.95\t7.89\t5.03\n",   # konkret
            "Tisch\t7.77\t1.71\t7.89\t5.46\n",   # konkret
            "Liebe\t2.30\t7.19\t4.44\t7.91\n",   # abstrakt
            "gehen\t5.00\t3.00\t5.50\t5.00\n",
        ]
        content = header + "".join(rows)
        path = tmp_path / ("lex.txt.gz" if gz else "lex.txt")
        if gz:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def test_load_plain_and_gz_identical(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            plain = self._write_lexicon(tmp, gz=False)
            gz = self._write_lexicon(tmp, gz=True)
            lex_plain = features05.load_koeper_lexicon(plain)
            lex_gz = features05.load_koeper_lexicon(gz)
            self.assertIsNotNone(lex_plain)
            self.assertIsNotNone(lex_gz)
            self.assertEqual(lex_plain, lex_gz)
            self.assertEqual(lex_plain["Stuhl"]["concreteness"], 8.01)
            self.assertEqual(lex_plain["Stuhl"]["imageability"], 7.89)

    def test_missing_file_returns_none(self):
        self.assertIsNone(features05.load_koeper_lexicon(Path("does_not_exist.txt.gz")))

    def test_scale_direction_concrete_higher_than_abstract(self):
        """Regressionstest für die empirisch geprüfte Skalenrichtung: konkrete
        Wörter (Stuhl/Tisch) müssen höhere Rohwerte haben als abstrakte (Liebe)."""
        with tempfile.TemporaryDirectory() as td:
            path = self._write_lexicon(Path(td), gz=False)
            lex = features05.load_koeper_lexicon(path)
            self.assertGreater(lex["Stuhl"]["concreteness"], lex["Liebe"]["concreteness"])
            self.assertGreater(lex["Tisch"]["concreteness"], lex["Liebe"]["concreteness"])


class TestConcretenessReal(unittest.TestCase):
    def _lexicon(self):
        return {
            "Stuhl": {"concreteness": 8.0, "imageability": 7.9},
            "Tisch": {"concreteness": 7.8, "imageability": 7.9},
            "denken": {"concreteness": 2.5, "imageability": 3.0},
        }

    def test_unavailable_lexicon(self):
        doc = [FakeToken("Stuhl", lemma="Stuhl", pos="NOUN")]
        out = features05.compute_concreteness_real(doc, None)
        self.assertEqual(out["concreteness_implementation"], "unavailable")
        self.assertTrue(out["coverage_concreteness"] != out["coverage_concreteness"])  # NaN

    def test_partial_coverage(self):
        lex = self._lexicon()
        doc = [
            FakeToken("Stuhl", lemma="Stuhl", pos="NOUN"),
            FakeToken("Tisch", lemma="Tisch", pos="NOUN"),
            FakeToken("Wolke", lemma="Wolke", pos="NOUN"),  # nicht im Lexikon
        ]
        out = features05.compute_concreteness_real(doc, lex)
        self.assertEqual(out["concreteness_implementation"], "real_koeper2016")
        self.assertAlmostEqual(out["coverage_concreteness"], 100.0 * 2 / 3)
        self.assertAlmostEqual(out["concreteness_koeper_mean"], (8.0 + 7.8) / 2)

    def test_content_pos_filter_excludes_function_words(self):
        lex = self._lexicon()
        doc = [
            FakeToken("Stuhl", lemma="Stuhl", pos="NOUN"),
            FakeToken("der", lemma="der", pos="DET"),  # kein Inhaltswort -> zaehlt nicht
        ]
        out = features05.compute_concreteness_real(doc, lex)
        self.assertEqual(out["n_content_tokens_concreteness"], 1)

    def test_cw_fallback_used_for_uncovered_lemma(self):
        lex = self._lexicon()
        cw = {"Wolke": 6.5}
        doc = [
            FakeToken("Stuhl", lemma="Stuhl", pos="NOUN"),
            FakeToken("Wolke", lemma="Wolke", pos="NOUN"),
        ]
        out = features05.compute_concreteness_real(doc, lex, cw)
        self.assertEqual(out["n_cw_fallback_used"], 1)
        self.assertAlmostEqual(out["concreteness_with_cw_fallback_mean"], (8.0 + 6.5) / 2)


class TestLiwcDictionary(unittest.TestCase):
    MINI_DICT = (
        "%\n"
        "1\tarticle\n"
        "2\tprep\n"
        "3\tpronoun\n"
        "4\tauxverb\n"
        "5\tnegate\n"
        "%\n"
        "der\t1\n"
        "in\t2\n"
        "ich\t3\n"
        "ist*\t4\n"
        "nicht\t5\n"
    )

    def _write_dict(self, tmp_path: Path) -> Path:
        path = tmp_path / "mini.dic"
        path.write_text(self.MINI_DICT, encoding="utf-8")
        return path

    def test_missing_file_returns_none(self):
        self.assertIsNone(features05.load_liwc_dictionary(Path("no_such.dic")))

    def test_parses_categories_and_words(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_dict(Path(td))
            liwc = features05.load_liwc_dictionary(path)
            self.assertEqual(len(liwc["categories"]), 5)
            self.assertEqual(liwc["categories"]["1"], "article")
            self.assertIn("der", liwc["words"])
            self.assertEqual(liwc["words"]["der"], {"article"})

    def test_wildcard_prefix_matching(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_dict(Path(td))
            liwc = features05.load_liwc_dictionary(path)
            # "ist*" -> matcht "ist" und "istwas" (Praefix), nicht "gewesen"
            self.assertEqual(features05._liwc_lookup("ist", liwc), {"auxverb"})
            self.assertEqual(features05._liwc_lookup("istwas", liwc), {"auxverb"})
            self.assertEqual(features05._liwc_lookup("gewesen", liwc), set())

    def test_case_insensitive_lookup(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_dict(Path(td))
            liwc = features05.load_liwc_dictionary(path)
            self.assertEqual(features05._liwc_lookup("Der", liwc), {"article"})

    def test_malformed_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "broken.dic"
            path.write_text("kein prozent-marker hier\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                features05.load_liwc_dictionary(path)


class TestAnalyticLiwc(unittest.TestCase):
    MINI_DICT = TestLiwcDictionary.MINI_DICT

    def _liwc(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mini.dic"
            path.write_text(self.MINI_DICT, encoding="utf-8")
            return features05.load_liwc_dictionary(path)

    def test_unavailable_dictionary(self):
        doc = [FakeToken("Haus")]
        out = features05.compute_analytic_liwc(doc, None)
        self.assertEqual(out["analytic_implementation"], "unavailable")

    def test_cdi_formula_matches_hand_calculation(self):
        liwc = self._liwc()
        # 10 Tokens: je 1x article/prep/pronoun/auxverb(wildcard)/negate,
        # 5x ohne Kategorie-Treffer.
        words = ["der", "Haus", "in", "ich", "ist", "gehe", "nicht", "Tisch", "und", "schnell"]
        doc = [FakeToken(w, lemma=w, pos="X") for w in words]
        out = features05.compute_analytic_liwc(doc, liwc)
        self.assertEqual(out["analytic_implementation"], "liwc_de_wolf2008")
        # Analytic = 30 + 10(article) + 10(prep) - 10(pronoun) - 10(auxverb)
        #            - 0(conj) - 0(adverb) - 10(negate) = 20.0
        self.assertAlmostEqual(out["analytic_liwc"], 20.0, places=6)

    def test_empty_doc(self):
        liwc = self._liwc()
        out = features05.compute_analytic_liwc([], liwc)
        self.assertNotEqual(out["analytic_liwc"], out["analytic_liwc"])  # NaN != NaN


class TestCharbonnierWartenaFallback(unittest.TestCase):
    def test_missing_file_returns_none(self):
        self.assertIsNone(features05.load_charbonnier_wartena(Path("no_such.tsv")))

    def test_loads_simple_tsv(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cw.tsv"
            path.write_text("word\tconcreteness\nWolke\t6.5\nFreude\t2.1\n", encoding="utf-8")
            lex = features05.load_charbonnier_wartena(path)
            self.assertEqual(lex["Wolke"], 6.5)
            self.assertEqual(lex["Freude"], 2.1)


if __name__ == "__main__":
    unittest.main()
