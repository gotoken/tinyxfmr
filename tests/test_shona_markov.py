import importlib.util
import math
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_shona_markov.py"
SPEC = importlib.util.spec_from_file_location("shona_markov", SCRIPT)
shona = importlib.util.module_from_spec(SPEC)
sys.modules["shona_markov"] = shona
assert SPEC.loader is not None
SPEC.loader.exec_module(shona)


class ParseTests(unittest.TestCase):
    def test_parse_weights_and_vowels(self):
        raw = b"b a n d i r a\t2\nch e r e ng a 1\n"
        corpus = shona.parse(raw)
        self.assertEqual(corpus.source_rows, 2)
        self.assertEqual(corpus.entries, 3)
        self.assertEqual(corpus.form_counts[("b", "a", "n", "d", "i", "r", "a")], 2)
        self.assertEqual(shona.vowels(("b", "a", "n", "d", "i", "r", "a")), ("a", "i", "a"))

    def test_grouped_split_keeps_duplicate_forms_together(self):
        forms = {
            ("x", "a"): 3,
            ("x", "e"): 2,
            ("x", "i"): 2,
            ("x", "o"): 2,
            ("x", "u"): 2,
            ("y", "a"): 2,
        }
        splits = shona.split_forms(forms, seed=1)
        membership = {}
        for split_name, items in splits.items():
            for item in set(items):
                self.assertNotIn(item, membership)
                membership[item] = split_name
            for item in set(items):
                self.assertEqual(items.count(item), forms[item])


class TargetAndModelTests(unittest.TestCase):
    def test_final_target_exclusion(self):
        seq = ("a", "i", "a")
        all_targets = list(shona.examples(seq, exclude_final=False))
        no_final = list(shona.examples(seq, exclude_final=True))
        self.assertEqual(all_targets, [(('a',), 'i'), (('a', 'i'), 'a')])
        self.assertEqual(no_final, [(('a',), 'i')])

    def test_first_order_model_learns_context(self):
        sequences = [("a", "i", "a"), ("a", "i", "a"), ("u", "u", "a")]
        unigram = shona.Model(0, alpha=0.5)
        bigram = shona.Model(1, alpha=0.5)
        unigram.fit(sequences, exclude_final=False)
        bigram.fit(sequences, exclude_final=False)
        self.assertGreater(bigram.probs(("a",))["i"], unigram.probs(())["i"])
        self.assertGreater(bigram.probs(("i",))["a"], unigram.probs(())["a"])

    def test_second_order_uses_shorter_history_for_first_transition(self):
        model = shona.Model(2, alpha=0.5)
        model.fit([("a", "i", "a"), ("a", "i", "a")], exclude_final=False)
        self.assertGreater(model.probs(("a",))["i"], 0.2)
        self.assertGreater(model.probs(("a", "i"))["a"], 0.2)

    def test_metrics_are_finite(self):
        sequences = [("a", "i", "a"), ("u", "u", "a"), ("o", "o", "a")]
        model = shona.Model(1, alpha=0.5)
        model.fit(sequences, exclude_final=False)
        metrics = shona.evaluate(model, sequences, exclude_final=False)
        self.assertEqual(metrics.targets, 6)
        self.assertTrue(math.isfinite(metrics.nll_nats))
        self.assertGreaterEqual(metrics.accuracy, 0)
        self.assertLessEqual(metrics.accuracy, 1)


class ReferenceTests(unittest.TestCase):
    def test_table6_reference_has_all_25_pairs(self):
        self.assertEqual(set(shona.TABLE6), {a + b for a in shona.VOWELS for b in shona.VOWELS})
        self.assertEqual(shona.TABLE6["aa"], 1443)
        self.assertEqual(shona.TABLE6["uo"], 1)


if __name__ == "__main__":
    unittest.main()
