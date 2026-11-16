from datetime import datetime
import unittest

import numpy as np
import pandas as pd

import run as r
from tests.test_feature_state import _match


class ComponentTests(unittest.TestCase):
    def test_elo_has_no_country_identity_or_counts(self):
        self.assertEqual(r.VARIANTS["elo_only"], r.s.LOCATION + r.RATINGS)
        self.assertFalse(set(r.VARIANTS["elo_only"]) & set(r.COUNTS + r.s.IDENTITY))

    def test_identity_has_no_elo_or_counts(self):
        self.assertEqual(r.VARIANTS["identity_only"], r.s.LOCATION + r.s.IDENTITY)
        self.assertFalse(set(r.VARIANTS["identity_only"]) & set(r.RATINGS + r.COUNTS))

    def test_counts_are_isolated(self):
        self.assertEqual(r.VARIANTS["counts_only"], r.s.LOCATION + r.COUNTS)
        self.assertFalse(set(r.VARIANTS["elo_identity"]) & set(r.COUNTS))

    def test_primary_contrasts_add_only_the_named_component(self):
        both = set(r.VARIANTS["elo_identity"])
        self.assertEqual(both - set(r.VARIANTS["identity_only"]), set(r.RATINGS))
        self.assertEqual(both - set(r.VARIANTS["elo_only"]), set(r.s.IDENTITY))
        self.assertEqual(set(r.VARIANTS["original_combined"]) - both, set(r.COUNTS))

    def test_original_combined_order_and_common_context_are_preserved(self):
        self.assertEqual(r.VARIANTS["original_combined"], r.ORIGINAL_COMBINED)
        for columns in r.VARIANTS.values():
            self.assertEqual(columns[:len(r.s.LOCATION)], r.s.LOCATION)
            self.assertEqual(len(columns), len(set(columns)))
            self.assertFalse(set(columns) & set(r.s.HOME + r.s.DISTANCE))

    def test_every_variant_swaps_and_predicts_symmetrically(self):
        base = r.s.e.FeatureBuilder().preview(_match("future", datetime(2025, 1, 20), 1))
        frame = pd.DataFrame([base])
        frame["discipline"] = pd.Categorical(frame["discipline"], categories=["MS", "WS", "MD", "WD", "XD", "OTHER"])
        for column in r.VARIANTS["original_combined"]:
            frame[column] = "AAA" if column == "country_id_t1" else "BBB" if column == "country_id_t2" else 0.0
        frame["country_rating_t1"], frame["country_rating_t2"], frame["country_rating_diff"] = 1550, 1450, 100

        class Model:
            def predict_proba(self, data):
                score = np.zeros(len(data))
                if "country_rating_diff" in data:
                    score += data["country_rating_diff"].to_numpy()/400
                if "country_id_t1" in data:
                    score += data["country_id_t1"].astype(object).map({"AAA": 1, "BBB": -1}).to_numpy()/4
                p = 1/(1+np.exp(-score))
                return np.column_stack((1-p, p))

        for variant in r.VARIANTS:
            with self.subTest(variant=variant):
                first, _ = r.s.matrices(frame, frame, variant)
                swapped = r.s.reverse(first, variant)
                pd.testing.assert_frame_equal(first, r.s.reverse(swapped, variant))
                np.testing.assert_allclose(r.s.predict(Model(), first, variant), 1-r.s.predict(Model(), swapped, variant), atol=1e-15)


if __name__ == "__main__":
    unittest.main()
