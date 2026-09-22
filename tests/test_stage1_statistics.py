"""Statistics-only tests: never import or execute a model."""

import sys
import subprocess
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

from scripts.analyze_stage1_results import (
    RATIOS, bootstrap_mean_ci, classify, compare_frames, holm, keyed_rng,
    main, paired_statistics, recall_table,
)


class Stage1StatisticsTests(unittest.TestCase):
    def test_holm_known_values_and_original_order(self):
        np.testing.assert_allclose(holm([.04, .001, .02]), [.04, .003, .04])
        np.testing.assert_allclose(holm([.8, .9, 1]), [1, 1, 1])

    def test_bootstrap_resamples_rows_and_is_batch_invariant(self):
        x = np.array([[1., -1.], [4., -4.], [8., -8.]])
        rng = keyed_rng(2024, "test")
        indices = rng.integers(0, 3, size=(400, 3))
        expected = np.quantile(x[indices].mean(axis=1), [.025, .975], axis=0)
        for batch in [1, 37, 256]:
            ci = bootstrap_mean_ci(x, seed=2024, key="test", resamples=400, batch_size=batch)
            np.testing.assert_array_equal(ci, expected)
        np.testing.assert_array_equal(ci[:, 0], -ci[::-1, 1])

    def test_effect_sizes_ties_and_paired_test(self):
        d = np.array([-1., 2., 3., 0.])
        r = paired_statistics(d+10, np.ones(4)*10, seed=9, resamples=300, key="effects")
        self.assertEqual(r["n_queries"], 4)
        self.assertEqual(r["mean_paired_difference"], 1.)
        self.assertEqual(r["median_paired_difference"], 1.)
        self.assertAlmostEqual(r["cohen_dz"], 1/d.std(ddof=1))
        self.assertAlmostEqual(r["rank_biserial"], 2/3)
        self.assertEqual(r["left_win_rate"], .5)
        self.assertEqual(r["right_win_rate"], .25)
        self.assertEqual(r["tie_rate"], .25)
        self.assertAlmostEqual(r["t_p_raw"], stats.ttest_1samp(d, 0).pvalue)
        opposite = paired_statistics(np.zeros(4), d, seed=9, resamples=300, key="effects")
        self.assertAlmostEqual(opposite["rank_biserial"], -r["rank_biserial"])
        self.assertAlmostEqual(opposite["cohen_dz"], -r["cohen_dz"])

    def test_all_zero_and_constant_difference(self):
        r = paired_statistics([1, 1, 1], [1, 1, 1], seed=2, resamples=100, key="zeros")
        for k in ["mean_paired_difference", "ci_low", "ci_high", "cohen_dz", "rank_biserial"]:
            self.assertEqual(r[k], 0)
        self.assertEqual(r["wilcoxon_p_raw"], 1)
        self.assertEqual(r["t_p_raw"], 1)
        self.assertEqual(r["tie_rate"], 1)
        r = paired_statistics([2, 2, 2], [1, 1, 1], seed=2, resamples=100, key="constant")
        self.assertTrue(np.isnan(r["cohen_dz"]))
        self.assertEqual(r["ci_low"], 1)
        self.assertEqual(r["ci_high"], 1)

    def test_saved_aggregate_validation_rejects_corruption_or_duplicate(self):
        expected = pd.DataFrame(dict(query_id=[1, 2], value=[.2, .6]))
        compare_frames(expected.iloc[::-1], expected, ["query_id"], ["value"])
        corrupt = expected.copy()
        corrupt.loc[0, "value"] = .21
        with self.assertRaisesRegex(ValueError, "disagree"):
            compare_frames(corrupt, expected, ["query_id"], ["value"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare_frames(pd.concat([expected, expected]), expected, ["query_id"], ["value"])

    def test_recall_sign_and_clean_subset_use_queries_not_draws(self):
        rows = []
        for ratio in RATIOS:
            for q, clean_rank, attention, random in [(1, 1, 1., 1.), (2, 1, 0., .4), (3, 4, 0., .2)]:
                row = dict(query_id=q, mask_mode="connected_topk", baseline="random_token", mask_ratio=ratio,
                           clean_rank=clean_rank, attention_failure_flip=float(clean_rank == 1)*(1-attention),
                           random_failure_flip=float(clean_rank == 1)*(1-random))
                for k in [1, 5, 10]:
                    row[f"attention_hit_at_{k}"] = attention
                    row[f"random_hit_at_{k}"] = random
                rows.append(row)
        result = recall_table(pd.DataFrame(rows), SimpleNamespace(seed=4, bootstrap_resamples=300))
        all_r1 = result[(result.subset == "all_queries") & (result.metric == "hit_at_1")].iloc[0]
        all_flip = result[(result.subset == "all_queries") & (result.metric == "failure_flip")].iloc[0]
        subset = result[result.subset == "clean_R1_correct"]
        self.assertEqual(all_r1.n_queries, 3)
        self.assertAlmostEqual(all_r1.damage_advantage, .2)
        self.assertAlmostEqual(all_flip.damage_advantage, .4/3)
        self.assertTrue((subset.n_queries == 2).all())
        np.testing.assert_allclose(subset.damage_advantage, .2)
        # On clean successes, recall damage and failure flip advantage coincide.
        a = subset[subset.metric == "hit_at_1"]
        b = subset[subset.metric == "failure_flip"]
        np.testing.assert_allclose(a[["ci_low", "ci_high"]], b[["ci_low", "ci_high"]])

    def test_classification_requires_three_consistent_primary_effects(self):
        primary = pd.DataFrame(dict(ci_low=[.1, .2, .3], mean_paired_difference=[.2, .3, .4], wilcoxon_p_holm=[.001]*3))
        self.assertEqual(classify(primary), "H1 supported")
        primary.loc[0, "ci_low"] = -.1
        self.assertEqual(classify(primary), "H1 weak")
        primary.mean_paired_difference = [-.1, -.2, -.3]
        self.assertEqual(classify(primary), "H1 unsupported")

    def test_cli_rejects_insufficient_bootstrap_before_reading_files(self):
        with self.assertRaisesRegex(ValueError, "10,000"):
            main(["--bootstrap-resamples", "9999"])

    def test_no_model_library_imported(self):
        # Isolate this check from other repository tests that may import torch.
        subprocess.run([sys.executable, "-c", "import sys; import scripts.analyze_stage1_results; "
                        "assert 'torch' not in sys.modules; assert 'src.model' not in sys.modules"], check=True)


if __name__ == "__main__":
    unittest.main()
