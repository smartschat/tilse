from __future__ import division

import unittest

from tilse.evaluation import util


class TestUtil(unittest.TestCase):
    def test_get_f_score(self):
        self.assertEqual(0, util.get_f_score(0, 1, 1))
        self.assertEqual(1, util.get_f_score(1, 1, 1))
        self.assertEqual(1, util.get_f_score(1, 1, 1))
        self.assertAlmostEqual(0.555555556, util.get_f_score(0.5, 1, 0.5))

    def test_compute_rouge_approximation(self):
        pred_sent = ["A test .", "Another test ."]
        ground_truth1 = [[" A nice test ."]]
        ground_truth2 = [[" A nice test .", "Another ."]]
        ground_truth3 = [[" A nice test ."], ["I like the test ."]]

        self.assertEqual(2 * 0.5 * 0.75 / (0.5 + 0.75), util.compute_rouge_approximation(pred_sent, ground_truth1))
        self.assertEqual(2 * (5 / 6) * (5 / 6) / ((5 / 6) + (5 / 6)),
                         util.compute_rouge_approximation(pred_sent, ground_truth2))
        self.assertEqual(2 * (5 / 12) * (5 / 9) / ((5 / 12) + (5 / 9)),
                         util.compute_rouge_approximation(pred_sent, ground_truth3))

if __name__ == '__main__':
    unittest.main()
