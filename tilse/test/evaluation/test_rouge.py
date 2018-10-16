from __future__ import division

import unittest
import datetime

from tilse.data import timelines
from tilse.evaluation import rouge


class TestRouge(unittest.TestCase):
    def setUp(self):
        self.evaluator = rouge.TimelineRougeEvaluator()

        self.ground_truth = timelines.GroundTruth(
            [
                timelines.Timeline(
                    {
                        datetime.date(2010, 1, 1): ["timeline summarization ."],
                        datetime.date(2010, 1, 2): ["timeline summarization is awesome .",
                                                    "coreference resolution is , too ."],
                        datetime.date(2010, 1, 4): ["alignments are really nice"]
                    }
                ),
                timelines.Timeline(
                    {
                        datetime.date(2010, 1, 2): ["metrics are complicated ."]
                    }
                ),
            ]
        )

        self.output_same_number_of_dates = timelines.Timeline(
            {
                datetime.date(2010, 1, 2): ["timeline summarization ."],
                datetime.date(2010, 1, 3): ["doing some metric checks ."],
                datetime.date(2010, 1, 5): ["checks for alignments ."],
            }
        )

        self.output_same_number_of_dates_scores_higher_with_date_content_costs = timelines.Timeline(
            {
                datetime.date(2010, 1, 2): ["timeline summarization ."],
                datetime.date(2010, 1, 3): ["alignments are really nice"],
                datetime.date(2010, 1, 5): ["timeline summarization ."],
            }
        )

        self.output_less_dates = timelines.Timeline(
            {
                datetime.date(2010, 1, 2): ["timeline summarization ."],
                datetime.date(2010, 1, 3): ["doing some metric checks ."],
            }
        )

        self.output_more_dates = timelines.Timeline(
            {
                datetime.date(2010, 1, 2): ["timeline summarization ."],
                datetime.date(2010, 1, 3): ["doing some metric checks ."],
                datetime.date(2010, 1, 5): ["checks for alignments ."],
                datetime.date(2010, 1, 6): ["asdf"],
            }
        )

    def test_evaluate_concat(self):
        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 4 / 12,
                    "recall": 4 / 11,
                    "f_score": 2 * (4 / 12) * (4 / 11) / (4 / 12 + 4 / 11)
                }
            },
            self.evaluator.evaluate_concat(self.output_same_number_of_dates, self.ground_truth)
        )

        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 3 / 8,
                    "recall": 3 / 11,
                    "f_score": 2 * (3 / 8) * (3 / 11) / (3 / 8 + 3 / 11)
                }
            },
            self.evaluator.evaluate_concat(self.output_less_dates, self.ground_truth)
        )

        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 4 / 14,
                    "recall": 4 / 11,
                    "f_score": 2 * (4 / 14) * (4 / 11) / (4 / 14 + 4 / 11)
                }
            },
            self.evaluator.evaluate_concat(self.output_more_dates, self.ground_truth)
        )

    def test_evaluate_agreement(self):
        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 2 / 12,
                    "recall": 2 / 11,
                    "f_score": 2 * (2 / 12) * (2 / 11) / (2 / 12 + 2 / 11)
                }
            },
            self.evaluator.evaluate_agreement(self.output_same_number_of_dates, self.ground_truth)
        )

        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 2 / 8,
                    "recall": 2 / 11,
                    "f_score": 2 * (2 / 8) * (2 / 11) / (2 / 8 + 2 / 11)
                }
            },
            self.evaluator.evaluate_agreement(self.output_less_dates, self.ground_truth)
        )

        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 2 / 14,
                    "recall": 2 / 11,
                    "f_score": 2 * (2 / 14) * (2 / 11) / (2 / 14 + 2 / 11)
                }
            },
            self.evaluator.evaluate_agreement(self.output_more_dates, self.ground_truth)
        )

    def test_evaluate_align_date_costs(self):
        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 2.5 / 12,
                    "recall": 2.5 / 11,
                    "f_score": 2 * (2.5 / 12) * (2.5 / 11) / (2.5 / 12 + 2.5 / 11)
                }
            },
            self.evaluator.evaluate_align_date_costs(self.output_same_number_of_dates, self.ground_truth)
        )

        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 2 / 12,
                    "recall": 2 / 11,
                    "f_score": 2 * (2 / 12) * (2 / 11) / (2 / 12 + 2 / 11)
                }
            },
            self.evaluator.evaluate_align_date_costs(self.output_same_number_of_dates_scores_higher_with_date_content_costs,
                                                     self.ground_truth)
        )

        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 2 / 8,
                    "recall": 2 / 11,
                    "f_score": 2 * (2 / 8) * (2 / 11) / (2 / 8 + 2 / 11)
                }
            },
            self.evaluator.evaluate_align_date_costs(self.output_less_dates, self.ground_truth)
        )

        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 2.5 / 14,
                    "recall": 2.5 / 11,
                    "f_score": 2 * (2.5 / 14) * (2.5 / 11) / (2.5 / 14 + 2.5 / 11)
                }
            },
            self.evaluator.evaluate_align_date_costs(self.output_more_dates, self.ground_truth)
        )

    def test_evaluate_align_date_content_costs(self):
        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 2 / 12,
                    "recall": 2.5 / 11,
                    "f_score": 2 * (2 / 12) * (2.5 / 11) / (2 / 12 + 2.5 / 11)
                 }
            },
            self.evaluator.evaluate_align_date_content_costs(self.output_same_number_of_dates, self.ground_truth)
        )

        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 3.4 / 12,
                    "recall": 3.4 / 11,
                    "f_score": 2 * (3.4 / 12) * (3.4 / 11) / (3.4 / 12 + 3.4 / 11)
                }
            },
            self.evaluator.evaluate_align_date_content_costs(
                self.output_same_number_of_dates_scores_higher_with_date_content_costs,
                self.ground_truth
            )
        )

        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 2 / 8,
                    "recall": 2 / 11,
                    "f_score": 2 * (2 / 8) * (2 / 11) / (2 / 8 + 2 / 11)
                 }
            },
            self.evaluator.evaluate_align_date_content_costs(self.output_less_dates, self.ground_truth)
        )

        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 2.5 / 14,
                    "recall": 2.5 / 11,
                    "f_score": 2 * (2.5 / 14) * (2.5 / 11) / (2.5 / 14 + 2.5 / 11)
                }
            },
            self.evaluator.evaluate_align_date_content_costs(self.output_more_dates, self.ground_truth)
        )

    def test_evaluate_align_date_content_costs_many_to_one(self):
        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 3 / 12,
                    "recall": 3.5 / 11,
                    "f_score": 2 * (3 / 12) * (3.5 / 11) / (3 / 12 + 3.5 / 11)
                }
            },
            self.evaluator.evaluate_align_date_content_costs_many_to_one(self.output_same_number_of_dates,
                                                                         self.ground_truth)
        )

        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 2.5 / 8,
                    "recall": 3 / 11,
                    "f_score": 2 * (2.5 / 8) * (3 / 11) / (2.5 / 8 + 3 / 11)
                }
            },
            self.evaluator.evaluate_align_date_content_costs_many_to_one(self.output_less_dates,
                                                                         self.ground_truth)
        )

        self.assertEqual(
            {"rouge_1":
                {
                    "precision": 3 / 14,
                    "recall": 3.5 / 11,
                    "f_score": 2 * (3 / 14) * (3.5 / 11) / (3 / 14 + 3.5 / 11)
                }
            },
            self.evaluator.evaluate_align_date_content_costs_many_to_one(self.output_more_dates,
                                                                         self.ground_truth)
        )

if __name__ == '__main__':
    unittest.main()
