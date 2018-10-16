import datetime
import random

from tilse.data import timelines
from tilse.evaluation import scores


class MetricTests:
    """ Run tests for metrics for evaluating timeline summarization.

    This class implements tests similar to the tests derscribed in Martschat and Markert (2017).

    References:

    Sebastian Martschat and Katja Markert (2017).
    Improving ROUGE for Timeline Summarization.
    To appear in Proceedings of the 15th Conference of the European Chapter of the Association for Computational
    Linguistics, volume 2: Short Papers, Valencia, Spain, 3-7 April 2017.

    Attributes:
        rouge (evaluation.TimelineRougeEvaluator): an object implementing the metrics to check.

    """

    def __init__(self, rouge):
        """ Initialize metric tests.

        Args:
            rouge (evaluation.TimelineRougeEvaluator): an object implementing the metrics to check.
        """
        self.rouge = rouge
        random.seed(23)

    def test_metrics(self, topics_to_groundtruth):
        """ Test metrics as described in Martschat and Markert (2017).

        In this implementation, we slightly changed the 'add' test described in the paper: we
        do not add the first sentence from the first article of the day, but instead add a sentence
        consisting of 10 'asdf' tokens. This removes the dependence on a corpus.

        Args:
            topics_to_groundtruth (dict(str, Groundtruth)): a mapping of topics to a Groundtruth object describing
                the groundtruth timelines for the topic.

        Returns:
            Two lists `names`, `results`. `names[i]` contains the name of the ith test, while `results[i]` contains
            a `Scores` objects describing the results of the ith test.

        """
        test_names_and_functions = [
            ("remove_random", MetricTests._metric_test_remove_random),
            ("add", MetricTests._metric_test_add_nonsense),
            ("merge", MetricTests._metric_test_merge),
            ("shift_1_day", MetricTests._metric_test_shift_1_day),
            ("shift_5_days", MetricTests._metric_test_shift_5_days)
        ]

        test_results = []

        ref_tl_names = []
        ref_tls = []

        for topic in sorted(topics_to_groundtruth.keys()):
            for i, tl in enumerate(topics_to_groundtruth[topic].timelines, 1):
                ref_tl_names.append(topic + "_" + str(i))
                ref_tls.append(tl)

        for _, test_function in test_names_and_functions:
            test_results.append(
                self._run_test(
                    test_function,
                    ref_tls,
                    ref_tl_names,
                )
            )

        return [name for name, _ in test_names_and_functions], test_results

    def _run_test(self, test_to_run, ref_tls, names):
        results = {}

        for name, ref_tl in zip(names, ref_tls):
            tl = timelines.Timeline({})
            for date, sents in ref_tl.dates_to_summaries.items():
                tl.dates_to_summaries[date] = [sent for sent in sents]

            test_to_run(tl)

            results[name] = self.rouge.evaluate_all(tl, timelines.GroundTruth([ref_tl]))

        return scores.Scores(results)

    @staticmethod
    def _metric_test_remove_random(tl):
        tl.dates_to_summaries.pop(random.choice(
            sorted(list(tl.dates_to_summaries.keys()))))

    @staticmethod
    def _metric_test_add_nonsense(tl):
        # determine date
        date_set = set(tl.get_dates())

        for date in sorted(list(date_set)):
            plus_one = date + datetime.timedelta(1)

            if plus_one not in date_set:
                to_add = ["asdf"] * 10

                tl.dates_to_summaries[plus_one] = \
                    [" ".join(to_add)]

                return

        raise ValueError("Could not add date for meaningful test")

    @staticmethod
    def _metric_test_merge(tl):
        tl_dates = sorted(list(tl.dates_to_summaries.keys()))
        tl_dates_set = tl.dates_to_summaries.keys()

        i = 1

        while True:
            for date in tl_dates:
                mod_date = date + datetime.timedelta(i)
                if mod_date in tl_dates_set:
                    tl.dates_to_summaries[date].extend(
                        tl.dates_to_summaries.pop(mod_date)
                    )

                    return

            i += 1

    @staticmethod
    def _metric_test_shift_1_day(tl):
        new_mapping = {}

        for date in tl.dates_to_summaries.keys():
            new_mapping[date + datetime.timedelta(1)] = tl.dates_to_summaries[date]

        tl.dates_to_summaries = new_mapping

    @staticmethod
    def _metric_test_shift_5_days(tl):
        new_mapping = {}

        for date in tl.dates_to_summaries.keys():
            new_mapping[date + datetime.timedelta(5)] = tl.dates_to_summaries[date]

        tl.dates_to_summaries = new_mapping
