from __future__ import division

import collections
import os

import nltk
import numpy
from scipy import optimize

import pyrouge
from tilse.evaluation import util


class TimelineRougeEvaluator:
    """ Evaluate timelines with respect to a set of reference timelines.

    This class implements several evaluation metrics based on ROUGE to
    compare predicted timelines with a set of reference timelines. The
    measures are described in Martschat and Markert (2017).

    References:

    Sebastian Martschat and Katja Markert (2017).
    Improving ROUGE for Timeline Summarization.
    In Proceedings of the 15th Conference of the European Chapter of the
    Association for Computational Linguistics, volume 2: Short Papers,
    Valencia, Spain, 3-7 April 2017.

    Attributes:
        measures (set(str)): ROUGE measures to use when computing scores.
        rouge (pyrouge.Rouge155 or RougeReimplementation): Object to perform
            ROUGE computation.
        beta (float): Value controlling the recall/precision trade-off when
            computing F_beta scores. Defaults to 1.

    """

    def __init__(self, measures={"rouge_1"}, rouge_computation="original",
                 beta=1):
        """ Initialize the evaluator.

        Args:
            measures (set(str)): ROUGE measures to use when computing scores.
                Defaults to `rouge_1`.
            rouge_computation (str): Whether to use the original ROUGE perl
                script ("original") or an approximate Python reimplementation
                ("reimpl"). Defaults to "original".
            beta (float): Value controlling the recall/precision trade-off when
                computing F_beta scores. Defaults to 1.
        """
        self.measures = measures

        if rouge_computation == "reimpl":
            self.rouge = RougeReimplementation()
        elif rouge_computation == "original":
            self.rouge = pyrouge.Rouge155(average="raw", stem=True,
                                          ignore_stopwords=True)

        self.beta = beta

    def evaluate_concat(self, predicted_timeline, reference_timelines):
        """ Evaluate a predicted timeline w.r.t. a set of reference timelines using the
        'concat' ROUGE variant.

        This variant first concatenates all daily summaries of the respective timelines. The
        resulting documents are then evaluated using the ROUGE measure.

        Args:
            predicted_timeline (data.timelines.Timeline): A timeline.
            reference_timelines (data.timelines.GroundTruth): A ground truth of timelines.

        Returns:
            A dict(str, dict(str, str)) object mapping each ROUGE measure in `self.measures`
            to a dict that maps 'precision', 'recall' and 'f_score' to the corresponding values,
            e.g.

                {"rouge_1": {"precision": 1.0, "recall": 1.0, "f_score": 1.0}}
        """
        pred_sents = []

        for date in sorted(list(predicted_timeline.get_dates())):
            pred_sents.extend([sent.split() for sent in predicted_timeline[date]])

        ref_sents = {}

        for i, timeline in enumerate(reference_timelines.timelines):
            ref_sents[str(i)] = []
            timeline_dates = sorted(list(timeline.get_dates()))
            for date in timeline_dates:
                ref_sents[str(i)].extend([sent.split() for sent in timeline[date]])

        scores = self._get_rouge_counts(pred_sents, ref_sents)

        output_scores = {}

        for measure in self.measures:

            prec = scores[measure]["prec_num"] / scores[measure]["prec_denom"]
            rec = scores[measure]["rec_num"] / scores[measure]["rec_denom"]

            output_scores[measure] = {
                "precision": prec,
                "recall": rec,
                "f_score": util.get_f_score(prec, rec, beta=self.beta)
            }

        return output_scores

    def evaluate_agreement(self, predicted_timeline, reference_timelines):
        """ Evaluate a predicted timeline w.r.t. a set of reference timelines using the
        'agreement' ROUGE variant.

        This variant compares the daily summaries of a date if the date appears in both the
        predicted timeline and in one of the reference timelines.

        Args:
            predicted_timeline (data.timelines.Timeline): A timeline.
            reference_timelines (data.timelines.GroundTruth): A ground truth of timelines.

        Returns:
            A dict(str, dict(str, str)) object mapping each ROUGE measure in `self.measures`
            to a dict that maps 'precision', 'recall' and 'f_score' to the corresponding values,
            e.g.

                {"rouge_1": {"precision": 1.0, "recall": 1.0, "f_score": 1.0}}
        """
        precision_numerator = collections.defaultdict(list)
        precision_denominator = collections.defaultdict(list)
        recall_numerator = collections.defaultdict(list)
        recall_denominator = collections.defaultdict(list)

        pred_dates = predicted_timeline.get_dates()
        ref_dates = reference_timelines.get_dates()

        all_dates = pred_dates.union(ref_dates)

        for date in all_dates:
            temp_groundtruth = reference_timelines[date]
            groundtruth = {}
            for name in temp_groundtruth:
                groundtruth[name] = [sent.split() for sent in temp_groundtruth[name]]

            scores = self._get_rouge_counts(
                [sent.split() for sent in predicted_timeline[date]],
                groundtruth
            )

            for measure in self.measures:
                if date in pred_dates:
                    precision_numerator[measure].append(scores[measure]["prec_num"])
                    precision_denominator[measure].append(scores[measure]["prec_denom"])

                if date in ref_dates:
                    recall_numerator[measure].append(scores[measure]["rec_num"])
                    recall_denominator[measure].append(scores[measure]["rec_denom"])

        output_scores = {}

        for measure in self.measures:
            prec_denom_sum = sum(precision_denominator[measure])

            if prec_denom_sum == 0:
                prec = 0
            else:
                prec = sum(precision_numerator[measure]) / prec_denom_sum

            rec_denom_sum = sum(recall_denominator[measure])

            if rec_denom_sum == 0:
                rec = 0
            else:
                rec = sum(recall_numerator[measure]) / rec_denom_sum

            output_scores[measure] = {
                "precision": prec,
                "recall": rec,
                "f_score": util.get_f_score(prec, rec, beta=self.beta)
            }

        return output_scores

    def evaluate_align_date_costs(self, predicted_timeline, reference_timelines):
        """ Evaluate a predicted timeline w.r.t. a set of reference timelines using an injective alignment
        that uses costs based on date differences.

        This variant first aligns dates in predicted and reference timelines based on costs induced by
        date distance. In then compares the summaries of the aligned dates using ROUGE and weights the
        score by date distance.

        In our EACL'17 paper we denoted this variant as 'align'.

        Args:
            predicted_timeline (data.timelines.Timeline): A timeline.
            reference_timelines (data.timelines.GroundTruth): A ground truth of timelines.

        Returns:
            A dict(str, dict(str, str)) object mapping each ROUGE measure in `self.measures`
            to a dict that maps 'precision', 'recall' and 'f_score' to the corresponding values,
            e.g.

                {"rouge_1": {"precision": 1.0, "recall": 1.0, "f_score": 1.0}}
        """
        return self._evaluate_per_day_mapping_micro(
            predicted_timeline,
            reference_timelines,
            TimelineRougeEvaluator._get_date_costs,
            optimize.linear_sum_assignment
        )

    def evaluate_align_date_content_costs(self, predicted_timeline, reference_timelines):
        """ Evaluate a predicted timeline w.r.t. a set of reference timelines using an injective alignment
        that uses costs based on date differences and content overlap.

        This variant first aligns dates in predicted and reference timelines based on costs induced by
        date distance and content overlap (computed by an approximation of ROUGE-1). It then compares the summaries
        of the aligned dates using ROUGE and weights the score by date distance.

        In our EACL'17 paper we denoted this variant as 'align+'.

        Args:
            predicted_timeline (data.timelines.Timeline): A timeline.
            reference_timelines (data.timelines.GroundTruth): A ground truth of timelines.

        Returns:
            A dict(str, dict(str, str)) object mapping each ROUGE measure in `self.measures`
            to a dict that maps 'precision', 'recall' and 'f_score' to the corresponding values,
            e.g.

                {"rouge_1": {"precision": 1.0, "recall": 1.0, "f_score": 1.0}}
        """
        return self._evaluate_per_day_mapping_micro(
            predicted_timeline,
            reference_timelines,
            TimelineRougeEvaluator._get_date_content_costs,
            optimize.linear_sum_assignment
        )

    def evaluate_align_date_content_costs_many_to_one(
            self,
            predicted_timeline,
            reference_timelines):
        """ Evaluate a predicted timeline w.r.t. a set of reference timelines using a many-to-one
        alignment that uses costs based on date differences and content overlap.

        This variant first (many-to-one-)aligns dates in predicted and reference timelines based on costs induced by
        date distance and content overlap (computed by an approximation of ROUGE-1). It then compares the summaries
        of the aligned dates using ROUGE and weights the score by date distance.

        In our EACL'17 paper we denoted this variant as 'align+ m:1'.

        Args:
            predicted_timeline (data.timelines.Timeline): A timeline.
            reference_timelines (data.timelines.GroundTruth): A ground truth of timelines.

        Returns:
            A dict(str, dict(str, str)) object mapping each ROUGE measure in `self.measures`
            to a dict that maps 'precision', 'recall' and 'f_score' to the corresponding values,
            e.g.

                {"rouge_1": {"precision": 1.0, "recall": 1.0, "f_score": 1.0}}
        """
        return self._evaluate_per_day_mapping_micro(
            predicted_timeline,
            reference_timelines,
            TimelineRougeEvaluator._get_date_content_costs,
            TimelineRougeEvaluator._assign_to_min_cost
        )

    def evaluate_all(self, predicted_timeline, reference_timelines):
        """ Evaluate a predicted timeline w.r.t. a set of reference timelines using the metrics 'concat',
        'agreement', 'align', 'align+' and 'align+ m:1'.

        Args:
            predicted_timeline (data.timelines.Timeline): A timeline.
            reference_timelines (data.timelines.GroundTruth): A ground truth of timelines.

        Returns:
            A collections.OrderedDict object, mapping a description of the metric of the metric to the
            corresponding dict(str, dict(str, str)) object describing precision/recall/f scores for each
            underlying ROUGE measure in `self.measures`.

            Metric      Description
            ------      -----------
            concat      concat
            agreement   agreement
            align       align_date_costs
            align+      align_date_content_costs
            align+ m:1  align_date_content_costs_many_to_one

            One example entry is

                {"concat": {"rouge_1": {"precision": 1.0, "recall": 1.0, "f_score": 1.0}}}
        """
        return collections.OrderedDict([
            ("concat", self.evaluate_concat(predicted_timeline, reference_timelines)),
            ("agreement", self.evaluate_agreement(predicted_timeline, reference_timelines)),
            ("align_date_costs", self.evaluate_align_date_costs(predicted_timeline, reference_timelines)),
            ("align_date_content_costs", self.evaluate_align_date_content_costs(predicted_timeline, reference_timelines)),
            ("align_date_content_costs_many_to_one",
             self.evaluate_align_date_content_costs_many_to_one(predicted_timeline, reference_timelines)),
        ])

    def _evaluate_per_day_mapping_micro(
            self,
            predicted_timeline,
            reference_timelines,
            compute_costs,
            optimize_assignment):
        precision_numerator = collections.defaultdict(list)
        precision_denominator = collections.defaultdict(list)

        recall_numerator = collections.defaultdict(list)
        recall_denominator = collections.defaultdict(list)

        pred_dates = sorted(list(predicted_timeline.get_dates()))
        ref_dates = sorted(list(reference_timelines.get_dates()))

        prec_costs = compute_costs(pred_dates, ref_dates, predicted_timeline,
                                   reference_timelines, axis=0)
        rec_costs = compute_costs(pred_dates, ref_dates, predicted_timeline,
                                  reference_timelines, axis=1)

        prec_row, prec_col = optimize_assignment(prec_costs)
        rec_row, rec_col = optimize_assignment(rec_costs)

        # precision
        for row, col in zip(prec_row, prec_col):
            pred_date = pred_dates[row]
            ref_date = ref_dates[col]

            temp_groundtruth = reference_timelines[ref_date]
            groundtruth = {}
            for name in temp_groundtruth:
                groundtruth[name] = [sent.split() for sent in temp_groundtruth[name]]

            scores = self._get_rouge_counts(
                [sent.split() for sent in predicted_timeline[pred_date]],
                groundtruth
            )

            for measure in self.measures:
                precision_numerator[measure].append(
                    (1 / (abs(pred_date.toordinal() - ref_date.toordinal()) + 1)) * scores[measure]["prec_num"])
                precision_denominator[measure].append(scores[measure]["prec_denom"])

        matched_prec = set(list(prec_row))

        for i, date in enumerate(pred_dates):
            if i not in matched_prec:
                pred_date = pred_dates[i]

                scores = self._get_rouge_counts(
                    [sent.split() for sent in predicted_timeline[pred_date]],
                    {str(i): [[""]] for i, _ in enumerate(reference_timelines.timelines)}
                )

                for measure in self.measures:
                    precision_numerator[measure].append(scores[measure]["prec_num"])
                    precision_denominator[measure].append(scores[measure]["prec_denom"])

        # recall
        for row, col in zip(rec_row, rec_col):
            pred_date = pred_dates[col]
            ref_date = ref_dates[row]

            temp_groundtruth = reference_timelines[ref_date]
            groundtruth = {}
            for name in temp_groundtruth:
                groundtruth[name] = [sent.split() for sent in temp_groundtruth[name]]

            scores = self._get_rouge_counts(
                [sent.split() for sent in predicted_timeline[pred_date]],
                groundtruth
            )

            for measure in self.measures:
                recall_numerator[measure].append(
                    (1 / (abs(pred_date.toordinal() - ref_date.toordinal()) + 1)) * scores[measure]["rec_num"])
                recall_denominator[measure].append(scores[measure]["rec_denom"])

        matched_rec = set(list(rec_row))

        for i, date in enumerate(ref_dates):
            if i not in matched_rec:
                ref_date = ref_dates[i]

                temp_groundtruth = reference_timelines[ref_date]
                groundtruth = {}
                for name in temp_groundtruth:
                    groundtruth[name] = [sent.split() for sent in temp_groundtruth[name]]

                scores = self._get_rouge_counts(
                    [[""]],
                    groundtruth
                )

                for measure in self.measures:
                    recall_numerator[measure].append(scores[measure]["rec_num"])
                    recall_denominator[measure].append(scores[measure]["rec_denom"])

        output_scores = {}

        for measure in self.measures:
            prec_denom_sum = sum(precision_denominator[measure])

            if prec_denom_sum == 0:
                prec = 0
            else:
                prec = sum(precision_numerator[measure]) / prec_denom_sum

            rec_denom_sum = sum(recall_denominator[measure])

            if rec_denom_sum == 0:
                rec = 0
            else:
                rec = sum(recall_numerator[measure]) / rec_denom_sum

            output_scores[measure] = {
                "precision": prec,
                "recall": rec,
                "f_score": util.get_f_score(prec, rec, beta=self.beta)
            }

        return output_scores

    @staticmethod
    def _get_date_costs(source_dates, target_dates, tl, ref_tls, axis=0):
        costs = []

        if axis == 0:
            (a, b) = (source_dates, target_dates)
        elif axis == 1:
            (a, b) = (target_dates, source_dates)

        for s_date in a:
            to_add = []

            for t_date in b:
                to_add.append(1 - 1 / (abs(s_date.toordinal() - t_date.toordinal()) + 1))

            costs.append(to_add)

        return numpy.array(costs)

    @staticmethod
    def _get_date_content_costs(
            source_dates,
            target_dates,
            tl,
            ref_tls,
            axis=0):
        costs = []

        if axis == 0:
            (a, b) = (source_dates, target_dates)
        elif axis == 1:
            (a, b) = (target_dates, source_dates)

        for s_date in a:
            to_add = []
            for t_date in b:
                date_factor = 1 - 1 / (abs(s_date.toordinal() - t_date.toordinal()) + 1)

                date_pred = s_date
                date_ref = t_date

                if axis == 1:
                    date_pred = t_date
                    date_ref = s_date

                content_factor = 1 - util.compute_rouge_approximation(
                    tl[date_pred],
                    [ref_tls[date_ref][name] for name in ref_tls[date_ref]]
                )

                to_add.append(date_factor * content_factor)

            costs.append(to_add)

        return numpy.array(costs)

    @staticmethod
    def _assign_to_min_cost(cost_matrix):
        row_indices = []
        column_indices = []
        for i, row in enumerate(cost_matrix):
            row_indices.append(i)
            column_indices.append(row.argmin())

        return numpy.array(row_indices), numpy.array(column_indices)

    def _get_rouge_counts(self, pred, ref):
        scores = {}

        temp_scores = self.rouge.score_summary(pred, ref)

        for measure in self.measures:
            scores[measure] = {}

            scores[measure]["prec_num"] = temp_scores[measure + "_h_count"]
            scores[measure]["prec_denom"] = temp_scores[measure + "_p_count"]
            scores[measure]["rec_num"] = temp_scores[measure + "_h_count"]
            scores[measure]["rec_denom"] = temp_scores[measure + "_m_count"]

        return scores


class RougeReimplementation:
    """
    An approximate reimplementation of ROUGE-1 and ROUGE-2.

    It does not exactly match scores from the Perl script. It therefore
    should not be used for computing scores on development and test
    sets when preparing results for papers or for comparison to other
    systems. However, due to improved speed it is useful during development
    (scores also should not differ too much from the original
    implementation).

    Attributes:
        stem (bool): Whether to stem words before evaluation.
        ignore_stopwords (bool): Whether to ignore stopwords before
            evaluation.
        porter_stemmer (PorterStemmer): nltk's implementation of the
            Porter stemmer.
        stem_function (func): Utility function for performing stemming.
        stopwords (set(str)): Stopwords, set to the list used in
            ROUGE's Perl evaluation script.
    """
    def __init__(self, stem=True, ignore_stopwords=True):
        """
        Initializes ROUGE reimplementation.

        Params:
            stem (bool): Whether to stem words before evaluation. Defaults
                to True.
            ignore_stopwords (bool): Whether to ignore stopwords before
                evaluation. Defaults to True.
        """
        self.stem = stem
        self.ignore_stopwords = ignore_stopwords
        self.stopwords = set()
        self.porter_stemmer = nltk.stem.PorterStemmer()

        self.stem_function = self._identity

        if stem:
            self.stem_function = self._robust_porter_stemmer

        dir_path = os.path.dirname(os.path.realpath(__file__))

        if ignore_stopwords:
            with open(dir_path + "/../../pyrouge/tools/ROUGE-1.5.5/data/smart_common_words.txt") as my_file:
                self.stopwords = set(my_file.read().splitlines())

    def score_summary(self, summary, references):
        """
        Scores a summary with ROUGE-1 and ROUGE-2.

        Params:
            summary (list(list(str))): A list of tokenized sentences,
                representing a predicted summary.

            references dict(int, list(list(str))): A mapping of integers
                to lists of tokenized sentences, representing reference
                summaries.

        Returns:
              A mapping from strings to integers, with the
              following meaning (same representation as pyrouge):
                "rouge_1_h_count": ROUGE-1 recall/precision numerator,
                "rouge_1_p_count": ROUGE-1 precision denominator,
                "rouge_1_m_count": ROUGE-1 recall denominator.

                Analogous for ROUGE-2.
        """
        punctuation = [".", ",", ";", ":", "``", "''", "-", '"']

        to_ignore = self.stopwords.union(punctuation)

        pred_tokens_lowercased = [self.stem_function(k.lower()) for sent in summary for k in sent
                                  if k.lower() not in to_ignore]

        ref_tokens_lowercased = {}

        for i, ref_summary in references.items():
            ref_tokens_lowercased[i] = [self.stem_function(k.lower()) for sent in ref_summary for k
                                        in sent if k.lower() not in to_ignore]

        eval_scores = {}
        eval_scores.update(
            self._rouge_1(pred_tokens_lowercased, ref_tokens_lowercased))
        eval_scores.update(
            self._rouge_2(pred_tokens_lowercased, ref_tokens_lowercased))

        return eval_scores

    def _identity(self, x):
        return x

    def _robust_porter_stemmer(self, x):
        stem = x

        try:
            stem = self.porter_stemmer.stem(x)
        except IndexError:
            pass

        return stem

    def _rouge_1(self, pred_tokens, ref_tokens):
        # unigrams
        pred_counts = collections.Counter(pred_tokens)

        ref_counts = {}

        for i, tokens in ref_tokens.items():
            ref_counts[i] = collections.Counter(tokens)

        # approximate ROUGE-1 score
        match = 0
        for tok in pred_counts:
            match += sum([min(pred_counts[tok], ref_counts[x][tok]) for x in
                          ref_counts.keys()])

        prec_denom = (len(ref_counts.keys()) * sum(pred_counts.values()))

        recall_denom = sum([sum(ref_counts[x].values()) for x in ref_counts])

        return {
            "rouge_1_h_count": match,
            "rouge_1_p_count": prec_denom,
            "rouge_1_m_count": recall_denom,
        }

    def _rouge_2(self, pred_tokens, ref_tokens):
        pred_counts = collections.Counter(zip(pred_tokens, pred_tokens[1:]))

        ref_counts = {}

        for i, tokens in ref_tokens.items():
            ref_counts[i] = collections.Counter(zip(tokens, tokens[1:]))

        # approximate ROUGE-1 score
        match = 0
        for tok in pred_counts:
            match += sum([min(pred_counts[tok], ref_counts[x][tok]) for x in
                          ref_counts.keys()])

        prec_denom = (len(ref_counts.keys()) * sum(pred_counts.values()))

        recall_denom = sum([sum(ref_counts[x].values()) for x in ref_counts])

        return {
            "rouge_2_h_count": match,
            "rouge_2_p_count": prec_denom,
            "rouge_2_m_count": recall_denom,
        }
