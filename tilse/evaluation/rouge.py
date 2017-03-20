from __future__ import division

import collections

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
    To appear in Proceedings of the 15th Conference of the European Chapter of the Association for Computational
    Linguistics, volume 2: Short Papers, Valencia, Spain, 3-7 April 2017.

    Attributes:
        measures set(str): ROUGE measures to use when computing scores.
            Defaults to `rouge_1`.
        rouge (pyrouge.Rouge155): object to interfer with the ROUGE script.
        alpha (float): Valua controlloing the recall/precision trade-off when
            computing F_alpha scores. Defaults to 1.

    """
    def __init__(self, measures={"rouge_1"}, beta=1):
        """ Initialize the evaluator.

        Args:
            measures set(str): ROUGE measures to use when computing scores.
                Defaults to `rouge_1`.
            beta (float): Valua controlloing the recall/precision trade-off when
                computing F_beta scores. Defaults to 1.
        """
        self.measures = measures
        self.rouge = pyrouge.Rouge155(average="raw", stem=True, ignore_stopwords=True)
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

            if date in pred_dates:
                for measure in self.measures:
                    precision_numerator[measure].append(scores[measure]["prec_num"])
                    precision_denominator[measure].append(scores[measure]["prec_denom"])

            if date in ref_dates:
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
            ("align_date_content_costs_many_to_one", self.evaluate_align_date_content_costs_many_to_one(predicted_timeline, reference_timelines)),
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
                to_add.append(1 - 1/(abs(s_date.toordinal() - t_date.toordinal())+1))

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
                date_factor = 1 - 1/(abs(s_date.toordinal() - t_date.toordinal())+1)

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
