import logging
from collections import defaultdict

import numpy
from numba import jit

from tilse.data import timelines
from tilse.evaluation.rouge import RougeReimplementation
from tilse.evaluation.util import get_f_score
from tilse.models import models
from tilse.util.import_helper import import_helper


class UpperBounds(models.Model):
    """
     Approximates an upper bound for timeline prediction using oracle
     information.

     Timelines are constructed using a greedy algorithm optimizing (under
     suitable constraints) a submodular objective function containing ROUGE
     scores for sentences.

     For more details, see Martschat and Markert (CoNLL 2018): A temporally
     sensitive submodularity framework for timeline summarization.

     Attributes:
         summary_length_assesor (function(Groundtruth, int)): A function to assess length of
             daily summaries given a reference groundtruth.
         sentence_representation_computer (tilse.representations.sentence_representations.SentenceRepresentation): A model for computing sentence
             representations.
         rouge (pyrouge.Rouge155 or tilse.evaluation.rouge.RougeReimplementation): An object for computing ROUGE scores.
         is_valid_function (function): A function specifying constraints, see
             the functions in the module `tilse.models.submodular.constraints`.
     """
    def __init__(self, config, rouge):
        """
        Initializes an upper bound computation for timeline summarization.

        Params:
            config (dict): A configuration dict. needs to have at least entries for `"assess_length"`
                (should be a function from `tilse.models.assess_length`,
                `"sentence_representations"` (should be a class from
                `tilse.representations.sentence_representations`), and nested entries
                for `"properties"`:
                    * `"constraint"`: Function from `tilse.models.submodular.constraints`

            rouge (pyrouge.Rouge155 or tilse.evaluation.rouge.RougeReimplementation): An object for computing ROUGE scores.

        Returns:
            An model for computing upper bounds for timeline summarization
            intialized with the above parameters.
        """
        super(UpperBounds, self).__init__(config, rouge)

        self.is_valid_function = import_helper(
            "tilse.models.submodular.constraints",
            config["properties"]["constraint"]
        )

    def predict(self, corpus, preprocessed_information, timeline_properties, params):
        """
        Predicts a timeline with oracle information, yielding an
        approximation of an upper bound for timeline summarization.
        For details on how the prediction works, see the docstring for this
        class and Martschat and Markert (CoNLL 2018): A temporally sensitive
        submodularity framework for timeline summarization.

        Params:
            corpus (tilse.data.corpora.Corpus): A corpus.
            preprocessed_information (object): Sentence ranks and extents
                obtained from preprocessing.
            timeline_properties (tilse.models.timeline_properties.TimelineProperties): Properties of the timeline to
                predict.
            params (dict(tilse.models.timeline_properties.TimelineProperties, numpy.array)): A mapping of timeline properties
                to an numpy array of per-day ROUGE-1 F1 scores for all sentences
                in the corresponding corpus.

        Returns:
            A timeline (tilse.data.timelines.Timeline).
        """
        # that's a hack...
        rouge_vals = params[timeline_properties]

        all_sents = []
        all_sent_dates = []

        for doc in corpus.docs:
            for sent in doc:
                all_sents.append(sent)
                all_sent_dates.append(sent.date)


        logging.info("Run greedy algorithm")

        # greedy algorithm
        date_to_sent_mapping = defaultdict(list)
        selected_sent_indices = list()
        unselected_sent_indices = list(range(len(all_sents)))
        candidate_indices = [k for k in unselected_sent_indices
                             if self.is_valid_function(k,
                                                       date_to_sent_mapping,
                                                       all_sent_dates,
                                                       timeline_properties)]

        while candidate_indices:
            # numba workaround (cannot handle empty lists)
            if not selected_sent_indices:
                selected_sent_indices.append(-1)

            index, val = _rouge_submodular(
                candidate_indices,
                rouge_vals
            )

            if val >= 0:
                selected_sent_indices.append(index)
                date_to_sent_mapping[all_sent_dates[index]].append(all_sents[index])

            # numba workaround
            if selected_sent_indices[0] == -1:
                selected_sent_indices = selected_sent_indices[1:]

            unselected_sent_indices.remove(index)

            candidate_indices = [k for k in unselected_sent_indices if
                                 self.is_valid_function(k,
                                                        date_to_sent_mapping,
                                                        all_sent_dates,
                                                        timeline_properties)]

        return timelines.Timeline.from_sent_objects(date_to_sent_mapping)

    def train(self, corpora, preprocessed_information, timelines, topic_to_evaluate):
        """
        Computes per-day ROUGE F1 for each sentence in the corpus for
        `topic_to_evaluate` (This is quite a misuse of the semantics of
        this function).

        Params:
            corpora (dict(str, tilse.data.corpora.Corpus)): A mapping of topic names to corresponding corpora.
            preprocessed_information (object): Arbitrary information obtained from preprocessing.
            reference_timelines (dict(str, tilse.data.timelines.Groundtruth)): A mapping of topic names
                to corresponding reference timelines.
            topic_to_evaluate (str): The topic to evaluate (must be a key in `corpora`. The given topic will not
                be used during training (such that it can serve as evaluation data later).

        Returns:
            A mapping of timeline properties for each of the timelines in
            `timelines[`topic_to_evaluate`]` to an numpy array of per-day
            ROUGE-1 F1 scores for all sentences in the corresponding corpus.

        """
        rouge = RougeReimplementation()
        corpus = corpora[topic_to_evaluate]
        reference_timelines = timelines[topic_to_evaluate]

        rouge_vals = {}

        for tl in reference_timelines.timelines:
            tp = self.get_timeline_properties(tl)

            rouge_vals[tp] = []

            for doc in corpus.docs:
                for sent in doc:
                    sent_processed = [[x.content for x in sent]]
                    ref_processed = {"0": [[x for x in s.split()] for s in
                                           tl[sent.date]]}

                    rouge_computed = rouge.score_summary(sent_processed,
                                                         ref_processed)

                    if rouge_computed["rouge_1_p_count"] == 0:
                        prec = 0
                    else:
                        prec = rouge_computed["rouge_1_h_count"] / \
                               rouge_computed["rouge_1_p_count"]

                    if rouge_computed["rouge_1_m_count"] == 0:
                        rec = 0
                    else:
                        rec = rouge_computed["rouge_1_h_count"] / \
                              rouge_computed["rouge_1_m_count"]

                    f1 = get_f_score(prec, rec)

                    rouge_vals[tp].append(f1)

            rouge_vals[tp] = numpy.array(rouge_vals[tp])

        return rouge_vals

    def preprocess(self, topic_name, corpus):
        """
        No functionality, this model does not use any preprocessing.
        """
        pass


@jit(nopython=True)
def _rouge_submodular(candidate_indices, rouge_vals):
    best = -1
    best_val = -numpy.inf

    for i in candidate_indices:
        my_sum = rouge_vals[i]

        if my_sum > best_val:
            best = i
            best_val = my_sum

    return best, best_val
