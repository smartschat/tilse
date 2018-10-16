import logging
import math
from collections import defaultdict, Counter

import numpy
from numba import jit

from tilse.data import timelines
from tilse.models import models
from tilse.util.import_helper import import_helper


class Submodular(models.Model):
    """
    Predicts timelines using submodular optimization.

    Timelines are constructed using a greedy algorithm optimizing a submodular
    objective function under suitable constraints.

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
        semantic_cluster_computation_function (function): A function for
            computing semantic clusters of sentences, see the functions in the
            module `tilse.models.submodular.semantic_cluster_functions`.
        date_cluster_computation_function (function): A function for
            computing date clusters of sentences, see the functions in the
            module `tilse.models.submodular.date_cluster_functions`.
        params (tuple(float)): A tuple specifying coefficients for parts of the
            submodular objective function (in order: coverage, diversity w.r.t.
            semantic clusters, diversity w.r.t. date clusters, date references).

    """
    def __init__(self, config, rouge):
        """
        Initializes a submodular model for timeline summarization.

        Params:
            config (dict): A configuration dict. needs to have at least entries for `"assess_length"`
                (should be a function from `tilse.models.assess_length`,
                `"sentence_representations"` (should be a class from
                `tilse.representations.sentence_representations`), and nested entries
                for `"properties"`:
                    * `"constraint"`: Function from `tilse.models.submodular.constraints`
                    * `"semantic_cluster"`: Function from `tilse.models.submodular.semantic_cluster_functions`
                    * `"date_cluster"`: Function from `tilse.models.submodular.date_cluster_functions`
                    * `"coefficients"`: List of floats of length 4.

            rouge (pyrouge.Rouge155 or tilse.evaluation.rouge.RougeReimplementation): An object for computing ROUGE scores.

        Returns:
            A model for timeline summarization intialized with the above parameters.
        """
        super(Submodular, self).__init__(config, rouge)

        self.is_valid_function = import_helper(
            "tilse.models.submodular.constraints",
            config["properties"]["constraint"]
        )
        self.semantic_cluster_computation_function = import_helper(
            "tilse.models.submodular.semantic_cluster_functions",
            config["properties"]["semantic_cluster"]
        )
        self.date_cluster_computation_function = import_helper(
            "tilse.models.submodular.date_cluster_functions",
            config["properties"]["date_cluster"]
        )
        self.params = config["properties"]["coefficients"]

    def predict(self, corpus, preprocessed_information, timeline_properties, params):
        """
        Predicts a timeline. For details on how the prediction works,
        see the docstring for this class and Martschat and Markert (CoNLL 2018):
        A temporally sensitive submodularity framework for timeline
        summarization.

        Params:
            corpus (tilse.data.corpora.Corpus): A corpus.
            preprocessed_information (object): Sentence ranks and extents
                obtained from preprocessing.
            timeline_properties (tilse.models.timeline_properties.TimelineProperties): Properties of the timeline to
                predict.
            params (tuple(float)): A tuple specifying coefficients for parts of the
                submodular objective function (in order: coverage, diversity w.r.t.
                semantic clusters, diversity w.r.t. date clusters, date references).

        Returns:
            A timeline (tilse.data.timelines.Timeline).
        """
        (coeff_coverage,
         coeff_semantic_redundancy,
         coeff_date_redundancy,
         coeff_date_references) = params

        all_sents = []
        all_sent_dates = []

        for doc in corpus.docs:
            for sent in doc:
                all_sents.append(sent)
                all_sent_dates.append(sent.date)

        (coverage_values,
         sent_cluster_indices_semantic,
         sent_cluster_indices_date,
         sent_date_indices,
         date_references,
         singleton_rewards_semantic,
         singleton_rewards_date) = preprocessed_information

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

        dates_selected = numpy.zeros(len(date_references))

        # contains partially precomputed per-cluster sums to facilitate greedy
        # algorithm diversity difference computation
        sums_semantic = numpy.zeros(max(sent_cluster_indices_semantic) + 1)
        sums_date = numpy.zeros(max(sent_cluster_indices_date) + 1)

        while candidate_indices:
            # numba workaround (cannot handle empty lists)
            if not selected_sent_indices:
                selected_sent_indices.append(-1)

            index, val = _objective_function(
                candidate_indices,
                coverage_values,
                singleton_rewards_semantic,
                singleton_rewards_date,
                sent_cluster_indices_semantic,
                sent_cluster_indices_date,
                sums_semantic,
                sums_date,
                sent_date_indices,
                dates_selected,
                date_references,
                coeff_coverage,
                coeff_semantic_redundancy,
                coeff_date_redundancy,
                coeff_date_references
            )

            if val >= 0:
                selected_sent_indices.append(index)
                date_to_sent_mapping[all_sent_dates[index]].append(all_sents[index])

                sums_semantic[
                    sent_cluster_indices_semantic[index]
                ] += singleton_rewards_semantic[index]

                sums_date[sent_cluster_indices_date[index]] += singleton_rewards_date[index]

                dates_selected[sent_date_indices[index]] = 1

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
        Returns parameters for the model read from the config dict when it was
        initialized.

        Params:
            corpora (dict(str, tilse.data.corpora.Corpus)): A mapping of topic names to corresponding corpora.
            preprocessed_information (object): Arbitrary information obtained from preprocessing.
            reference_timelines (dict(str, tilse.data.timelines.Groundtruth)): A mapping of topic names
                to corresponding reference timelines.
            topic_to_evaluate (str): The topic to evaluate (must be a key in `corpora`. The given topic will not
                be used during training (such that it can serve as evaluation data later).

        Returns:
            A four-tuple of floats, specifiying coefficients for coverage, diversity w.r.t.
            semantic clusters, diversity w.r.t. date clusters, date references
        """
        return self.params

    def preprocess(self, topic_name, corpus):
        """
        Computes various information for use in the objective function. For
        details, see below.

        Params:
            topic_name (str): name of the topic to which the corpus belongs.
            corpus (tilse.data.corpora.Corpus): A corpus.

        Returns:
            A 7-tuple containing:
                coverage_values (numpy.array): Coverage values for all
                    sentences.
                sent_cluster_indices_semantic (list(int)): Sentence cluster
                    indices for semantic cluster function.
                sent_cluster_indices_date (list(int)): Sentence cluster
                    indices for date cluster function.
                sent_date_indices list(int): Date index for each sentence (two
                    sentences have the same date iff they have the same index).
                date_references (numpy.array): For each date (represented by
                    its index), the number of references to it in the corpus.
                singleton_rewards_semantic (numpy.array): Singleton rewards
                    for each sentence according to semantic cluster function.
                singleton_rewards_date (numpy.array): Singleton rewards
                    for each sentence according to date cluster function.
        """
        all_sents = [sent for doc in corpus for sent in doc]

        logging.info("Compute sentence similarities")
        sims = self.sentence_representation_computer(corpus).compute_pairwise_similarities()

        logging.info("Cluster sentence representations")
        sent_cluster_indices_semantic = self.semantic_cluster_computation_function(corpus)
        sent_cluster_indices_date = self.date_cluster_computation_function(corpus)

        # assign indices to dates
        date_to_index = self._get_date_to_index_mapping(all_sents)

        # map sentence indicies to indices of their dates
        sent_date_indices = numpy.zeros(len(all_sents), dtype=int)
        for i, sent in enumerate(all_sents):
            sent_date_indices[i] = date_to_index[sent.date]

        logging.info("Compute date references")
        date_references = numpy.zeros(len(date_to_index))

        dates_that_are_referred_to = []

        for doc in corpus:
            for sent in doc:
                if sent.date != doc.publication_date and sent.time_span == "d":
                    # found reference to sent.date
                    dates_that_are_referred_to.append(sent.date)

        dates_with_frequency = Counter(dates_that_are_referred_to)

        # maps date index to how often date was referred to
        for date, i in date_to_index.items():
            date_references[i] = dates_with_frequency[date]

        logging.info("Compute singleton rewards")
        singleton_rewards_semantic = numpy.zeros(len(all_sents))
        singleton_rewards_date = numpy.zeros(len(all_sents))

        # compute singleton reward of each sentence as total similarity to
        # all sentences
        for i in range(len(all_sents)):
            for j in range(len(all_sents)):
                singleton_rewards_semantic[i] += sims[i, j]
                singleton_rewards_date[i] += sims[i, j]

        logging.info("Normalize objective function values")
        sims = sims / sims.sum()
        date_references = date_references / date_references.sum()

        coverage_values = numpy.sum(sims, axis=1)

        cluster_to_sents_sum_semantic = defaultdict(float)
        cluster_to_sents_sum_date = defaultdict(float)
        for i, clust in enumerate(sent_cluster_indices_semantic):
            cluster_to_sents_sum_semantic[clust] += singleton_rewards_semantic[i]

        for i, clust in enumerate(sent_cluster_indices_date):
            cluster_to_sents_sum_date[clust] += singleton_rewards_date[i]

        singleton_rewards_semantic = singleton_rewards_semantic / sum(
            [math.sqrt(x) for x in cluster_to_sents_sum_semantic.values()]
        )
        singleton_rewards_date = singleton_rewards_date / sum(
            [math.sqrt(x) for x in cluster_to_sents_sum_date.values()]
        )

        return (coverage_values,
                sent_cluster_indices_semantic,
                sent_cluster_indices_date,
                sent_date_indices,
                date_references,
                singleton_rewards_semantic,
                singleton_rewards_date)

    def _get_date_to_index_mapping(self, sents):
        date_to_index = {}
        i = 0
        for sent in sents:
            date = sent.date
            if date not in date_to_index:
                date_to_index[date] = i
                i += 1

        return date_to_index


@jit(nopython=True)
def _objective_function(candidate_indices,
                        coverage_values,
                        singleton_rewards_semantic,
                        singleton_rewards_date,
                        sent_cluster_indices_semantic,
                        sent_cluster_indices_date,
                        sums_semantic,
                        sums_date,
                        sent_date_indices,
                        dates_selected,
                        date_references,
                        coeff_coverage,
                        coeff_semantic_redundancy,
                        coeff_date_redundancy,
                        coeff_date_references):
    best = -1
    best_val = -numpy.inf

    for i in candidate_indices:
        # coverage
        my_sum = coeff_coverage * coverage_values[i]

        # redundancy...

        # ...via semantic clusters
        cluster_of_sent_semantic = sent_cluster_indices_semantic[i]
        sum_before = sums_semantic[cluster_of_sent_semantic]
        my_sum += coeff_semantic_redundancy * (
            math.sqrt(sum_before + singleton_rewards_semantic[i]) - math.sqrt(sum_before)
        )

        # ...via date clusters
        cluster_of_sent_date = sent_cluster_indices_date[i]
        sum_before = sums_date[cluster_of_sent_date]
        my_sum += coeff_date_redundancy * (
            math.sqrt(sum_before + singleton_rewards_date[i]) - math.sqrt(sum_before)
        )

        # date references
        date_index = sent_date_indices[i]
        if dates_selected[date_index] == 0:
            my_sum += coeff_date_references * date_references[date_index]

        # update best
        if my_sum > best_val:
            best = i
            best_val = my_sum

    return best, best_val
