import logging

import numpy
from numba import jit

from tilse.models import models, post_processing
from tilse.util.import_helper import import_helper


class Chieu(models.Model):
    """
    Predicts timelines using the model of Chieu and Lee (2004): Query-based
    event extraction along a timeline.

    It operates in two stages. First, it ranks sentences based on similarity:
    for each sentence s, similarities to all sentences in a 10-day window around
    the date of s are summed up. This yields a ranked list of sentences,
    sorted by highest to lowest summed up similarities. Using this list, a
    timeline containing one-sentence daily summaries is constructed as follows:
    iterating through the ranked sentence list, a sentence is added to the
    timeline depending on the extent of the sentences already in the
    timeline.

    The extent is computed using an "interest" function that can be supplied
    when initializing an object of this class. If the candidate sentence does
    not fall into the extent of any sentence already in the timeline,
    it is added to the timeline.

    Attributes:
        summary_length_assesor (function(Groundtruth, int)): A function to assess length of
            daily summaries given a reference groundtruth.
        sentence_representation_computer (tilse.representations.sentence_representations.SentenceRepresentation): A model for computing sentence
            representations.
        rouge (pyrouge.Rouge155 or tilse.evaluation.rouge.RougeReimplementation): An object for computing ROUGE scores.
        interest_measure (function): A function for computing interest,
            for examples see the module `tilse.models.chieu.interest_measures`
    """
    def __init__(self, config, rouge):
        super(Chieu, self).__init__(config, rouge)
        self.interest_measure = import_helper("tilse.models.chieu.interest_measures", config["properties"]["interest_measure"])

    def predict(self,
            corpus,
            preprocessed_information,
            timeline_properties,
            params):
        """
        Predicts a timeline. For details on how the prediction works,
        see the docstring for this class.

        Params:
            corpus (tilse.data.corpora.Corpus): A corpus.
            preprocessed_information (object): Sentence ranks and extents
                obtained from preprocessing.
            timeline_properties (tilse.models.timeline_properties.TimelineProperties): Properties of the timeline to
                predict.
            params (object): Information obtained from training -- `None`, since
                this model is unsupervised.

        Returns:
            A timeline (tilse.data.timelines.Timeline).
        """

        sents = [sent for doc in corpus for sent in doc]

        ranked_sentences, extents = preprocessed_information

        logging.info("\tPost processing")
        post_processed = post_processing.post_process(
            [sents[i] for i in reversed(ranked_sentences.argsort())],
            [extents[i] for i in reversed(ranked_sentences.argsort())],
            1, # daily summary length
            timeline_properties.num_sentences, # timeline length
            start=timeline_properties.start,
            end=timeline_properties.end
            )

        return post_processed

    def preprocess(self, topic_name, corpus):
        """
        Computes sentence ranks and extents.

        Params:
            topic_name (str): name of the topic to which the corpus belongs.
            corpus (tilse.data.corpora.Corpus): A corpus.

        Returns:
            A tuple consisting of a list of sentence ids with ranks (list(
            float)), and extents for the sentences (list(int)).
        """
        logging.info("\tSimilarities")
        similarities = self.sentence_representation_computer(corpus).compute_pairwise_similarities()

        logging.info("\tDates to ordinal")
        dates_in_ordinal = Chieu._get_dates_to_ordinal(corpus)

        logging.info("\tDate diffs")
        date_diffs = Chieu._get_date_diffs(dates_in_ordinal)

        logging.info("\tRanking")
        ranked_sentences, extents = self.interest_measure(similarities, date_diffs)

        return ranked_sentences, extents

    def train(self, corpora, preprocessed_information, timelines, timeline_to_evaluate):
        """
        No functionality since the model of Chieu and Lee is unsupervised.
        """
        pass

    @staticmethod
    def _get_dates_to_ordinal(corpus):
        dates_in_ordinal = []
        for doc in corpus:
            for sent in doc:
                dates_in_ordinal.append(sent.date.toordinal())

        return numpy.array(dates_in_ordinal)

    @staticmethod
    @jit
    def _get_date_diffs(dates_in_ordinal):
        date_diffs = numpy.zeros((len(dates_in_ordinal), len(dates_in_ordinal)),
                                 dtype=numpy.uint32)

        for i in range(len(dates_in_ordinal)):
            for j in range(0, i):
                date1 = dates_in_ordinal[i]
                date2 = dates_in_ordinal[j]
                diff = abs(date1 - date2)
                date_diffs[i][j] = diff
                date_diffs[j][i] = diff

        return date_diffs
