import math
from collections import Counter

import numpy
import textacy
from sklearn import linear_model, feature_extraction

from tilse.evaluation import util
from tilse.evaluation.rouge import RougeReimplementation
from tilse.models import models, post_processing


class Regression(models.Model):
    """
    Predicts timelines using regression of per-day ROUGE-1 F1 score.

    Represents sentences with length, number NEs, avg/sum tf-idf scores and
    lexical unigram features, and learns a regression model using
    sklearn's LinearRegression with default parameters except for
    normalize=True.

    When predicting, sentences are selected greedily respecting
    the constraints of maximum number of days in a timeline and maximum
    daily summary length.

    Attributes:
        summary_length_assesor (function(Groundtruth, int)): A function to assess length of
            daily summaries given a reference groundtruth.
        sentence_representation_computer (tilse.representations.sentence_representations.SentenceRepresentation): A model for computing sentence
            representations.
        rouge (pyrouge.Rouge155 or tilse.evaluation.rouge.RougeReimplementation): An object for computing ROUGE scores.
        model (sklearn.linear_model.LinearRegression): Machine learning model
            used for regression
        vectorizer (sklearn.feature_extraction.DictVectorizer): Vectorizer
            used for computing features for sentences.

    """
    def __init__(self, config, rouge):
        super(Regression, self).__init__(config, rouge)

        self.model = linear_model.LinearRegression(normalize=True)
        self.vectorizer = feature_extraction.DictVectorizer()

    def train(self, corpora, preprocessed_information, reference_timelines, timeline_to_evaluate):
        """
        Trains the model.

        For details on training, see the docstring of this class.

        Params:
            corpora (dict(str, tilse.data.corpora.Corpus)): A mapping of topic names to corresponding corpora.
            preprocessed_information (object): Arbitrary information obtained from preprocessing.
            reference_timelines (dict(str, tilse.data.timelines.Groundtruth)): A mapping of topic names
                to corresponding reference timelines.
            topic_to_evaluate (str): The topic to evaluate (must be a key in `corpora`. The given topic will not
                be used during training (such that it can serve as evaluation data later).

        Returns:
            Nothing, `self.model` is updated.
        """
        rouge = RougeReimplementation()

        features = []
        f1_scores = []

        for t in corpora:
            if t == timeline_to_evaluate:
                continue

            corpus = corpora[t]
            sum_tfidf, avg_tfidf = preprocessed_information[t]

            i = 0
            for doc in corpus:
                for sent in doc:
                    sent_processed = [[x.content for x in sent]]

                    ref_temp = reference_timelines[t][sent.date]

                    ref_processed = {}

                    for k, sents in ref_temp.items():
                        ref_processed[k] = [[x for x in s.split()] for s in
                                            sents]

                    rouge_computed = rouge.score_summary(sent_processed, ref_processed)

                    if rouge_computed["rouge_1_p_count"] == 0:
                        prec = 0
                    else:
                        prec = rouge_computed["rouge_1_h_count"] / rouge_computed["rouge_1_p_count"]

                    if rouge_computed["rouge_1_m_count"] == 0:
                        rec = 0
                    else:
                        rec = rouge_computed["rouge_1_h_count"] / rouge_computed["rouge_1_m_count"]

                    f1 = util.get_f_score(prec, rec)

                    features.append(Regression._compute_features_for_sent(
                        sent, i, sum_tfidf, avg_tfidf))

                    f1_scores.append(f1)

                    i += 1

        vectorized = self.vectorizer.fit_transform(features)

        self.model.fit(vectorized, f1_scores)

    def predict(self, corpus, preprocessed_information, timeline_properties, params):
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
        ranked_sentences = self._get_ranked_sentences(corpus, preprocessed_information)

        sents = [sent for doc in corpus for sent in doc]

        post_processed = post_processing.post_process(
            [sents[i] for i in reversed(ranked_sentences.argsort())],
            None,
            timeline_properties.daily_summary_length,
            timeline_properties.num_dates,
            start=timeline_properties.start,
            end=timeline_properties.end)

        return post_processed

    def _get_ranked_sentences(self, corpus, preprocessed_information):
        features = []

        sum_tfidf, avg_tfidf = preprocessed_information

        i = 0

        for doc in corpus:
            for sent in doc:
                features.append(Regression._compute_features_for_sent(sent, i, sum_tfidf, avg_tfidf))

        vectorized = self.vectorizer.transform(features)

        ranked_sentences = self.model.predict(vectorized)

        return ranked_sentences + math.fabs(min(ranked_sentences))

    def preprocess(self, topic_name, corpus):
        """
        Computes tf-idf scores for sentences by summing/averaging tf-idf scores
        of words in the sentences.

        Params:
            topic_name (str): name of the topic to which the corpus belongs.
            corpus (tilse.data.corpora.Corpus): A corpus.

        Returns:
            A tuple (of two numpy.array objects) containing scores for
            sentences obtained by  summing/averaging tf-idf scores for words
            in the sentences.
        """
        return Regression._compute_sum_and_avg_tfidf(corpus)

    @staticmethod
    def _compute_features_for_sent(sent, i, sum_tfidf, avg_tfidf):
        all_features = Counter([tok.lemma.lower() for tok in sent])

        num_nes = 0
        for token in sent.tokens:
            if token.ner_type != "":
                num_nes += 1

        all_features.update({
            "len": len(sent.tokens),
            "number_nes": num_nes,
            "sum_tfidf": sum_tfidf[i],
            "avg_tfidf": avg_tfidf[i]
        })

        return all_features

    @staticmethod
    def _compute_sum_and_avg_tfidf(corpus):
        vectorizer = textacy.vsm.Vectorizer(tf_type="linear", apply_idf=True, idf_type="smooth")
        sent_term_matrix = vectorizer.fit_transform(
            ([str(tok).lower() for tok in sent] for doc in corpus for sent in doc))

        tf_idf_sum = sent_term_matrix.sum(axis=1).A1
        nonzero_entries = numpy.diff(sent_term_matrix.indptr)

        return tf_idf_sum, tf_idf_sum / nonzero_entries
