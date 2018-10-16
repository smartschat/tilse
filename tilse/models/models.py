import logging
import random

from tilse.data import timelines
from tilse.evaluation import dates
from tilse.evaluation import scores
from tilse.models.timeline_properties import TimelineProperties
from tilse.util.import_helper import import_helper


class Model:
    """
    Manages a machine learning model for timeline summarization.
    
    In particular, provides functionality for model-specific preprocessing, training and
    prediction. This class is abstract, actual models should should inherit from this class and 
    implement the preprocessing, training and prediction methods.
    
    Attributes:
        summary_length_assesor (function(Groundtruth, int)): A function to assess length of
            daily summaries given a reference groundtruth.
        sentence_representation_computer (tilse.representations.sentence_representations.SentenceRepresentation): A model for computing sentence
            representations.
        rouge (pyrouge.Rouge155 or tilse.evaluation.rouge.RougeReimplementation): An object for computing ROUGE scores.
    """
    def __init__(self, config, rouge):
        """
        Initializes a machine learning model for timeline summarization.
    
        Params:
            config (dict): A configuration dict. needs to have at least entries for `"assess_length"` 
                (should be a function from `tilse.models.assess_length` and for `"sentence_representations"`
                (should be a class from `tilse.representations.sentence_representations`).
            rouge (pyrouge.Rouge155 or tilse.evaluation.rouge.RougeReimplementation): An object for computing ROUGE scores.
            
        Returns:
            A model for timeline summarization intialized with the above parameters.
        """
        self.summary_length_assessor = import_helper("tilse.models.assess_length", config["assess_length"])
        self.sentence_representation_computer = import_helper("tilse.representations.sentence_representations",
                                                              config["sentence_representation"])

        self.rouge = rouge
        random.seed(23)

    def run(self, corpora, reference_timelines):
        """
        Runs the model on a set of corpora and evaluates w.r.t reference timelines.
        
        Trains and evaluates model using topic-based cross-validation.
    
        Params:
            corpora (dict(str, tilse.data.corpora.Corpus)): A mapping of topic names to corresponding corpora.
            reference_timelines (dict(str, tilse.data.timelines.Groundtruth)): A mapping of topic names
                to corresponding reference timelines.
            
        Returns:
            A tuple consisting of predicted timelines (dict(str, tilse.data.timelines.Timeline), naming in the 
            dict by topic name and integer identifiers), and of corresponding evaluation
            results (of type tilse.evaluation.scores.Scores).
        """
    
        topics = sorted(list(corpora.keys()))
        results_rouge = {}
        results_date = {}
        returned_timelines = {}

        # preprocess
        topic_to_preprocessed = {}
        for t in topics:
            topic_to_preprocessed[t] = self.preprocess(t, corpora[t])

        for t in topics:
            new_results_rouge, new_results_date, new_returned_timelines = self._run_for_one(t, corpora,
                                                                                            topic_to_preprocessed,
                                                                                            reference_timelines)
            results_rouge.update(new_results_rouge)
            results_date.update(new_results_date)
            returned_timelines.update(new_returned_timelines)

        return returned_timelines, scores.Scores(results_rouge, results_date, beta=self.rouge.beta)

    def _run_for_one(self, t, corpora, topic_to_preprocessed, reference_timelines):
        logging.info(t)
        corpus = corpora[t]

        # train
        params = self.train(corpora, topic_to_preprocessed, reference_timelines, t)

        results_rouge = {}
        results_date_selection = {}
        returned_timelines = {}

        # predict
        for i, timeline in enumerate(reference_timelines[t].timelines):
            timeline_properties = self.get_timeline_properties(timeline)
            groundtruth = timelines.GroundTruth([timeline])

            pred = self.predict(corpus, topic_to_preprocessed[t], timeline_properties, params)

            # evaluate
            results_rouge[t + "_" + str(i)] = self.rouge.evaluate_all(pred, groundtruth)
            results_date_selection[t + "_" + str(i)] = dates.evaluate_dates(pred, groundtruth)
            returned_timelines[t + "_" + str(i)] = pred

        return results_rouge, results_date_selection, returned_timelines

    def get_timeline_properties(self, timeline):
        """
        Computes timeline properties for a given timeline.
        
        Params:
            timeline (tilse.data.timelines.Timeline): A timeline.
            
        Returns:
            A tilse.models.timeline_properties.TimelineProperties object,
            with:
                * `daily_summary_length` set to the output of `self.asses_length`,
                * `num_dates` set to the length of the input timeline in days,
                * `num_sentences` set to the length of the input timeline in sentences,
                * `start` and `end` set to the first and last days in the input timeline.
        """
        groundtruth = timelines.GroundTruth([timeline])

        groundtruth_dates = sorted(list(timeline.get_dates()))

        desired_timeline_length = len(groundtruth_dates)

        summary_length = self.summary_length_assessor(groundtruth)

        timeline_properties = TimelineProperties(
            summary_length,
            desired_timeline_length,
            timeline.get_number_of_sentences(),
            groundtruth_dates[0],
            groundtruth_dates[-1]
        )

        return timeline_properties

    def train(self, corpora, preprocessed_information, reference_timelines, topic_to_evaluate):
        """
        Trains the model. Needs to be implemented by subclasses.
    
        Params:
            corpora (dict(str, tilse.data.corpora.Corpus)): A mapping of topic names to corresponding corpora.
            preprocessed_information (object): Arbitrary information obtained from preprocessing.
            reference_timelines (dict(str, tilse.data.timelines.Groundtruth)): A mapping of topic names
                to corresponding reference timelines.
            topic_to_evaluate (str): The topic to evaluate (must be a key in `corpora`. The given topic will not
                be used during training (such that it can serve as evaluation data later).
            
        Returns:
            Arbitrary information specifying results of training.
        """
        
        raise NotImplementedError("Needs to be implemented by subclass.")

    def predict(self, corpus, preprocessed_information, timeline_properties, params):
        """
        Predicts a timeline. Needs to be implemented by subclasses.
            
        Params:
            corpus (tilse.data.corpora.Corpus): A corpus.
            preprocessed_information (object): Arbitrary information obtained from preprocessing.
            timeline_properties (tilse.models.timeline_properties.TimelineProperties): Properties of the timeline to
                predict.
            params (object): Arbitrary information obtained from training.
            
        Returns:
            A timeline (tilse.data.timelines.Timeline).
        """    
        
        raise NotImplementedError("Needs to be implemented by subclass.")

    def preprocess(self, topic_name, corpus):
        """
        Preprocesses a corpus. Needs to be implemented by subclasses.
    
        Params:
            topic_name (str): name of the topic to which the corpus belongs.
            corpus (tilse.data.corpora.Corpus): A corpus.
            
        Returns:
            Arbitrary preprocessed information.
        """    
        
        raise NotImplementedError("Needs to be implemented by subclass.")
