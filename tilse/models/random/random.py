import random

from tilse.data import timelines
from tilse.models import models
from tilse.models import post_processing


class Random(models.Model):
    """
    Predicts timelines by choosing sentences at random.

    Sentences are selected randomly respecting the constraints of maximum
    number of days in a timeline and maximum daily summary length.
    """

    def train(self, corpora, preprocessed_information, reference_timelines,
              topic_to_evaluate):
        """
        Dummy training, just returns None.

        Params:
            corpora (dict(str, tilse.data.corpora.Corpus)): A mapping of topic names to corresponding corpora.
            preprocessed_information (object): Arbitrary information obtained from preprocessing.
            reference_timelines (dict(str, tilse.data.timelines.Groundtruth)): A mapping of topic names
                to corresponding reference timelines.
            topic_to_evaluate (str): The topic to evaluate (must be a key in `corpora`. The given topic will not
                be used during training (such that it can serve as evaluation data later).

        Returns:
            None
        """

        return None

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
        sents = [sent for doc in corpus for sent in doc]

        random.shuffle(sents)

        post_processed = post_processing.post_process(
            sents,
            None,
            timeline_properties.daily_summary_length,
            timeline_properties.num_dates,
            timeline_properties.start,
            timeline_properties.end
        )

        return post_processed


    def preprocess(self, topic_name, corpus):
        """
        Dummy preprocessing, just returns None.

        Params:
            topic_name (str): name of the topic to which the corpus belongs.
            corpus (tilse.data.corpora.Corpus): A corpus.

        Returns:
            None
        """

        return None