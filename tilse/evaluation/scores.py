from __future__ import division

from tilse.evaluation import util


class Scores:
    """" Store, process and display scores of predicted timelines evaluated
    against sets of reference timelines.

    In particular, this class supports nice printing of results and automated computation
    of average scores.

    Attributes:
        mapping (dict(str, dict(str, dict(str, str)))): A mapping of timeline/corpus identifiers to dicts
            containing scores for the corresponding timelines. One example entry is
            {
                "bpoil": {
                    "rouge_1": {"precision": 1.0, "recall": 1.0, "f_score": 1.0},
                }
                "mj": {
                    "rouge_1": {"precision": 1.0, "recall": 1.0, "f_score": 1.0},
                }
                "average_score": {
                    "rouge_1": {"precision": 1.0, "recall": 1.0, "f_score": 1.0},
                }
            }
        alpha (float): Alpha value to compute the average F_alpha score.
    """
    def __init__(self, scores_mapping, beta=1):
        """" Initialize a scores object and compute average of scores..

        Args:
            mapping (dict(str, dict(str, dict(str, str)))): A mapping of timeline/corpus identifiers to dicts
                containing scores for the corresponding timelines. One example entry is
                {
                    "bpoil": {
                        "rouge_1": {"precision": 1.0, "recall": 1.0, "f_score": 1.0},
                    }
                    "mj": {
                        "rouge_1": {"precision": 1.0, "recall": 1.0, "f_score": 1.0},
                    }
                }
                This mapping is then extended with the average score over all timelines (accessible via
                "average_score").
            alpha (float): Alpha value to compute the average F_alpha score.

        """
        self.mapping = scores_mapping
        self.beta = beta

        self._compute_average()

    def _compute_average(self):
        topics = sorted(list(self.mapping.keys()))
        modes = self.mapping[list(topics)[0]].keys()

        self.mapping["average_score"] = {}

        for mode in modes:
            self.mapping["average_score"][mode] = {}
            for measure in self.mapping[list(topics)[0]][mode]:
                self.mapping["average_score"][mode][measure] = {}

                mode_scores = [self.mapping[t][mode][measure] for t in topics]
                self.mapping["average_score"][mode][measure]["precision"] = \
                    sum([s["precision"] for s in mode_scores]) / len(mode_scores)
                self.mapping["average_score"][mode][measure]["recall"] = \
                    sum([s["recall"] for s in mode_scores]) / len(mode_scores)

                self.mapping["average_score"][mode][measure]["f_score"] = util.get_f_score(
                    self.mapping["average_score"][mode][measure]["precision"],
                    self.mapping["average_score"][mode][measure]["recall"],
                    beta=self.beta
                )

    def __str__(self):
        output = ""

        topics = sorted(list(self.mapping.keys()))
        modes = self.mapping[list(topics)[0]].keys()

        for mode in sorted(modes):
            for measure in sorted(self.mapping[list(topics)[0]][mode]):
                output += measure + "\n" + mode + "\n"

                for topic in topics:
                    if topic != "average_score":
                        output += "\t" + \
                              topic.ljust(15) + \
                              "P: " + \
                              "%.3f" % self.mapping[topic][mode][measure]["precision"] + \
                              " R: " + \
                              "%.3f" % self.mapping[topic][mode][measure]["recall"] + \
                              " F: " + \
                              "%.3f" % self.mapping[topic][mode][measure]["f_score"] + \
                              "\n"

                output += "\t-------------\n"

                output += "\t" + \
                      "average_score".ljust(15) + \
                      "P: " + \
                      "%.3f" % self.mapping["average_score"][mode][measure]["precision"] + \
                      " R: " + \
                      "%.3f" % self.mapping["average_score"][mode][measure]["recall"] + \
                      " F: " + \
                      "%.3f" % self.mapping["average_score"][mode][measure]["f_score"] + \
                      "\n\n"

        return output