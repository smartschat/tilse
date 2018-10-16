"""
Contains functions for assessing length of daily summaries for
predicting timelines.

The functions take as input a groundtruth object (representing
reference timelines) and output a number.
"""

import math


def constant_length_2(groundtruth):
    """
    Always returns 2.

    Params:
        groundtruth (Groundtruth): Reference timelines.

    Returns:
        The number 2.
    """
    return 2


def average_length_in_sentences(groundtruth):
    """
    Returns the average length of all daily summaries (in sentences).

    The average ist first computed over all summaries in each timeline
    in `groundtruth`. Then the average over all averages obtained
    in this way is computed.

    Params:
        groundtruth (Groundtruth): Reference timelines.

    Returns:
        Average daily sumamry length.
    """
    all_avgs = []
    for tl in groundtruth.timelines:
        all_avgs.append(sum(len(x) for x in tl.dates_to_summaries.values()) / len(tl.get_dates()))

    return math.floor(sum(all_avgs) / len(all_avgs))


def max_length_in_sentences(groundtruth):
    """
    Returns maximum length over all daily summaries (in sentences).

    Params:
        groundtruth (Groundtruth): Reference timelines.

    Returns:
        Maximum daily sumamry length.
    """
    all_maxs = []
    for tl in groundtruth.timelines:
        all_maxs.append(max(len(x) for x in tl.dates_to_summaries.values()))

    return max(all_maxs)
