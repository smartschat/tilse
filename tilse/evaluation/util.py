from __future__ import division

from collections import Counter


def get_f_score(prec, rec, beta=1):
    """ Compute F_beta score.

    The formula is F_beta = (1+beta**2)*prec*rec/(beta*prec + rec)

    Args:
        prec (float): Precision
        rec (float): Recall
        beta (float): Weighting of precision

    Returns:
        The F_beta score.
    """
    if prec + rec == 0:
        return 0
    else:
        return (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec)


def compute_rouge_approximation(pred_summary, groundtruth):
    """ Compute an approximation of ROUGE-1 score for a sentence w.r.t. a set
    of summaries.

    Args:
        pred_summary (list(str)): A list of sentences. Each sentence is represented
            as a string with tokens separated by space.
        groundtruth (collection(list(str)): A collection of reference summaries. Each summary is
            a list of sentence where each sentence is represented
            as a string with tokens separated by space.

    Returns:
        An approximation of ROUGE-1 F1 score for `pred_summary` w.r.t. `groundtruth`
    """
    pred_counts = Counter()
    for sent in pred_summary:
        pred_counts.update(sent.split())

    ref_counts = {}

    for i, summary in enumerate(groundtruth):
        ref_counts[i] = Counter()
        for sent in summary:
            ref_counts[i].update(Counter(sent.split()))

    # approximate ROUGE-1 score
    match = 0
    for tok in pred_counts:
        match += sum([min(pred_counts[tok], ref_counts[x][tok]) for x in ref_counts.keys()])

    prec_denom = (len(ref_counts.keys()) * sum(pred_counts.values()))

    if prec_denom == 0:
        precision = 0
    else:
        precision = match / prec_denom

    recall_denom = sum([sum(ref_counts[x].values()) for x in ref_counts])

    if recall_denom == 0:
        recall = 0
    else:
        recall = match / recall_denom

    if precision + recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)