def evaluate_dates(pred, groundtruth):
    """
    Evaluates date prediction in terms of recall, precision
    and F1 score.

    Returns the evaluation as a mapping of the strings
    "precision", "recall" and "f_score" to the corresponding
    evaluation results.

    Params:
        pred (Timeline): Predicted timeline.
        groundtruth (Groundtruth): Reference timelines.

    Returns:
        A mapping containing the evaluation results, as described above.
    """
    pred_dates = pred.get_dates()
    ref_dates = groundtruth.get_dates()

    in_both = pred_dates.intersection(ref_dates)

    prec = len(in_both) / len(pred_dates)
    rec = len(in_both) / len(ref_dates)

    if prec + rec == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)

    return {
        "precision": prec,
        "recall": rec,
        "f_score": f1
    }
